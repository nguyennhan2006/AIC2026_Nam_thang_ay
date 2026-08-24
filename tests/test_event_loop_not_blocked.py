"""FB-001: nhánh retrieval không được khoá event loop.

Ở corpus thi đấu (873 video / 87.742 scene / 176.707 keyframe) mỗi nhánh quét
toàn bộ corpus. Khi phần quét đó chạy THẲNG trên event loop thì suốt quãng
đó server không làm được gì khác: `/v1/health` không trả lời, ảnh keyframe
không được phục vụ, và truy vấn của người thứ hai nằm chờ hết truy vấn của
người thứ nhất. Với một đội cùng dùng chung một server, đó chính là triệu
chứng "load lâu" — chứ không phải một truy vấn đơn lẻ chậm đi.

Các test dưới đây đo đúng thứ đó: trong lúc `search()` đang chạy, một coroutine
nhịp tim có được chạy hay không. Chạy ngược lại các adapter trước khi sửa thì
mọi test ở đây đều fail (nhịp tim = 0 tick).
"""

from __future__ import annotations

import asyncio
import time
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.color_search import ColorSearchRetriever
from online.adapters.vector_stores import InMemoryVectorStore
from online.domain.models import (
    Modality,
    QueryEvent,
    QueryPlan,
    SceneDocument,
    SearchFilters,
    TaskType,
)
from online.domain.search_config import SearchOptions

# Đủ dài để nhịp tim 5ms tick được nhiều lần, đủ ngắn để test không lê thê.
BLOCK_SEC = 0.25
HEARTBEAT_SEC = 0.005


def plan_for(query: str = "mau do") -> QueryPlan:
    return QueryPlan(
        task=TaskType.TEXTUAL_KIS,
        original_query=query,
        normalized_query=query,
        events=[QueryEvent(event_idx=0, text=query)],
        modality_weights={
            Modality.CAPTION: 1.0,
            Modality.OCR: 1.0,
            Modality.VISUAL: 1.0,
            Modality.COLOR: 1.0,
        },
        filters=SearchFilters(),
        search_options=SearchOptions(),
    )


async def with_heartbeat(coro):
    """Chạy `coro`, đếm số nhịp mà event loop kịp phục vụ trong lúc đó.

    Đây là phép đo TRỰC TIẾP của "server còn trả lời được người khác không",
    chứ không phải đo thời gian chạy — một truy vấn chậm mà không khoá loop
    vẫn là hành vi đúng.
    """

    ticks = 0
    running = True

    async def heartbeat():
        nonlocal ticks
        while running:
            await asyncio.sleep(HEARTBEAT_SEC)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        result = await coro
    finally:
        running = False
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass
    return result, ticks


class SlowIndex:
    """Đứng thay `BM25Index`: `time.sleep` mô phỏng quét corpus thật.

    `time.sleep` nhả GIL đúng như numpy và như phần lớn thời gian của một vòng
    quét Python dài, nên nếu adapter đã đẩy sang thread thì event loop chạy
    tiếp được; còn nếu gọi thẳng thì loop đứng.
    """

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, limit: int):
        self.calls += 1
        time.sleep(BLOCK_SEC)
        return []


def scene(scene_id: str, colors: list[str]) -> SceneDocument:
    return SceneDocument(
        scene_id=scene_id,
        video_id=scene_id.rsplit("_S", 1)[0],
        scene_idx=0,
        start_frame=0,
        end_frame_exclusive=10,
        start_sec=0.0,
        end_sec=1.0,
        captions=["caption"],
        color_names=colors,
    )


class LexicalOffLoopTest(unittest.TestCase):
    def test_bm25_search_khong_khoa_event_loop(self):
        index = SlowIndex()
        retriever = LexicalRetriever("caption", index, None)

        async def scenario():
            return await with_heartbeat(retriever.search(plan_for(), limit=5))

        _result, ticks = asyncio.run(scenario())
        self.assertEqual(index.calls, 1, "phải thật sự gọi index, không đi đường tắt")
        self.assertGreater(
            ticks, 0,
            "event loop bị khoá suốt lúc BM25 quét — cả server đứng hình với mọi người",
        )


class VectorStoreOffLoopTest(unittest.TestCase):
    """Vector store thật, chỉ làm phép nhân chậm lại bằng một `_score` giả."""

    def test_dense_search_khong_khoa_event_loop(self):
        store = InMemoryVectorStore([
            ("s1", "L01_V001_S000", [1.0, 0.0], {"scene_id": "L01_V001_S000", "video_id": "L01_V001"}),
        ])
        original = store._score

        def slow_score(vector, selected):
            time.sleep(BLOCK_SEC)
            return original(vector, selected)

        store._score = slow_score

        async def scenario():
            return await with_heartbeat(
                store.search([1.0, 0.0], limit=1, filters=SearchFilters())
            )

        _result, ticks = asyncio.run(scenario())
        self.assertGreater(
            ticks, 0,
            "event loop bị khoá suốt lúc chấm điểm cosine 176k vector",
        )


class ColorSearchOffLoopTest(unittest.TestCase):
    def test_color_search_khong_khoa_event_loop(self):
        documents = [scene(f"L01_V001_S{index:03d}", ["red"]) for index in range(3)]
        retriever = ColorSearchRetriever(documents)
        original = retriever._search_sync

        def slow(*args, **kwargs):
            time.sleep(BLOCK_SEC)
            return original(*args, **kwargs)

        retriever._search_sync = slow

        async def scenario():
            return await with_heartbeat(retriever.search(plan_for("màu đỏ"), limit=5))

        result, ticks = asyncio.run(scenario())
        self.assertTrue(result, "query có tag màu nên phải ra candidate")
        self.assertGreater(ticks, 0, "event loop bị khoá suốt lúc quét màu toàn corpus")


class TwoUsersAtOnceTest(unittest.TestCase):
    """Hai người search cùng lúc: request thứ hai không phải xếp hàng sau."""

    def test_hai_truy_van_chong_lan_chu_khong_noi_duoi(self):
        first = LexicalRetriever("caption", SlowIndex(), None)
        second = LexicalRetriever("caption", SlowIndex(), None)

        async def scenario():
            started = time.perf_counter()
            await asyncio.gather(
                first.search(plan_for(), limit=5),
                second.search(plan_for(), limit=5),
            )
            return time.perf_counter() - started

        elapsed = asyncio.run(scenario())
        # Nối đuôi thì mất 2 x BLOCK_SEC. Chồng lấn thì xấp xỉ 1 x.
        #
        # CẢNH BÁO khi đọc con số này: `time.sleep` NHẢ GIL, nên nó mô phỏng
        # đúng nhánh numpy (dense_visual, caption_dense) chứ KHÔNG mô phỏng
        # BM25 Python thuần. Test này chứng minh hai request được nhận và chạy
        # chồng lấn — không chứng minh hai lượt BM25 chạy song song thật.
        self.assertLess(
            elapsed, BLOCK_SEC * 1.6,
            f"hai truy vấn vẫn nối đuôi nhau ({elapsed:.2f}s cho 2 x {BLOCK_SEC}s)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
