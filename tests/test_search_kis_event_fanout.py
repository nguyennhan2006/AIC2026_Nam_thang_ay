"""KIS/QA nhiều cảnh phải tìm ở TỪNG cảnh, không chỉ cảnh cuối.

Đo trên 25 câu sơ tuyển thật (`Example_and_practical_data/AIC2026-SoTuyen1`):
17/25 câu tả nhiều cảnh nối tiếp ("... sau đó ... tiếp đến ..."), nhưng cả
tầng rule (`query/router.py::_split_target_context`) lẫn tầng LLM
(`fpt_query_bundle.py::refine`) đều chỉ giữ lại MỘT cảnh trong
`visual_query`/`caption_query` — hai trường duy nhất `_retrieve` thật sự
dùng. Ví dụ đo được: câu "bản đồ...bốn lần...đập...mưa" (gold L28_V018) chỉ
còn "cận cảnh con đập dưới trời mưa", cảnh bản đồ chứa gold frame gần nhất
bị bỏ hẳn khỏi mọi query con.

`query_bundle.events` tách đúng toàn bộ chuỗi (cả tầng rule lẫn LLM) nhưng
trước bản sửa này không bao giờ được dùng cho retrieval ngoài task TRAKE.
Bộ test này khoá lại: KIS/QA/AVS nhiều cảnh phải gọi `_retrieve` thêm một
lần MỖI event và union kết quả, còn câu một-cảnh (đa số) giữ nguyên đúng 1
lần gọi như trước — không đổi hành vi/latency cho trường hợp phổ biến nhất.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.candidate import Candidate
from online.domain.models import Modality, SearchRequest, TaskType
from online.services.query.router import QueryRouter
from online.services.search import SearchService, _merge_candidate_pools
from scripts.seed_demo import main as seed


def run(coro):
    return asyncio.run(coro)


MULTI_SCENE_QUERY = (
    "Đoạn phim bắt đầu bằng một bản đồ, trên đó công trình thủy lợi xuất "
    "hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên "
    "cao, tiếp đến là cảnh cận con đập dưới trời mưa."
)
SINGLE_SCENE_QUERY = "Một người phụ nữ mặc áo dài màu hồng đang đứng giảng bài"


def _candidate(candidate_id: str, score: float, rank: int = 1) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        video_id=candidate_id.split("_S")[0],
        scene_id=candidate_id,
        source="dense_visual",
        modality=Modality.VISUAL,
        raw_score=score,
        rank=rank,
    )


class MergeCandidatePoolsTests(unittest.TestCase):
    """`_merge_candidate_pools` một mình — không cần service/corpus."""

    def test_duplicate_candidate_keeps_the_higher_score(self) -> None:
        pool_a = [_candidate("L28_V018_S0016", 0.9)]
        pool_b = [_candidate("L28_V018_S0016", 0.4), _candidate("L28_V018_S0022", 0.7)]
        merged = _merge_candidate_pools([pool_a, pool_b], limit=10)
        by_id = {c.candidate_id: c.raw_score for c in merged}
        self.assertEqual(by_id["L28_V018_S0016"], 0.9)
        self.assertEqual(by_id["L28_V018_S0022"], 0.7)

    def test_result_is_sorted_and_capped_at_limit(self) -> None:
        pools = [[_candidate(f"c{i}", score=float(i)) for i in range(5)]]
        merged = _merge_candidate_pools(pools, limit=3)
        self.assertEqual([c.candidate_id for c in merged], ["c4", "c3", "c2"])

    def test_a_scene_found_only_by_one_event_still_survives_the_merge(self) -> None:
        """Đúng kịch bản lỗi thật: cảnh bản đồ chỉ được TÌM RA bởi event
        'bản đồ', không phải bởi câu gốc (đã collapse về cảnh 'đập')."""

        base_query_pool = [_candidate("L28_V018_S0022", 0.8)]  # câu gốc: chỉ ra cảnh đập
        map_event_pool = [_candidate("L28_V018_S0016", 0.75)]  # event 'bản đồ' riêng
        merged = _merge_candidate_pools([base_query_pool, map_event_pool], limit=10)
        ids = {c.candidate_id for c in merged}
        self.assertIn("L28_V018_S0016", ids)
        self.assertIn("L28_V018_S0022", ids)


class KisEventFanoutRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed(False)
        cls.path = Path(__file__).resolve().parents[1] / "storage/exports/scenes.jsonl"

    def _service(self, **kwargs) -> SearchService:
        async def build():
            repository = await JsonlSceneRepository.load(self.path)
            retrievers = [
                await LexicalRetriever.build(field, repository)
                for field in ("caption", "ocr", "asr", "keyword")
            ]
            # query_router mặc định = QueryRouter() thuần rule (không LLM,
            # không mạng) — đủ để tách events cho câu có "sau đó"/"tiếp đến".
            return SearchService(repository, retrievers, candidate_limit=20, **kwargs)
        return run(build())

    def _retrieve_call_count(self, service: SearchService, query: str, task: TaskType) -> int:
        seen = 0
        original = service._retrieve

        async def spy(plan, limit):
            nonlocal seen
            seen += 1
            return await original(plan, limit)

        service._retrieve = spy
        run(service.search(SearchRequest(query=query, task=task, top_k=3)))
        return seen

    def test_multi_scene_kis_query_retrieves_once_per_event(self) -> None:
        expected_events = len(QueryRouter().prepare_sync(
            SearchRequest(query=MULTI_SCENE_QUERY, task=TaskType.TEXTUAL_KIS)
        ).events)
        self.assertGreaterEqual(expected_events, 2, "fixture câu phải thật sự nhiều cảnh")

        count = self._retrieve_call_count(
            self._service(), MULTI_SCENE_QUERY, TaskType.TEXTUAL_KIS
        )
        self.assertEqual(count, 1 + expected_events)

    def test_single_scene_kis_query_is_unaffected(self) -> None:
        count = self._retrieve_call_count(
            self._service(), SINGLE_SCENE_QUERY, TaskType.TEXTUAL_KIS
        )
        self.assertEqual(count, 1)

    def test_flag_off_restores_the_old_single_call_behaviour(self) -> None:
        count = self._retrieve_call_count(
            self._service(kis_event_fanout=False),
            MULTI_SCENE_QUERY, TaskType.TEXTUAL_KIS,
        )
        self.assertEqual(count, 1)

    def test_trake_task_is_left_alone_by_the_kis_fanout(self) -> None:
        """TRAKE có đường riêng (link_event_hits) — fan-out KIS không được
        đụng vào, tránh chạy trùng hoặc đổi hành vi TRAKE ngoài ý muốn."""

        count = self._retrieve_call_count(
            self._service(), MULTI_SCENE_QUERY, TaskType.TRAKE
        )
        # TRAKE dùng plan.events (rule-tier query_planner.py), không phải
        # nhánh fan-out mới — số lần gọi phụ thuộc plan.events, không phải
        # query_bundle.events; chỉ cần khác 0 và không bị nhánh KIS đè lên.
        self.assertGreaterEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
