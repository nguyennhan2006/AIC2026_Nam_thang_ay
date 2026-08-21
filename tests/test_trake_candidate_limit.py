"""TRAKE phải lấy candidate rộng hơn ba task kia — và lý do là cấu trúc.

`link_event_hits` chỉ dựng được chuỗi khi **mọi step có candidate trong CÙNG một
video**, nhưng mỗi step lại lấy top-K trên TOÀN corpus. Hai điều đó cộng lại tạo
ra một hỏng hóc chỉ lộ ra khi corpus đủ lớn: K slot rải trên V video, số video
có mặt ở cả n step tụt xuống rất nhanh theo V.

Mô phỏng 873 video, 3 step, số video TRAKE trả về::

    K=100 -> 0      K=200 -> 2      K=500 -> 13     K=1000 -> 13

Ở K=100 (mặc định deployment) TRAKE trả về đúng một video, và đó thường là video
"nam châm" — dài, cùng chủ đề, caption dày — chứ không phải video đáp án. Trên
corpus 3 video thì giao = 3/3 nên không có gì để hỏng, đó là lý do lỗi này sống
sót qua mọi lần đo trước.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest, TaskType
from online.services.search import SearchService
from scripts.seed_demo import main as seed


def run(coro):
    return asyncio.run(coro)


class TrakeCandidateLimitTests(unittest.TestCase):
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
            return SearchService(repository, retrievers, candidate_limit=20, **kwargs)
        return run(build())

    def _limits_used(self, service: SearchService) -> list[int]:
        """Ghi lại K mà mỗi lần retrieval thực sự dùng."""

        seen: list[int] = []
        original = service._retrieve

        async def spy(plan, limit):
            seen.append(limit)
            return await original(plan, limit)

        service._retrieve = spy
        run(service.search(SearchRequest(
            query="cào muối, sau đó vẫy tay, cuối cùng đứng trước căn nhà",
            task=TaskType.TRAKE, top_k=3,
        )))
        return seen

    def test_without_override_trake_uses_the_shared_limit(self) -> None:
        self.assertEqual(set(self._limits_used(self._service())), {20})

    def test_override_widens_only_the_trake_path(self) -> None:
        limits = self._limits_used(self._service(trake_candidate_limit=200))
        self.assertEqual(set(limits), {200})
        # Một lần retrieval MỖI STEP — đây là lý do không nâng `candidate_limit`
        # chung: chi phí nhân với số step.
        self.assertGreaterEqual(len(limits), 2)

    def test_kis_is_left_alone_by_the_override(self) -> None:
        """Cùng service, task khác: KIS vẫn phải chạy ở K chung."""

        service = self._service(trake_candidate_limit=200)
        seen: list[int] = []
        original = service._retrieve

        async def spy(plan, limit):
            seen.append(limit)
            return await original(plan, limit)

        service._retrieve = spy
        run(service.search(SearchRequest(
            query="cào muối", task=TaskType.TEXTUAL_KIS, top_k=3,
        )))
        self.assertEqual(set(seen), {20})


if __name__ == "__main__":
    unittest.main()
