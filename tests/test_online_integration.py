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


class OnlineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed(False)
        cls.path = Path(__file__).resolve().parents[1] / "storage/exports/scenes.jsonl"

    def service(self):
        async def build():
            repository = await JsonlSceneRepository.load(self.path)
            retrievers = [await LexicalRetriever.build(field, repository) for field in ("caption", "ocr", "asr", "keyword")]
            return SearchService(repository, retrievers, candidate_limit=20)
        return run(build())

    def test_exact_ocr_returns_scene_and_best_keyframe(self) -> None:
        result = run(self.service().search(SearchRequest(query='"Gừng cay muối mặn"', task=TaskType.KIS, top_k=1)))
        hit = result.results[0]
        self.assertEqual(hit.scene_id, "L01_V001_S0002")
        self.assertEqual(hit.best_keyframe_id, "L01_V001_S0002_F000750")
        self.assertEqual(hit.best_timestamp_sec, 25.0)
        self.assertEqual(hit.video_path, "raw/videos/L01_V001.mp4")

    def test_sequence_is_same_video_and_increasing(self) -> None:
        result = run(self.service().search(SearchRequest(
            query="cào muối, sau đó vẫy tay, cuối cùng đứng trước căn nhà",
            task=TaskType.SEQUENCE, top_k=3,
        )))
        self.assertTrue(result.sequences)
        scenes = result.sequences[0].scenes
        self.assertEqual([x.scene_idx for x in scenes], [0, 1, 2])
        self.assertEqual(len({x.video_id for x in scenes}), 1)


if __name__ == "__main__":
    unittest.main()
