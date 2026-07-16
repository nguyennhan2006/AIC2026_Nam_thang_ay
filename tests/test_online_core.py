"""Dependency-free regression tests for the complete Online V1 core."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from online.adapters.bm25 import LexicalRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import HashingTextEncoder
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.vector_stores import (
    InMemoryVectorStore,
    QdrantVectorStore,
    qdrant_point_id,
)
from online.domain.models import (
    Candidate,
    Modality,
    SearchHit,
    SearchRequest,
    SearchFilters,
    TaskType,
)
from online.services.fusion import weighted_rrf
from online.services.query_planner import RuleBasedQueryPlanner
from online.services.search import SearchService
from online.services.temporal import link_event_hits


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSONL = ROOT / "examples" / "scenes.jsonl"


def run(coro):
    return asyncio.run(coro)


class OnlineCoreTests(unittest.TestCase):
    def test_repository_projects_nested_datasection_scene(self) -> None:
        repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        scene = run(repository.get("L01_V001_S0003"))
        self.assertIsNotNone(scene)
        self.assertIn("Gừng cay muối mặn", scene.ocr_texts[0])
        self.assertEqual(scene.keyframe_ids, ["L01_V001_S0003_F000600"])

    def test_query_planner_splits_ordered_events_and_boosts_ocr(self) -> None:
        request = SearchRequest(
            query=(
                'Người cào muối, sau đó đoàn người vẫy tay, cuối cùng trước '
                'căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"'
            ),
            task=TaskType.SEQUENCE,
        )
        plan = run(RuleBasedQueryPlanner().plan(request))
        self.assertEqual(len(plan.events), 3)
        self.assertGreater(plan.modality_weights[Modality.OCR], 1.0)

    def test_weighted_rrf_uses_rank_and_modality_weight(self) -> None:
        visual = [
            Candidate(
                entity_id="s1", scene_id="s1", video_id="v1",
                source="dense", modality=Modality.VISUAL, score=0.8, rank=1
            )
        ]
        ocr = [
            Candidate(
                entity_id="s2", scene_id="s2", video_id="v1",
                source="ocr", modality=Modality.OCR, score=10.0, rank=1,
                payload={"matched_text": "exact phrase"}
            )
        ]
        fused = weighted_rrf(
            [visual, ocr],
            {Modality.VISUAL: 1.0, Modality.OCR: 2.0},
            rrf_k=60,
        )
        self.assertEqual(fused[0].scene_id, "s2")

    def test_temporal_linker_requires_same_video_and_increasing_scene(self) -> None:
        def hit(scene: int, score: float, video: str = "L01_V001") -> SearchHit:
            return SearchHit(
                scene_id=f"{video}_S{scene:04d}", video_id=video, scene_idx=scene,
                start_sec=scene * 5.0, end_sec=scene * 5.0 + 4.0, score=score
            )
        sequences = link_event_hits([[hit(1, 1.0)], [hit(2, 0.9)], [hit(3, 0.8)]])
        self.assertEqual([x.scene_idx for x in sequences[0].scenes], [1, 2, 3])
        self.assertEqual(link_event_hits([[hit(2, 1)], [hit(1, 1)]]), [])

    def test_qdrant_id_mapping_is_stable_uuid(self) -> None:
        first = qdrant_point_id("L01_V001_S0003")
        self.assertEqual(first, qdrant_point_id("L01_V001_S0003"))
        self.assertRegex(
            first,
            re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
        )

    def test_qdrant_query_api_payload_and_response_mapping(self) -> None:
        captured = {}

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps({"result": {"points": [{
                    "id": qdrant_point_id("L01_V001_S0003"),
                    "score": 0.91,
                    "payload": {"scene_id": "L01_V001_S0003", "video_id": "L01_V001"}
                }]}}).encode()

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return Response()

        store = QdrantVectorStore(
            "http://qdrant:6333", "aic_scenes_v1", "visual", timeout_sec=3
        )
        with patch("online.adapters.vector_stores.urlopen", fake_urlopen):
            results = run(store.search(
                [0.1, 0.2], limit=5,
                filters=SearchFilters(video_ids=["L01_V001"], has_ocr=True)
            ))
        self.assertEqual(captured["body"]["using"], "visual")
        self.assertIn("filter", captured["body"])
        self.assertEqual(results[0].scene_id, "L01_V001_S0003")
        self.assertEqual(results[0].score, 0.91)

    def test_end_to_end_kis_and_sequence(self) -> None:
        async def scenario():
            repository = await JsonlSceneRepository.load(EXAMPLE_JSONL)
            encoder = HashingTextEncoder(128)
            rows = []
            for scene in await repository.all():
                vector = await encoder.encode(" ".join(scene.captions + scene.keywords))
                rows.append((scene.scene_id, scene.video_id, vector, {
                    "scene_id": scene.scene_id, "video_id": scene.video_id,
                    "has_ocr": bool(scene.ocr_texts), "has_asr": bool(scene.asr_texts)
                }))
            retrievers = [DenseRetriever(encoder, InMemoryVectorStore(rows))]
            retrievers.extend([
                await LexicalRetriever.build(field, repository)
                for field in ("caption", "ocr", "asr", "keyword")
            ])
            service = SearchService(repository, retrievers, candidate_limit=20)
            kis = await service.search(SearchRequest(
                query='căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"',
                task=TaskType.KIS, top_k=3
            ))
            sequence = await service.search(SearchRequest(
                query=("người cào muối, sau đó đoàn người vẫy tay phía sau bảng chữ, "
                       "cuối cùng đứng trước căn nhà"),
                task=TaskType.SEQUENCE, top_k=3
            ))
            return kis, sequence

        kis, sequence = run(scenario())
        self.assertEqual(kis.results[0].scene_id, "L01_V001_S0003")
        self.assertTrue(sequence.sequences)
        self.assertEqual(
            [item.scene_id for item in sequence.sequences[0].scenes],
            ["L01_V001_S0001", "L01_V001_S0002", "L01_V001_S0003"],
        )


if __name__ == "__main__":
    unittest.main()
