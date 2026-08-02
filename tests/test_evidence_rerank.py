"""PR-06: evidence pack đầy đủ + rerank cascade degrade tường minh.

Hai rủi ro cụ thể các test này chặn:

1. Reranker lỗi làm rỗng hoặc xáo loạn kết quả retrieval vốn đang đúng.
2. Reranker trả kết quả *một phần* (thiếu candidate) mà pipeline vẫn dùng —
   thứ hạng sẽ lệch âm thầm.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import Candidate, Modality
from online.domain.search_config import (
    RerankOptions,
    TextRerankOptions,
    VlmRerankOptions,
)
from online.errors import DependencyUnavailableError
from online.services.evidence_builder import EvidenceBuilder
from online.services.rerank_pipeline import RerankPipeline

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSONL = ROOT / "examples" / "scenes.jsonl"


def run(coro):
    return asyncio.run(coro)


def candidate(scene_id: str, *, rank: int = 1, score: float = 1.0) -> Candidate:
    return Candidate(
        candidate_id=scene_id,
        scene_id=scene_id,
        video_id=scene_id.rsplit("_S", 1)[0],
        source="fusion_rrf",
        modality=Modality.VISUAL,
        raw_score=score,
        rank=rank,
        payload={
            "component_scores": {"bm25_ocr.raw": 8.4},
            "branch_contributions": {"bm25_ocr.raw": 0.016},
            "matched_branches": ["bm25_ocr.raw"],
            "rule_adjustments": {"ocr_exact": 0.02},
        },
    )


class EvidencePackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        self.builder = EvidenceBuilder(self.repository, dataset_version="fixture-v1")

    def test_pack_carries_frames_scores_and_provenance(self) -> None:
        pack = run(self.builder.build(candidate("L01_V001_S0003")))
        self.assertIsNotNone(pack)
        self.assertEqual(pack.video_id, "L01_V001")
        self.assertEqual(pack.start_frame, 600)
        self.assertTrue(pack.keyframes)
        self.assertIn("Gừng cay muối mặn", pack.ocr_text)
        self.assertIn("bm25_ocr.raw", pack.branch_contributions)
        self.assertEqual(pack.rule_adjustments[0].rule, "ocr_exact")
        self.assertEqual(pack.dataset_version, "fixture-v1")

    def test_neighbor_context_is_the_adjacent_scene(self) -> None:
        pack = run(self.builder.build(candidate("L01_V001_S0002")))
        self.assertEqual(pack.previous_context.scene_id, "L01_V001_S0001")
        self.assertEqual(pack.next_context.scene_id, "L01_V001_S0003")

    def test_first_scene_has_no_previous_context(self) -> None:
        pack = run(self.builder.build(candidate("L02_V001_S0001")))
        # scene_idx = 1 nên vẫn tra S0000, không tồn tại -> None chứ không lỗi.
        self.assertIsNone(pack.previous_context)

    def test_rerank_text_is_compact_not_raw_json(self) -> None:
        pack = run(self.builder.build(candidate("L01_V001_S0003")))
        text = pack.rerank_text()
        self.assertIn("Caption:", text)
        self.assertIn("OCR:", text)
        self.assertNotIn("{", text)
        self.assertLessEqual(len(text), 1200)

    def test_unknown_scene_returns_none_instead_of_raising(self) -> None:
        self.assertIsNone(run(self.builder.build(candidate("L99_V999_S0000"))))


class FakeTextReranker:
    def __init__(self, scores: list[float] | None = None, fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.calls = 0

    async def score(self, query: str, packs: list) -> list[float]:
        self.calls += 1
        if self.fail:
            raise DependencyUnavailableError("rerank service unavailable: connection refused")
        if self.scores is not None:
            return self.scores[: len(packs)]
        return [float(len(packs) - index) for index in range(len(packs))]


class FakeVlmReranker:
    def __init__(self, relevance: list[float] | None = None, fail: bool = False) -> None:
        self.relevance = relevance
        self.fail = fail

    async def score(self, query: str, packs: list) -> list[dict]:
        if self.fail:
            raise DependencyUnavailableError("VLM reranker bỏ sót candidate")
        values = self.relevance or [0.5] * len(packs)
        return [
            {
                "candidate_id": pack.candidate_id,
                "relevance": values[index],
                "evidence_summary": "ok",
            }
            for index, pack in enumerate(packs)
        ]


class RerankCascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        self.builder = EvidenceBuilder(self.repository)
        self.candidates = [
            candidate("L01_V001_S0001", rank=1, score=0.9),
            candidate("L01_V001_S0002", rank=2, score=0.8),
            candidate("L01_V001_S0003", rank=3, score=0.7),
        ]

    def _options(self, **kwargs) -> RerankOptions:
        defaults = {
            "text": TextRerankOptions(enabled=False),
            "vlm": VlmRerankOptions(enabled=False),
        }
        defaults.update(kwargs)
        return RerankOptions(**defaults)

    def test_no_reranker_configured_is_a_passthrough_with_a_warning(self) -> None:
        pipeline = RerankPipeline(self.builder)
        outcome = run(pipeline.run("q", self.candidates, self._options(
            text=TextRerankOptions(enabled=True)
        )))
        self.assertEqual([item.scene_id for item in outcome.candidates],
                         [item.scene_id for item in self.candidates])
        self.assertTrue(any("AIC_RERANK_TEXT_URL" in item for item in outcome.warnings))

    def test_text_reranker_reorders_and_keeps_the_tail(self) -> None:
        # Điểm đảo ngược thứ tự fusion.
        pipeline = RerankPipeline(self.builder, text_reranker=FakeTextReranker([0.1, 0.2, 0.9]))
        outcome = run(pipeline.run("q", self.candidates, self._options(
            text=TextRerankOptions(enabled=True, input_top_k=3, output_top_k=2)
        )))
        self.assertEqual(
            [item.scene_id for item in outcome.candidates],
            ["L01_V001_S0003", "L01_V001_S0002", "L01_V001_S0001"],
        )
        # output_top_k cắt cái gì thì cái đó tụt xuống chứ không bị vứt —
        # giữ recall cho các mốc chấm 50/100.
        self.assertEqual(len(outcome.candidates), 3)
        self.assertEqual([item.rank for item in outcome.candidates], [1, 2, 3])

    def test_failed_text_reranker_keeps_fusion_order_and_warns(self) -> None:
        pipeline = RerankPipeline(self.builder, text_reranker=FakeTextReranker(fail=True))
        outcome = run(pipeline.run("q", self.candidates, self._options(
            text=TextRerankOptions(enabled=True)
        )))
        self.assertEqual(
            [item.scene_id for item in outcome.candidates],
            [item.scene_id for item in self.candidates],
        )
        self.assertTrue(any("giữ nguyên thứ hạng" in item for item in outcome.warnings))
        self.assertFalse(outcome.stages[0].applied)

    def test_vlm_verdict_is_attached_to_the_candidate(self) -> None:
        pipeline = RerankPipeline(self.builder, vlm_reranker=FakeVlmReranker([0.2, 0.9, 0.1]))
        outcome = run(pipeline.run("q", self.candidates, self._options(
            vlm=VlmRerankOptions(enabled=True, input_top_k=3)
        )))
        self.assertEqual(outcome.candidates[0].scene_id, "L01_V001_S0002")
        self.assertEqual(outcome.candidates[0].payload["vlm_verdict"]["evidence_summary"], "ok")

    def test_failed_vlm_falls_back_to_the_text_stage_result(self) -> None:
        pipeline = RerankPipeline(
            self.builder,
            text_reranker=FakeTextReranker([0.1, 0.2, 0.9]),
            vlm_reranker=FakeVlmReranker(fail=True),
        )
        outcome = run(pipeline.run("q", self.candidates, self._options(
            text=TextRerankOptions(enabled=True, input_top_k=3, output_top_k=3),
            vlm=VlmRerankOptions(enabled=True, input_top_k=3),
        )))
        # Thứ tự của stage text được giữ, không rơi ngược về fusion.
        self.assertEqual(outcome.candidates[0].scene_id, "L01_V001_S0003")
        self.assertTrue(any("rơi về kết quả của stage text" in item for item in outcome.warnings))

    def test_empty_input_short_circuits(self) -> None:
        reranker = FakeTextReranker()
        pipeline = RerankPipeline(self.builder, text_reranker=reranker)
        outcome = run(pipeline.run("q", [], self._options(text=TextRerankOptions(enabled=True))))
        self.assertEqual(outcome.candidates, [])
        self.assertEqual(reranker.calls, 0)

    def test_available_stages_reports_what_is_actually_wired(self) -> None:
        self.assertEqual(
            RerankPipeline(self.builder).available_stages, {"text": False, "vlm": False}
        )
        self.assertEqual(
            RerankPipeline(self.builder, text_reranker=FakeTextReranker()).available_stages,
            {"text": True, "vlm": False},
        )


if __name__ == "__main__":
    unittest.main()
