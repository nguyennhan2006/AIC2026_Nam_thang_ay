"""Structured KIS: manual slots route queries per branch and score softly."""

from __future__ import annotations

import asyncio
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.domain.candidate import FrameEvidence
from online.domain.models import (
    KisConstraints,
    Modality,
    QueryEvent,
    QueryPlan,
    SceneDocument,
    SearchFilters,
    SearchHit,
    SearchRequest,
    TaskType,
)
from online.domain.search_config import SearchOptions
from online.services.branch_query import get_branch_query
from online.services.kis import KisProcessor
from online.services.query_planner import RuleBasedQueryPlanner


def run(coro):
    return asyncio.run(coro)


def scene(
    scene_id: str,
    *,
    caption: str = "",
    ocr: str = "",
    asr: str = "",
    objects: list[str] | None = None,
    actions: list[str] | None = None,
    frame_text: str = "",
) -> SceneDocument:
    frame = FrameEvidence(
        keyframe_id=f"{scene_id}_F000100",
        video_id="L01_V001",
        scene_id=scene_id,
        frame_idx=100,
        timestamp_sec=3.3,
        image_path=f"processed/keyframes/L01_V001/{scene_id}.jpg",
        captions=[frame_text] if frame_text else [],
    )
    return SceneDocument(
        scene_id=scene_id,
        video_id="L01_V001",
        scene_idx=int(scene_id.rsplit("_S", 1)[1]),
        start_frame=0,
        end_frame_exclusive=300,
        start_sec=0.0,
        end_sec=10.0,
        keyframes=[frame],
        captions=[caption] if caption else [],
        ocr_texts=[ocr] if ocr else [],
        asr_texts=[asr] if asr else [],
        object_labels=objects or [],
        action_tags=actions or [],
    )


def hit(scene_id: str, *, score: float = 1.0) -> SearchHit:
    return SearchHit(
        rank=1,
        candidate_id=scene_id,
        scene_id=scene_id,
        video_id="L01_V001",
        scene_idx=int(scene_id.rsplit("_S", 1)[1]),
        start_frame=0,
        end_frame_exclusive=300,
        start_sec=0.0,
        end_sec=10.0,
        best_frame_idx=100,
        score=score,
        matched_modalities=[Modality.VISUAL],
        matched_branches=["dense_visual.raw"],
    )


class StructuredPlannerTests(unittest.TestCase):
    def test_simple_kis_keeps_branch_queries_empty(self) -> None:
        plan = run(
            RuleBasedQueryPlanner().plan(
                SearchRequest(query="người đi xe máy", task=TaskType.TEXTUAL_KIS)
            )
        )
        self.assertEqual(plan.branch_queries, {})
        self.assertEqual(plan.modality_queries, {})

    def test_structured_kis_builds_branch_queries(self) -> None:
        request = SearchRequest(
            query="debug query",
            task=TaskType.TEXTUAL_KIS,
            kis_constraints=KisConstraints(
                visual=["con đập", "trời mưa"],
                ocr=["Cấm vào"],
                asr=["nước dâng"],
                must=["cảnh cận"],
                negative=["bản đồ"],
            ),
        )
        plan = run(RuleBasedQueryPlanner().plan(request))

        self.assertEqual(plan.branch_queries["dense_visual"], "con đập trời mưa cảnh cận")
        self.assertEqual(plan.branch_queries["bm25_ocr"], "Cấm vào")
        self.assertEqual(plan.branch_queries["ocr_fuzzy"], "Cấm vào")
        self.assertEqual(plan.branch_queries["bm25_asr"], "nước dâng")
        self.assertEqual(
            plan.branch_queries["bm25_caption"],
            "con đập trời mưa cảnh cận nước dâng",
        )
        self.assertNotIn("bản đồ", " ".join(plan.branch_queries.values()))

    def test_non_kis_ignores_kis_constraints(self) -> None:
        plan = run(
            RuleBasedQueryPlanner().plan(
                SearchRequest(
                    query="câu hỏi",
                    task=TaskType.QA,
                    kis_constraints=KisConstraints(ocr=["không dùng"]),
                )
            )
        )
        self.assertEqual(plan.branch_queries, {})


class StructuredBranchQueryTests(unittest.TestCase):
    def test_helper_uses_branch_then_modality_then_fallback(self) -> None:
        plan = QueryPlan(
            task=TaskType.TEXTUAL_KIS,
            original_query="q",
            normalized_query="q",
            events=[QueryEvent(event_idx=0, text="q")],
            modality_weights={Modality.CAPTION: 1.0},
            filters=SearchFilters(),
            branch_queries={"bm25_caption": "branch text"},
            modality_queries={Modality.CAPTION: "modality text"},
        )
        self.assertEqual(
            get_branch_query(plan, "bm25_caption", Modality.CAPTION, "fallback"),
            "branch text",
        )
        self.assertEqual(
            get_branch_query(plan, "caption_dense", Modality.CAPTION, "fallback"),
            "modality text",
        )

    def test_empty_structured_branch_query_means_skip(self) -> None:
        plan = QueryPlan(
            task=TaskType.TEXTUAL_KIS,
            original_query="q",
            normalized_query="q",
            events=[QueryEvent(event_idx=0, text="q")],
            modality_weights={Modality.OCR: 1.0},
            filters=SearchFilters(),
            branch_queries={"bm25_ocr": ""},
        )
        self.assertIsNone(get_branch_query(plan, "bm25_ocr", Modality.OCR, "fallback"))

    def test_lexical_retriever_skips_empty_structured_query(self) -> None:
        docs = [scene("L01_V001_S0000", ocr="Cấm vào")]
        retriever = run(LexicalRetriever.build("ocr", _Repo(docs)))
        plan = QueryPlan(
            task=TaskType.TEXTUAL_KIS,
            original_query="debug",
            normalized_query="debug",
            events=[QueryEvent(event_idx=0, text="debug")],
            modality_weights={Modality.OCR: 1.0},
            filters=SearchFilters(),
            search_options=SearchOptions(),
            branch_queries={"bm25_ocr": ""},
        )
        self.assertEqual(run(retriever.search(plan, limit=10)), [])


class StructuredKisProcessorTests(unittest.TestCase):
    def test_constraints_boost_scene_matching_visual_asr_and_must(self) -> None:
        good = scene(
            "L01_V001_S0000",
            caption="một con đập dưới trời mưa",
            asr="người dẫn nói nước dâng nhanh",
            objects=["dam"],
            actions=["rain"],
            frame_text="cảnh cận con đập",
        )
        weak = scene(
            "L01_V001_S0001",
            caption="người dẫn nói nước dâng nhanh",
            asr="nước dâng nhanh",
        )
        ranked = KisProcessor().rank(
            "con đập trời mưa nước dâng cảnh cận",
            [hit(good.scene_id, score=1.0), hit(weak.scene_id, score=1.0)],
            {good.scene_id: good, weak.scene_id: weak},
            constraints=KisConstraints(
                visual=["con đập", "trời mưa"],
                asr=["nước dâng"],
                must=["cảnh cận"],
            ),
        )
        self.assertEqual(ranked[0].scene_id, good.scene_id)

    def test_negative_constraint_penalizes_but_does_not_filter(self) -> None:
        clean = scene("L01_V001_S0000", caption="người đi xe máy")
        bad = scene("L01_V001_S0001", caption="người đi xe máy cạnh ô tô")
        ranked = KisProcessor().rank(
            "người đi xe máy",
            [hit(bad.scene_id, score=1.0), hit(clean.scene_id, score=0.9)],
            {clean.scene_id: clean, bad.scene_id: bad},
            constraints=KisConstraints(visual=["xe máy"], negative=["ô tô"]),
        )
        self.assertEqual({item.scene_id for item in ranked}, {clean.scene_id, bad.scene_id})
        self.assertEqual(ranked[0].scene_id, clean.scene_id)

    def test_constraints_none_keeps_old_order_for_equal_inputs(self) -> None:
        first = scene("L01_V001_S0000", caption="người đi xe máy")
        second = scene("L01_V001_S0001", caption="người đi xe máy")
        ranked = KisProcessor().rank(
            "người đi xe máy",
            [hit(second.scene_id, score=1.0), hit(first.scene_id, score=1.0)],
            {first.scene_id: first, second.scene_id: second},
        )
        self.assertEqual([item.scene_id for item in ranked], [second.scene_id, first.scene_id])


class _Repo:
    def __init__(self, docs: list[SceneDocument]) -> None:
        self._docs = docs

    async def all(self) -> list[SceneDocument]:
        return self._docs


if __name__ == "__main__":
    unittest.main()
