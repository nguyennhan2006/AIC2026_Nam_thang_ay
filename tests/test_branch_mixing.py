"""Search Mixing Console W3/W5: per-branch overrides, new branches, fusion methods."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from online.adapters.color_search import ColorSearchRetriever, extract_color_tags
from online.adapters.event_search import EventSearchRetriever, JsonlEventRepository, project_event
from online.domain.models import Candidate, Modality, QueryEvent, QueryPlan, SceneDocument, SearchFilters, TaskType
from online.domain.search_config import BranchRuntimeOptions, FusionOptions, SearchOptions
from online.services.branch_options import effective_limit, effective_weight
from online.services.fusion import fuse_candidates


def run(coro):
    return asyncio.run(coro)


def _plan(
    *, modality_weights: dict[Modality, float], search_options: SearchOptions | None = None, query: str = "q",
) -> QueryPlan:
    return QueryPlan(
        task=TaskType.KIS, original_query=query, normalized_query=query,
        events=[QueryEvent(event_idx=0, text=query)],
        modality_weights=modality_weights, filters=SearchFilters(),
        search_options=search_options or SearchOptions(),
    )


class BranchOptionsTests(unittest.TestCase):
    def test_no_override_falls_back_to_modality_weight(self) -> None:
        plan = _plan(modality_weights={Modality.OCR: 0.35})
        self.assertEqual(effective_weight(plan, "bm25_ocr", Modality.OCR), 0.35)
        self.assertEqual(effective_limit(plan, "bm25_ocr", 100), 100)

    def test_override_enabled_false_disables_regardless_of_modality_weight(self) -> None:
        options = SearchOptions(branches={"bm25_ocr": BranchRuntimeOptions(enabled=False)})
        plan = _plan(modality_weights={Modality.OCR: 5.0}, search_options=options)
        self.assertEqual(effective_weight(plan, "bm25_ocr", Modality.OCR), 0.0)

    def test_override_weight_and_top_k_replace_modality_default(self) -> None:
        options = SearchOptions(branches={"bm25_ocr": BranchRuntimeOptions(weight=3.0, top_k=10)})
        plan = _plan(modality_weights={Modality.OCR: 0.35}, search_options=options)
        self.assertEqual(effective_weight(plan, "bm25_ocr", Modality.OCR), 3.0)
        self.assertEqual(effective_limit(plan, "bm25_ocr", 100), 10)

    def test_override_only_affects_its_own_branch_name(self) -> None:
        options = SearchOptions(branches={"ocr_fuzzy": BranchRuntimeOptions(enabled=False)})
        plan = _plan(modality_weights={Modality.OCR: 0.35}, search_options=options)
        self.assertEqual(effective_weight(plan, "bm25_ocr", Modality.OCR), 0.35)
        self.assertEqual(effective_weight(plan, "ocr_fuzzy", Modality.OCR), 0.0)


class ColorSearchTests(unittest.TestCase):
    def test_extract_color_tags_matches_vietnamese_and_english(self) -> None:
        self.assertEqual(extract_color_tags("áo đỏ và mũ vàng"), ["red", "yellow"])
        self.assertEqual(extract_color_tags("a blue car"), ["blue"])
        self.assertEqual(extract_color_tags("không có màu gì"), [])

    def _doc(self, scene_id: str, colors: list[str]) -> SceneDocument:
        return SceneDocument(
            scene_id=scene_id, video_id="L01_V001", scene_idx=0,
            start_sec=0.0, end_sec=1.0, color_names=colors,
        )

    def test_retriever_scores_by_color_overlap_ratio(self) -> None:
        retriever = ColorSearchRetriever([
            self._doc("L01_V001_S0000", ["red", "blue"]),
            self._doc("L01_V001_S0001", ["red"]),
            self._doc("L01_V001_S0002", ["green"]),
        ])
        plan = _plan(modality_weights={Modality.COLOR: 1.0}, query="đỏ")
        results = run(retriever.search(plan, limit=10))
        scene_ids = [item.scene_id for item in results]
        self.assertEqual(scene_ids, ["L01_V001_S0000", "L01_V001_S0001"])

    def test_disabled_when_modality_weight_is_zero(self) -> None:
        retriever = ColorSearchRetriever([self._doc("L01_V001_S0000", ["red"])])
        plan = _plan(modality_weights={Modality.COLOR: 0.0, Modality.VISUAL: 1.0}, query="đỏ")
        results = run(retriever.search(plan, limit=10))
        self.assertEqual(results, [])

    def test_no_color_words_in_query_returns_empty(self) -> None:
        retriever = ColorSearchRetriever([self._doc("L01_V001_S0000", ["red"])])
        plan = _plan(modality_weights={Modality.COLOR: 1.0}, query="người đàn ông")
        results = run(retriever.search(plan, limit=10))
        self.assertEqual(results, [])


class EventSearchTests(unittest.TestCase):
    def test_project_event_round_trips_required_fields(self) -> None:
        event = project_event({
            "event_id": "L01_V001_E0000", "video_id": "L01_V001",
            "scene_ids": ["L01_V001_S0000", "L01_V001_S0001"],
            "start_sec": 0.0, "end_sec": 2.0,
            "event_caption": "cào muối", "keywords": ["salt"], "action_tags": ["raking"],
            "previous_event_id": None, "next_event_id": None,
        })
        self.assertEqual(event.scene_ids, ["L01_V001_S0000", "L01_V001_S0001"])
        self.assertIn("cào muối", event.field_text())

    def test_repository_load_and_neighbor_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                '{"event_id": "L01_V001_E0000", "video_id": "L01_V001", "scene_ids": ["L01_V001_S0000"], '
                '"start_sec": 0.0, "end_sec": 1.0, "event_caption": "cào muối", "keywords": [], '
                '"action_tags": ["raking"], "previous_event_id": null, "next_event_id": "L01_V001_E0001"}\n'
                '{"event_id": "L01_V001_E0001", "video_id": "L01_V001", "scene_ids": ["L01_V001_S0001"], '
                '"start_sec": 1.0, "end_sec": 2.0, "event_caption": "vẫy tay", "keywords": [], '
                '"action_tags": ["waving"], "previous_event_id": "L01_V001_E0000", "next_event_id": null}\n',
                encoding="utf-8",
            )
            repository = run(JsonlEventRepository.load(path))
            self.assertEqual(len(run(repository.all())), 2)
            first = run(repository.get("L01_V001_E0000"))
            self.assertEqual(first.next_event_id, "L01_V001_E0001")

    def test_retriever_fans_out_matching_event_to_its_member_scenes(self) -> None:
        events = [project_event({
            "event_id": "L01_V001_E0000", "video_id": "L01_V001",
            "scene_ids": ["L01_V001_S0000", "L01_V001_S0001"],
            "start_sec": 0.0, "end_sec": 2.0, "event_caption": "cào muối trên cánh đồng",
            "keywords": [], "action_tags": ["raking"], "previous_event_id": None, "next_event_id": None,
        })]
        retriever = EventSearchRetriever(events)
        plan = _plan(modality_weights={Modality.EVENT: 1.0}, query="cào muối")
        results = run(retriever.search(plan, limit=10))
        self.assertEqual({item.scene_id for item in results}, {"L01_V001_S0000", "L01_V001_S0001"})
        self.assertTrue(all(item.payload["event_id"] == "L01_V001_E0000" for item in results))


class FuseCandidatesTests(unittest.TestCase):
    def _candidate(self, scene_id: str, source: str, modality: Modality, score: float, rank: int) -> Candidate:
        return Candidate(
            entity_id=scene_id, scene_id=scene_id, video_id="v1",
            source=source, modality=modality, score=score, rank=rank,
        )

    def test_rrf_unchanged_from_weighted_rrf_behavior(self) -> None:
        visual = [self._candidate("s1", "dense", Modality.VISUAL, 0.8, 1)]
        ocr = [self._candidate("s2", "ocr", Modality.OCR, 10.0, 1)]
        fused = fuse_candidates([visual, ocr], {Modality.VISUAL: 1.0, Modality.OCR: 2.0}, method="rrf", rrf_k=60)
        self.assertEqual(fused[0].scene_id, "s2")

    def test_branch_override_changes_rrf_ranking(self) -> None:
        visual = [self._candidate("s1", "dense_visual", Modality.VISUAL, 0.8, 1)]
        ocr = [self._candidate("s2", "bm25_ocr", Modality.OCR, 10.0, 1)]
        branches = {"bm25_ocr": BranchRuntimeOptions(enabled=False)}
        fused = fuse_candidates(
            [visual, ocr], {Modality.VISUAL: 1.0, Modality.OCR: 2.0}, method="rrf", rrf_k=60, branches=branches,
        )
        self.assertEqual(fused[0].scene_id, "s1")

    def test_max_score_takes_best_branch_not_sum(self) -> None:
        # Two weak branches both hit s1 at low rank; one strong branch hits s2 at rank 1.
        weak_a = [self._candidate("s1", "a", Modality.CAPTION, 1.0, 50)]
        weak_b = [self._candidate("s1", "b", Modality.KEYWORD, 1.0, 50)]
        strong = [self._candidate("s2", "c", Modality.OCR, 1.0, 1)]
        weights = {Modality.CAPTION: 1.0, Modality.KEYWORD: 1.0, Modality.OCR: 1.0}
        summed = fuse_candidates([weak_a, weak_b, strong], weights, method="weighted_sum", rrf_k=60)
        maxed = fuse_candidates([weak_a, weak_b, strong], weights, method="max_score", rrf_k=60)
        # weighted_sum: two weak contributions (1/110 each) added together beat one
        # strong contribution (1/61) -> s1 first.
        self.assertEqual(summed[0].scene_id, "s1")
        # max_score: s1's best single contribution (1/110) is still weaker than
        # s2's single contribution (1/61) -> s2 first.
        self.assertEqual(maxed[0].scene_id, "s2")

    def test_intersection_requires_minimum_matching_branches(self) -> None:
        only_a = [self._candidate("s1", "a", Modality.CAPTION, 1.0, 1)]
        both = [
            self._candidate("s2", "a", Modality.CAPTION, 1.0, 2),
        ]
        both_b = [self._candidate("s2", "b", Modality.KEYWORD, 1.0, 1)]
        weights = {Modality.CAPTION: 1.0, Modality.KEYWORD: 1.0}
        fused = fuse_candidates(
            [only_a, both, both_b], weights, method="intersection", rrf_k=60, minimum_matching_branches=2,
        )
        self.assertEqual([item.scene_id for item in fused], ["s2"])

    def test_union_keeps_every_candidate_seen_by_any_branch(self) -> None:
        only_a = [self._candidate("s1", "a", Modality.CAPTION, 1.0, 1)]
        only_b = [self._candidate("s2", "b", Modality.KEYWORD, 1.0, 1)]
        fused = fuse_candidates([only_a, only_b], {Modality.CAPTION: 1.0, Modality.KEYWORD: 1.0}, method="union")
        self.assertEqual({item.scene_id for item in fused}, {"s1", "s2"})


class SearchCapabilitiesTests(unittest.TestCase):
    def test_capabilities_lists_only_registered_retrievers(self) -> None:
        from online.api.routes import search_capabilities

        class FakeRetriever:
            def __init__(self, name: str, modality: Modality) -> None:
                self.name = name
                self.modality = modality

        class FakeSearchService:
            retrievers = [FakeRetriever("dense_visual", Modality.VISUAL), FakeRetriever("bm25_ocr", Modality.OCR)]
            rule_config = None

        class FakeContainer:
            search_service = FakeSearchService()
            event_repository = None

        result = run(search_capabilities(FakeContainer()))
        self.assertEqual({item["branch_id"] for item in result["branches"]}, {"dense_visual", "bm25_ocr"})
        self.assertFalse(result["events_available"])
        self.assertIn("rrf", result["fusion_methods"])
        self.assertFalse(result["rerank"]["vlm"])


class EventRoutesTests(unittest.TestCase):
    def _container(self, event_repository):
        class FakeContainer:
            pass

        container = FakeContainer()
        container.event_repository = event_repository
        return container

    def test_get_event_returns_404_without_event_data(self) -> None:
        from fastapi import HTTPException

        from online.api.routes import get_event

        with self.assertRaises(HTTPException) as ctx:
            run(get_event("L01_V001_E0000", self._container(None)))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_event_and_neighbors_round_trip(self) -> None:
        from online.api.routes import get_event, get_event_neighbors

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                '{"event_id": "L01_V001_E0000", "video_id": "L01_V001", "scene_ids": ["L01_V001_S0000"], '
                '"start_sec": 0.0, "end_sec": 1.0, "event_caption": null, "keywords": [], '
                '"action_tags": [], "previous_event_id": null, "next_event_id": "L01_V001_E0001"}\n'
                '{"event_id": "L01_V001_E0001", "video_id": "L01_V001", "scene_ids": ["L01_V001_S0001"], '
                '"start_sec": 1.0, "end_sec": 2.0, "event_caption": null, "keywords": [], '
                '"action_tags": [], "previous_event_id": "L01_V001_E0000", "next_event_id": null}\n',
                encoding="utf-8",
            )
            repository = run(JsonlEventRepository.load(path))
            container = self._container(repository)
            event = run(get_event("L01_V001_E0000", container))
            self.assertEqual(event["event_id"], "L01_V001_E0000")
            neighbors = run(get_event_neighbors("L01_V001_E0000", container))
            self.assertIsNone(neighbors["previous"])
            self.assertEqual(neighbors["next"]["event_id"], "L01_V001_E0001")


if __name__ == "__main__":
    unittest.main()
