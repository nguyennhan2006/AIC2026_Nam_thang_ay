"""online/api/container.py phải tôn trọng 4 cờ AIC_ENABLE_* (Phần 2 docs/14).

Mặc định tắt: container KHÔNG được đổi hành vi so với trước khi thêm cờ. Bật từng
cờ phải wire đúng module tương ứng (OcrFuzzyRetriever/PreparedQueryPlanner/
QueryExpansionRetriever/RuleConfig) — backend local + HashingTextEncoder đủ để
kiểm tra wiring, không cần semantic embedding thật.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.color_search import ColorSearchRetriever
from online.adapters.event_search import EventSearchRetriever
from online.adapters.ocr_fuzzy import OcrFuzzyRetriever
from online.api.container import build_container
from online.config import Settings
from online.domain.models import SearchRequest, TaskType
from online.services.query_expansion import QueryExpansionRetriever
from online.services.query_prep import PreparedQueryPlanner
from scripts.seed_demo import main as seed


def run(coro):
    return asyncio.run(coro)


def _settings(path: Path, **overrides) -> Settings:
    base = dict(
        app_name="test",
        environment="test",
        log_level="INFO",
        backend="local",
        metadata_jsonl=path,
        qdrant_url=None,
        qdrant_api_key=None,
        qdrant_scene_collection="aic_scenes_v1",
        qdrant_vector_name="visual",
        embedding_url=None,
        embedding_api_key=None,
        request_timeout_sec=10.0,
        candidate_limit=100,
        rrf_k=60,
        data_root=Path("storage"),
        cors_origins=(),
        api_key=None,
        enable_ocr_fuzzy=False,
        enable_query_prep=False,
        enable_expansion=False,
        enable_rules=False,
        enable_object_search=False,
        enable_action_search=False,
        enable_color_search=False,
        enable_event_search=False,
    )
    base.update(overrides)
    return Settings(**base)


class ContainerFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed(False)
        cls.path = Path(__file__).resolve().parents[1] / "storage/exports/scenes.jsonl"

    def build(self, **overrides):
        return run(build_container(_settings(self.path, **overrides)))

    def test_default_flags_off_keep_baseline_wiring(self) -> None:
        service = self.build().search_service
        self.assertEqual(len(service.retrievers), 5)  # dense + 4x bm25
        self.assertFalse(any(isinstance(r, OcrFuzzyRetriever) for r in service.retrievers))
        self.assertFalse(any(isinstance(r, QueryExpansionRetriever) for r in service.retrievers))
        self.assertIsNone(service.rule_config)
        self.assertNotIsInstance(service.planner, PreparedQueryPlanner)

    def test_enable_ocr_fuzzy_wires_retriever_and_matches_diacritic_free_query(self) -> None:
        service = self.build(enable_ocr_fuzzy=True).search_service
        self.assertEqual(len(service.retrievers), 6)
        self.assertTrue(any(isinstance(r, OcrFuzzyRetriever) for r in service.retrievers))
        # "Hẹn ngày gặp lại" (L01_V001_S0001) không share token nào với bm25/dense
        # khi bỏ hết dấu — chỉ OcrFuzzyRetriever bắt được, và bonus của nó (rank 1,
        # weight OCR mặc định 0.35) đủ lớn để thắng chênh lệch rank giữa 3 scene ở
        # nhánh dense (corpus demo chỉ có 3 scene nên khoảng cách rank rất nhỏ).
        result = run(service.search(SearchRequest(query="hen ngay gap lai", task=TaskType.TEXTUAL_KIS, top_k=1)))
        self.assertEqual(result.results[0].scene_id, "L01_V001_S0001")

    def test_enable_expansion_wraps_caption_and_keyword_only(self) -> None:
        service = self.build(enable_expansion=True).search_service
        wrapped = [r for r in service.retrievers if isinstance(r, QueryExpansionRetriever)]
        self.assertEqual({r.inner.field for r in wrapped}, {"caption", "keyword"})

    def test_enable_rules_adds_rule_adjustments_payload(self) -> None:
        service = self.build(enable_rules=True).search_service
        self.assertIsNotNone(service.rule_config)
        plan = run(service.planner.plan(SearchRequest(
            query='"Gừng cay muối mặn xin đừng quên nhau"', task=TaskType.TEXTUAL_KIS, top_k=5,
        )))
        candidates, _statuses = run(service._retrieve(plan, service.candidate_limit))
        target = next(c for c in candidates if c.scene_id == "L01_V001_S0002")
        self.assertIn("ocr_exact", target.payload.get("rule_adjustments", {}))

    def test_enable_object_search_wires_bm25_object_retriever(self) -> None:
        service = self.build(enable_object_search=True).search_service
        self.assertEqual(len(service.retrievers), 6)
        matches = [r for r in service.retrievers if isinstance(r, LexicalRetriever) and r.field == "object"]
        self.assertEqual(len(matches), 1)

    def test_enable_action_search_wires_bm25_action_retriever(self) -> None:
        service = self.build(enable_action_search=True).search_service
        self.assertEqual(len(service.retrievers), 6)
        matches = [r for r in service.retrievers if isinstance(r, LexicalRetriever) and r.field == "action"]
        self.assertEqual(len(matches), 1)

    def test_enable_color_search_wires_color_retriever(self) -> None:
        service = self.build(enable_color_search=True).search_service
        self.assertEqual(len(service.retrievers), 6)
        self.assertTrue(any(isinstance(r, ColorSearchRetriever) for r in service.retrievers))

    def test_enable_event_search_wires_event_retriever_since_events_jsonl_always_exported(self) -> None:
        # export_dataset (datasection/exporter.py) always writes events.jsonl now
        # (W1 event grouping), including for seed_demo's zero-event demo dataset.
        container = self.build(enable_event_search=True)
        self.assertIsNotNone(container.event_repository)
        self.assertEqual(len(container.search_service.retrievers), 6)
        self.assertTrue(any(isinstance(r, EventSearchRetriever) for r in container.search_service.retrievers))

    def test_event_repository_available_even_when_event_search_disabled(self) -> None:
        container = self.build()
        self.assertIsNotNone(container.event_repository)
        self.assertEqual(len(container.search_service.retrievers), 5)

    def test_enable_query_prep_rewrites_normalized_query_on_temporal_marker(self) -> None:
        service = self.build(enable_query_prep=True).search_service
        self.assertIsInstance(service.planner, PreparedQueryPlanner)
        plan = run(service.planner.plan(SearchRequest(
            query="cào muối, sau đó vẫy tay, cuối cùng đứng trước căn nhà",
            task=TaskType.TEXTUAL_KIS, top_k=5,
        )))
        self.assertEqual(plan.normalized_query, "đứng trước căn nhà")


if __name__ == "__main__":
    unittest.main()
