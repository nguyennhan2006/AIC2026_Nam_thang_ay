"""PR-03: một branch chết không được kéo đổ cả request.

Trước PR-03 `SearchService._retrieve` gọi `asyncio.gather` trần, nên bật
`AIC_ENABLE_EVENT_SEARCH` mà Qdrant treo là toàn bộ search trả 500. Các test
dưới đây khóa lại hành vi đúng: cô lập lỗi theo branch, báo trạng thái ra
ngoài, và chỉ thất bại khi KHÔNG branch nào chạy nổi.
"""

from __future__ import annotations

import asyncio
import unittest

from online.domain.models import (
    Candidate,
    Modality,
    QueryEvent,
    QueryPlan,
    SearchFilters,
    TaskType,
)
from online.domain.search_config import BranchRuntimeOptions, SearchOptions
from online.errors import DependencyUnavailableError
from online.services.query_expansion import QueryExpansionRetriever
from online.services.registry import RetrieverRegistry
from online.services.retrieval_orchestrator import RetrievalOrchestrator


def run(coro):
    return asyncio.run(coro)


def plan_for(options: SearchOptions | None = None) -> QueryPlan:
    return QueryPlan(
        task=TaskType.TEXTUAL_KIS,
        original_query="q",
        normalized_query="q",
        events=[QueryEvent(event_idx=0, text="q")],
        modality_weights={Modality.CAPTION: 1.0, Modality.OCR: 1.0, Modality.VISUAL: 1.0},
        filters=SearchFilters(),
        search_options=options or SearchOptions(),
    )


class FakeRetriever:
    backend_kind = "lexical"
    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(
        self,
        branch_id: str,
        *,
        variant: str = "raw",
        modality: Modality = Modality.CAPTION,
        behavior: str = "ok",
        delay: float = 0.0,
        count: int = 2,
    ) -> None:
        self.branch_id = branch_id
        self.execution_id = f"{branch_id}.{variant}"
        self.name = branch_id
        self.modality = modality
        self.behavior = behavior
        self.delay = delay
        self.count = count

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.behavior == "raise":
            raise RuntimeError("index corrupted")
        if self.behavior == "unavailable":
            raise DependencyUnavailableError("Qdrant unavailable: connection refused")
        if self.behavior == "empty":
            return []
        return [
            Candidate(
                candidate_id=f"L01_V001_S{index:04d}",
                scene_id=f"L01_V001_S{index:04d}",
                video_id="L01_V001",
                source=self.execution_id,
                modality=self.modality,
                raw_score=1.0 / (index + 1),
                rank=index + 1,
            )
            for index in range(self.count)
        ]


class OrchestratorTests(unittest.TestCase):
    def test_failed_branch_does_not_kill_the_request(self) -> None:
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("bm25_caption"),
            FakeRetriever("event_search", behavior="raise", modality=Modality.OCR),
        ])
        lists, statuses = run(orchestrator.execute(plan_for(), limit=10))
        self.assertEqual(len(lists[0]), 2)
        self.assertEqual(lists[1], [])
        states = {item.execution_id: item.state for item in statuses}
        self.assertEqual(states["bm25_caption.raw"], "success")
        self.assertEqual(states["event_search.raw"], "failed")
        failure = next(item for item in statuses if item.state == "failed")
        self.assertIn("RuntimeError", failure.warning)

    def test_dependency_error_is_reported_as_unavailable(self) -> None:
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("bm25_caption"),
            FakeRetriever("dense_visual", behavior="unavailable", modality=Modality.VISUAL),
        ])
        _lists, statuses = run(orchestrator.execute(plan_for(), limit=10))
        status = next(item for item in statuses if item.branch_id == "dense_visual")
        self.assertEqual(status.state, "unavailable")
        self.assertTrue(status.is_degraded)

    def test_slow_branch_is_cut_at_its_own_deadline(self) -> None:
        options = SearchOptions(
            branches={"slow_branch": BranchRuntimeOptions(timeout_ms=100)}
        )
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("bm25_caption"),
            FakeRetriever("slow_branch", behavior="ok", delay=5.0, modality=Modality.OCR),
        ])
        _lists, statuses = run(orchestrator.execute(plan_for(options), limit=10))
        status = next(item for item in statuses if item.branch_id == "slow_branch")
        self.assertEqual(status.state, "timeout")
        self.assertIn("100ms", status.warning)
        # Deadline phải thực sự cắt chứ không chờ hết 5 giây.
        self.assertLess(status.latency_ms, 2000)

    def test_empty_and_disabled_are_distinguished(self) -> None:
        options = SearchOptions(
            branches={"off_branch": BranchRuntimeOptions(enabled=False)}
        )
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("off_branch", behavior="empty"),
            FakeRetriever("no_match", behavior="empty", modality=Modality.OCR),
        ])
        _lists, statuses = run(orchestrator.execute(plan_for(options), limit=10))
        states = {item.branch_id: item.state for item in statuses}
        self.assertEqual(states["off_branch"], "disabled")
        self.assertEqual(states["no_match"], "empty")

    def test_all_branches_down_raises_instead_of_returning_nothing(self) -> None:
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("a", behavior="raise"),
            FakeRetriever("b", behavior="unavailable", modality=Modality.OCR),
        ])
        with self.assertRaises(DependencyUnavailableError) as ctx:
            run(orchestrator.execute(plan_for(), limit=10))
        self.assertIn("mọi retrieval branch", str(ctx.exception))

    def test_empty_result_is_not_treated_as_a_failure(self) -> None:
        orchestrator = RetrievalOrchestrator([
            FakeRetriever("a", behavior="empty"),
            FakeRetriever("b", behavior="empty", modality=Modality.OCR),
        ])
        _lists, statuses = run(orchestrator.execute(plan_for(), limit=10))
        self.assertTrue(all(not item.is_degraded for item in statuses))


class BranchIdentityTests(unittest.TestCase):
    def test_expansion_wrapper_keeps_branch_but_changes_execution(self) -> None:
        inner = FakeRetriever("bm25_caption")
        wrapper = QueryExpansionRetriever(inner)
        self.assertEqual(wrapper.branch_id, "bm25_caption")
        self.assertEqual(wrapper.execution_id, "bm25_caption.expanded")

    def test_expanded_candidates_carry_the_wrapper_execution_id(self) -> None:
        wrapper = QueryExpansionRetriever(FakeRetriever("bm25_caption"))
        candidates = run(wrapper.search(plan_for(), limit=5))
        self.assertTrue(candidates)
        # Trước PR-03 candidate vẫn mang source của inner, nên cấu hình cho
        # id mà /capabilities công bố hoàn toàn vô tác dụng.
        self.assertTrue(all(item.source == "bm25_caption.expanded" for item in candidates))

    def test_registry_groups_executions_under_their_branch(self) -> None:
        registry = RetrieverRegistry([
            FakeRetriever("bm25_caption"),
            QueryExpansionRetriever(FakeRetriever("bm25_caption")),
            FakeRetriever("bm25_ocr", modality=Modality.OCR),
        ])
        capabilities = {item.branch_id: item for item in registry.capabilities()}
        self.assertEqual(sorted(capabilities), ["bm25_caption", "bm25_ocr"])
        self.assertEqual(
            capabilities["bm25_caption"].execution_ids,
            ["bm25_caption.expanded", "bm25_caption.raw"],
        )

    def test_branch_level_config_applies_to_every_execution(self) -> None:
        from online.services.branch_options import effective_weight

        options = SearchOptions(branches={"bm25_caption": BranchRuntimeOptions(weight=4.0)})
        plan = plan_for(options)
        for execution_id in ("bm25_caption.raw", "bm25_caption.expanded"):
            with self.subTest(execution=execution_id):
                self.assertEqual(
                    effective_weight(plan, execution_id, Modality.CAPTION, "bm25_caption"), 4.0
                )

    def test_execution_level_config_beats_branch_level(self) -> None:
        from online.services.branch_options import effective_weight

        options = SearchOptions(branches={
            "bm25_caption": BranchRuntimeOptions(weight=4.0),
            "bm25_caption.expanded": BranchRuntimeOptions(enabled=False),
        })
        plan = plan_for(options)
        self.assertEqual(
            effective_weight(plan, "bm25_caption.raw", Modality.CAPTION, "bm25_caption"), 4.0
        )
        self.assertEqual(
            effective_weight(plan, "bm25_caption.expanded", Modality.CAPTION, "bm25_caption"), 0.0
        )


if __name__ == "__main__":
    unittest.main()
