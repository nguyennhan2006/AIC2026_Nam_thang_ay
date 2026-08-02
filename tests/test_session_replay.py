"""PR-09: session store + replay + orchestrator.stream.

Trước PR-09 không có gì được lưu lại sau một lần search — `query_id` sinh ra
rồi vứt đi. Các test này khóa lại: trace lưu đủ để replay, replay tạo session
MỚI (không ghi đè session gốc), và branch streaming thật sự phát sự kiện
theo thời gian branch xong chứ không phải giả lập.
"""

from __future__ import annotations

import asyncio
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.session_store import InMemorySessionStore
from online.domain.models import Candidate, Modality, QueryEvent, QueryPlan, SearchFilters, SearchRequest
from online.domain.search_config import SearchOptions
from online.domain.tasks import TaskType
from online.services.retrieval_orchestrator import RetrievalOrchestrator
from online.services.search import SearchService
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSONL = ROOT / "examples" / "scenes.jsonl"


def run(coro):
    return asyncio.run(coro)


def plan_for() -> QueryPlan:
    return QueryPlan(
        task=TaskType.TEXTUAL_KIS, original_query="q", normalized_query="q",
        events=[QueryEvent(event_idx=0, text="q")],
        modality_weights={Modality.CAPTION: 1.0}, filters=SearchFilters(),
        search_options=SearchOptions(),
    )


class SlowRetriever:
    branch_id = "slow"
    execution_id = "slow.raw"
    name = branch_id
    modality = Modality.CAPTION
    backend_kind = "lexical"

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        await asyncio.sleep(self.delay)
        return [Candidate(
            candidate_id="x", scene_id="x", video_id="v", source=self.execution_id,
            modality=self.modality, raw_score=1.0, rank=1,
        )]


class OrchestratorStreamTests(unittest.TestCase):
    def test_stream_yields_each_branch_exactly_once(self) -> None:
        async def scenario():
            orchestrator = RetrievalOrchestrator([SlowRetriever(0.01), SlowRetriever(0.02)])
            seen = []
            async for _candidates, status in orchestrator.stream(plan_for(), limit=10):
                seen.append(status.execution_id)
            return seen

        seen = run(scenario())
        self.assertEqual(len(seen), 2)

    def test_faster_branch_is_yielded_before_the_slower_one(self) -> None:
        async def scenario():
            orchestrator = RetrievalOrchestrator([
                _named(SlowRetriever(0.05), "slow"), _named(SlowRetriever(0.001), "fast"),
            ])
            order = []
            async for _candidates, status in orchestrator.stream(plan_for(), limit=10):
                order.append(status.branch_id)
            return order

        order = run(scenario())
        self.assertEqual(order[0], "fast")

    def test_execute_still_returns_in_retriever_order_unlike_stream(self) -> None:
        async def scenario():
            orchestrator = RetrievalOrchestrator([
                _named(SlowRetriever(0.03), "slow"), _named(SlowRetriever(0.001), "fast"),
            ])
            _lists, statuses = await orchestrator.execute(plan_for(), limit=10)
            return [item.branch_id for item in statuses]

        order = run(scenario())
        self.assertEqual(order, ["slow", "fast"])


def _named(retriever: SlowRetriever, name: str) -> SlowRetriever:
    retriever.branch_id = name
    retriever.execution_id = f"{name}.raw"
    retriever.name = name
    return retriever


class SessionStoreTests(unittest.TestCase):
    def test_put_then_get_round_trips(self) -> None:
        from online.domain.session import SearchExecutionTrace

        async def scenario():
            store = InMemorySessionStore()
            trace = SearchExecutionTrace(
                session_id="s1", task=TaskType.TEXTUAL_KIS,
                raw_request=SearchRequest(query="q", task=TaskType.TEXTUAL_KIS),
            )
            await store.put(trace)
            return await store.get("s1")

        fetched = run(scenario())
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.session_id, "s1")

    def test_missing_session_returns_none(self) -> None:
        store = InMemorySessionStore()
        self.assertIsNone(run(store.get("missing")))

    def test_store_evicts_oldest_beyond_max_size(self) -> None:
        from online.domain.session import SearchExecutionTrace

        async def scenario():
            store = InMemorySessionStore(max_size=2)
            for i in range(3):
                await store.put(SearchExecutionTrace(
                    session_id=f"s{i}", task=TaskType.TEXTUAL_KIS,
                    raw_request=SearchRequest(query="q", task=TaskType.TEXTUAL_KIS),
                ))
            return await store.get("s0"), await store.get("s2")

        oldest, newest = run(scenario())
        self.assertIsNone(oldest)
        self.assertIsNotNone(newest)


class SearchServiceSessionTests(unittest.TestCase):
    def _service(self, session_store: InMemorySessionStore) -> SearchService:
        repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        retrievers = [
            run(LexicalRetriever.build(field, repository))
            for field in ("caption", "ocr", "asr", "keyword")
        ]
        return SearchService(repository, retrievers, candidate_limit=20, session_store=session_store)

    def test_search_records_a_trace_when_store_is_configured(self) -> None:
        store = InMemorySessionStore()
        service = self._service(store)
        response = run(service.search(SearchRequest(query="căn nhà", task=TaskType.TEXTUAL_KIS)))
        trace = run(store.get(response.query_id))
        self.assertIsNotNone(trace)
        self.assertEqual(trace.task, TaskType.TEXTUAL_KIS)
        self.assertEqual(trace.raw_request.query, "căn nhà")

    def test_no_store_means_no_trace_but_search_still_works(self) -> None:
        repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        retrievers = [run(LexicalRetriever.build("caption", repository))]
        plain = SearchService(repository, retrievers, candidate_limit=20)
        response = run(plain.search(SearchRequest(query="căn nhà", task=TaskType.TEXTUAL_KIS)))
        self.assertTrue(response.results or response.kis)
        self.assertIsNone(plain.session_store)

    def test_replay_creates_a_new_session_linked_to_the_original(self) -> None:
        store = InMemorySessionStore()
        service = self._service(store)
        original = run(service.search(SearchRequest(query="căn nhà", task=TaskType.TEXTUAL_KIS)))
        replayed = run(service.replay(original.query_id))
        self.assertIsNotNone(replayed)
        self.assertEqual(replayed.replayed_from, original.query_id)
        self.assertNotEqual(replayed.query_id, original.query_id)

    def test_replay_of_unknown_session_returns_none(self) -> None:
        service = self._service(InMemorySessionStore())
        self.assertIsNone(run(service.replay("does-not-exist")))

    def test_replay_increments_the_replay_count_on_the_original_trace(self) -> None:
        store = InMemorySessionStore()
        service = self._service(store)
        original = run(service.search(SearchRequest(query="căn nhà", task=TaskType.TEXTUAL_KIS)))
        run(service.replay(original.query_id))
        run(service.replay(original.query_id))
        trace = run(store.get(original.query_id))
        self.assertEqual(trace.replay_count, 2)


class SearchStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        # Dựng service ở tầng sync (test method), KHÔNG dựng bên trong
        # coroutine `scenario` — `_service` gọi `run()` (asyncio.run), gọi nó
        # từ trong một coroutine đang chạy dưới asyncio.run khác sẽ vỡ.
        repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        retrievers = [
            run(LexicalRetriever.build(field, repository))
            for field in ("caption", "ocr", "asr", "keyword")
        ]
        self.service = SearchService(repository, retrievers, candidate_limit=20)

    def _collect(self, query: str = "căn nhà") -> list[dict]:
        async def scenario():
            return [
                event async for event in self.service.search_stream(
                    SearchRequest(query=query, task=TaskType.TEXTUAL_KIS, top_k=3)
                )
            ]

        return run(scenario())

    def test_event_sequence_starts_and_ends_correctly(self) -> None:
        events = [item["type"] for item in self._collect()]
        self.assertEqual(events[0], "search_started")
        self.assertEqual(events[1], "query_prepared")
        self.assertEqual(events[-1], "search_completed")
        self.assertIn("fusion_completed", events)
        self.assertIn("evidence_ready", events)

    def test_every_branch_gets_a_started_and_a_completed_event(self) -> None:
        events = self._collect()
        started = {e["execution_id"] for e in events if e["type"] == "branch_started"}
        completed = {
            e["execution_id"] for e in events if e["type"] in ("branch_completed", "branch_failed")
        }
        self.assertEqual(started, completed)
        self.assertTrue(started)

    def test_search_completed_event_carries_the_full_response(self) -> None:
        last = self._collect()[-1]
        self.assertEqual(last["type"], "search_completed")
        self.assertIn("results", last["response"])
        self.assertIn("kis", last["response"])


if __name__ == "__main__":
    unittest.main()
