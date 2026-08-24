"""Event repository + event search branch — Search Mixing Console W2/W3.

Baseline: BM25 (reusing the same `BM25Index` used for scene fields, see
online/adapters/bm25.py) over each event's aggregated caption/keywords/
action_tags — no event-level embedding (Event carries no pooled vector in
W1, see docs/14_TECHNICAL_PREPARATION.md). A matching event fans out to a
Candidate for each of its member scenes (SearchService only knows how to
hydrate scene_id-anchored candidates), all sharing the event's BM25 score.
"""

from __future__ import annotations

import asyncio

from pathlib import Path
from typing import Sequence

from online.adapters.bm25 import BM25Index
from online.domain.models import Candidate, EventDocument, Modality, QueryPlan
from online.errors import MetadataNotFoundError
from online.services.branch_options import effective_limit, effective_weight


def project_event(raw: dict) -> EventDocument:
    return EventDocument(
        event_id=raw["event_id"],
        video_id=raw["video_id"],
        scene_ids=list(raw.get("scene_ids", [])),
        start_sec=raw["start_sec"],
        end_sec=raw["end_sec"],
        event_caption=raw.get("event_caption"),
        keywords=list(raw.get("keywords", [])),
        action_tags=list(raw.get("action_tags", [])),
        previous_event_id=raw.get("previous_event_id"),
        next_event_id=raw.get("next_event_id"),
    )


class JsonlEventRepository:
    """In-memory read repository loaded from an events.jsonl export."""

    def __init__(self, events: dict[str, EventDocument]) -> None:
        self._events = events

    @classmethod
    async def load(cls, path: Path) -> "JsonlEventRepository":
        import asyncio
        import json

        def read() -> dict[str, EventDocument]:
            if not path.exists():
                raise MetadataNotFoundError(f"events JSONL not found: {path}")
            events: dict[str, EventDocument] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = project_event(json.loads(line))
                    events[event.event_id] = event
            return events

        return cls(await asyncio.to_thread(read))

    async def get(self, event_id: str) -> EventDocument | None:
        return self._events.get(event_id)

    async def get_many(self, event_ids: Sequence[str]) -> list[EventDocument]:
        return [self._events[item] for item in event_ids if item in self._events]

    async def all(self) -> list[EventDocument]:
        return list(self._events.values())


class EventSearchRetriever:
    """BM25 over event text, fanned out to each event's member scenes."""

    branch_id = "event_search"
    execution_id = "event_search.raw"
    name = branch_id
    modality = Modality.EVENT
    backend_kind = "lexical"
    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(self, events: list[EventDocument]) -> None:
        self.index = BM25Index(events, "text")

    @classmethod
    async def build(cls, repository: JsonlEventRepository) -> "EventSearchRetriever":
        return cls(await repository.all())

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []
        limit = effective_limit(plan, self.execution_id, limit, self.branch_id)
        query = plan.events[0].text if len(plan.events) == 1 else plan.normalized_query
        # BM25 Python thuần trên toàn bộ event — cùng lý do như bm25.py.
        results = await asyncio.to_thread(self.index.search, query, limit)
        candidates: list[Candidate] = []
        for event, score in results:
            if plan.filters.video_ids and event.video_id not in plan.filters.video_ids:
                continue
            for scene_id in event.scene_ids:
                if plan.filters.scene_ids and scene_id not in plan.filters.scene_ids:
                    continue
                candidates.append(Candidate(
                    candidate_id=scene_id, entity_type="scene", scene_id=scene_id,
                    event_id=event.event_id, video_id=event.video_id,
                    source=self.execution_id, modality=self.modality,
                    raw_score=score, score_kind="bm25",
                    rank=len(candidates) + 1,
                    index_id="bm25_event_inmemory",
                    payload={"matched_text": event.field_text()[:1000], "event_id": event.event_id},
                ))
                if len(candidates) >= limit:
                    return candidates
        return candidates


__all__ = ["JsonlEventRepository", "EventSearchRetriever", "project_event"]
