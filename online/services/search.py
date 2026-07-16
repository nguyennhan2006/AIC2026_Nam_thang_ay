"""Hybrid retrieval orchestration for KIS, AVS, and ordered visual sequences."""

from __future__ import annotations

import asyncio
import re
from time import perf_counter
from uuid import uuid4

from online.domain.models import (
    Candidate,
    Evidence,
    Modality,
    QueryPlan,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from online.ports.interfaces import Retriever, SceneRepository
from online.services.fusion import weighted_rrf
from online.services.query_planner import RuleBasedQueryPlanner
from online.services.temporal import link_event_hits


class SearchService:
    def __init__(
        self,
        repository: SceneRepository,
        retrievers: list[Retriever],
        *,
        planner: RuleBasedQueryPlanner | None = None,
        candidate_limit: int = 100,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.repository = repository
        self.retrievers = retrievers
        self.planner = planner or RuleBasedQueryPlanner()
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k

    async def _retrieve(self, plan: QueryPlan, limit: int) -> list[Candidate]:
        lists = await asyncio.gather(
            *(item.search(plan, limit=limit) for item in self.retrievers)
        )
        return weighted_rrf(
            lists,
            plan.modality_weights,
            rrf_k=self.rrf_k,
            limit=limit,
        )

    async def _hydrate(self, candidates: list[Candidate], query: str = "") -> list[SearchHit]:
        documents = {
            item.scene_id: item
            for item in await self.repository.get_many(
                [candidate.scene_id for candidate in candidates]
            )
        }
        hits: list[SearchHit] = []
        for candidate in candidates:
            document = documents.get(candidate.scene_id)
            if not document:
                continue
            payload = candidate.payload
            evidence = [Evidence.model_validate(item) for item in payload.get("evidence", [])]
            best_idx = 0
            query_tokens = set(re.findall(r"\w+", query.casefold()))
            if query_tokens and document.keyframe_evidence:
                scores = []
                for frame in document.keyframe_evidence:
                    tokens = set(re.findall(r"\w+", str(frame.get("text", "")).casefold()))
                    scores.append(len(query_tokens & tokens) / max(len(query_tokens), 1))
                best_idx = max(range(len(scores)), key=scores.__getitem__)
            elif document.keyframe_timestamps:
                midpoint = (document.start_sec + document.end_sec) / 2
                best_idx = min(range(len(document.keyframe_timestamps)), key=lambda i: abs(document.keyframe_timestamps[i] - midpoint))
            hits.append(
                SearchHit(
                    scene_id=document.scene_id,
                    video_id=document.video_id,
                    video_path=document.video_path,
                    scene_idx=document.scene_idx,
                    start_sec=document.start_sec,
                    end_sec=document.end_sec,
                    score=candidate.score,
                    keyframe_ids=document.keyframe_ids,
                    keyframe_paths=document.keyframe_paths,
                    keyframe_timestamps=document.keyframe_timestamps,
                    best_keyframe_id=document.keyframe_ids[best_idx] if document.keyframe_ids else None,
                    best_keyframe_path=document.keyframe_paths[best_idx] if document.keyframe_paths else None,
                    best_timestamp_sec=document.keyframe_timestamps[best_idx] if document.keyframe_timestamps else None,
                    matched_modalities=[
                        Modality(item) for item in payload.get("matched_modalities", [])
                    ],
                    evidence=evidence[:10],
                    component_scores=payload.get("component_scores", {}),
                )
            )
        return hits

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = perf_counter()
        query_id = str(uuid4())
        plan = await self.planner.plan(request)
        if request.task == TaskType.SEQUENCE and len(plan.events) >= 2:
            event_hit_lists: list[list[SearchHit]] = []
            for event in plan.events:
                event_plan = plan.model_copy(
                    update={"normalized_query": event.text, "events": [event]}
                )
                candidates = await self._retrieve(event_plan, self.candidate_limit)
                event_hit_lists.append(await self._hydrate(candidates, event.text))
            sequences = link_event_hits(event_hit_lists, limit=request.top_k)
            return SearchResponse(
                query_id=query_id,
                task=request.task,
                took_ms=(perf_counter() - started) * 1000,
                sequences=sequences,
                query_plan=plan if request.debug else None,
            )

        candidates = await self._retrieve(plan, self.candidate_limit)
        hits = await self._hydrate(candidates[: request.top_k], plan.normalized_query)
        if request.task == TaskType.AVS:
            hits = _diversify_avs(hits, request.top_k)
        return SearchResponse(
            query_id=query_id,
            task=request.task,
            took_ms=(perf_counter() - started) * 1000,
            results=hits[: request.top_k],
            query_plan=plan if request.debug else None,
        )


def _diversify_avs(hits: list[SearchHit], limit: int, per_video: int = 3) -> list[SearchHit]:
    counts: dict[str, int] = {}
    output: list[SearchHit] = []
    for hit in hits:
        if counts.get(hit.video_id, 0) >= per_video:
            continue
        output.append(hit)
        counts[hit.video_id] = counts.get(hit.video_id, 0) + 1
        if len(output) >= limit:
            break
    return output
