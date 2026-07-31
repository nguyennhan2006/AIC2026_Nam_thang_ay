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
from online.services.fusion import fuse_candidates
from online.services.negative_constraints import apply_negative_constraints, extract_negative_constraints
from online.services.query_planner import RuleBasedQueryPlanner
from online.services.rules import RuleConfig, apply_bonus_penalty
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
        rule_config: RuleConfig | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.repository = repository
        self.retrievers = retrievers
        self.planner = planner or RuleBasedQueryPlanner()
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        # Phương án E (bonus/penalty sau RRF), optional — None giữ nguyên hành vi
        # cũ; xem online/services/rules.py và docs/15_RESEARCH_AGENDA.md mục 5.
        self.rule_config = rule_config

    async def _retrieve(self, plan: QueryPlan, limit: int) -> list[Candidate]:
        lists = await asyncio.gather(
            *(item.search(plan, limit=limit) for item in self.retrievers)
        )
        fusion_options = plan.search_options.fusion
        # rrf_k also exists as a deployment-level default (SearchService.rrf_k /
        # AIC_RRF_K); only let a request override it when the caller actually set
        # search_options.fusion.rrf_k explicitly (model_fields_set), so a request
        # with no search_options keeps using the deployment default exactly as
        # before FusionOptions existed.
        rrf_k = fusion_options.rrf_k if "rrf_k" in fusion_options.model_fields_set else self.rrf_k
        candidates = fuse_candidates(
            lists,
            plan.modality_weights,
            method=fusion_options.method,
            rrf_k=rrf_k,
            limit=limit,
            branches=plan.search_options.branches,
            minimum_matching_branches=fusion_options.minimum_matching_branches,
        )
        constraints = (
            extract_negative_constraints(plan.normalized_query)
            if plan.search_options.query.enable_negative_constraints else []
        )
        if not constraints and self.rule_config is None:
            return candidates
        documents = {
            document.scene_id: document
            for document in await self.repository.get_many(
                [candidate.scene_id for candidate in candidates]
            )
        }
        if constraints:
            candidates = apply_negative_constraints(candidates, documents, constraints)
        if self.rule_config is None:
            return candidates
        exact_phrases = [
            phrase for event in plan.events for phrase in event.exact_phrases
        ]
        return apply_bonus_penalty(
            candidates,
            documents,
            exact_phrases=exact_phrases,
            query=plan.normalized_query,
            config=self.rule_config,
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
            per_video = plan.search_options.fusion.max_results_per_video or 3
            hits = _diversify_avs(hits, request.top_k, per_video=per_video)
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
