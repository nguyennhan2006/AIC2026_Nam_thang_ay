"""Hybrid retrieval orchestration for KIS, AVS, and ordered visual sequences."""

from __future__ import annotations

import re
from time import perf_counter
from uuid import uuid4

from online.domain.models import (
    Candidate,
    Evidence,
    FrameEvidence,
    Modality,
    QueryPlan,
    SceneDocument,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from online.domain.execution import BranchStatus
from online.ports.interfaces import Retriever, SceneRepository
from online.services.fusion import fuse_candidates
from online.services.negative_constraints import apply_negative_constraints, extract_negative_constraints
from online.services.query_planner import RuleBasedQueryPlanner
from online.services.registry import RetrieverRegistry
from online.services.retrieval_orchestrator import RetrievalOrchestrator
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
        self.registry = RetrieverRegistry(retrievers)
        self.orchestrator = RetrievalOrchestrator(retrievers)
        self.planner = planner or RuleBasedQueryPlanner()
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        # Phương án E (bonus/penalty sau RRF), optional — None giữ nguyên hành vi
        # cũ; xem online/services/rules.py và docs/15_RESEARCH_AGENDA.md mục 5.
        self.rule_config = rule_config

    async def _retrieve(
        self, plan: QueryPlan, limit: int
    ) -> tuple[list[Candidate], list[BranchStatus]]:
        # Orchestrator cô lập lỗi theo từng branch: một branch timeout không
        # còn kéo đổ cả request như `asyncio.gather` trần trước PR-03.
        lists, statuses = await self.orchestrator.execute(plan, limit)
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
            return candidates, statuses
        documents = {
            document.scene_id: document
            for document in await self.repository.get_many(
                [candidate.scene_id for candidate in candidates if candidate.scene_id]
            )
        }
        if constraints:
            candidates = apply_negative_constraints(candidates, documents, constraints)
        if self.rule_config is None:
            return candidates, statuses
        exact_phrases = [
            phrase for event in plan.events for phrase in event.exact_phrases
        ]
        return apply_bonus_penalty(
            candidates,
            documents,
            exact_phrases=exact_phrases,
            query=plan.normalized_query,
            config=self.rule_config,
        ), statuses

    @staticmethod
    def _select_frame(
        document: SceneDocument, candidate: Candidate, query: str
    ) -> FrameEvidence | None:
        """Chọn frame đại diện cho một scene candidate.

        Thứ tự ưu tiên:

        1. Frame do chính retrieval chỉ ra (candidate neo frame, vd dense frame
           index) — đáng tin hơn mọi suy đoán ở tầng này.
        2. Frame khớp nhiều token query nhất.
        3. Frame gần giữa scene nhất (ít dính biên/cut nhất).

        Đây là baseline; safe-frame scoring đầy đủ (quality, blur, boundary)
        thuộc PR-07.
        """

        if not document.keyframes:
            return None
        if candidate.frame_idx is not None:
            for frame in document.keyframes:
                if frame.frame_idx == candidate.frame_idx:
                    return frame
        query_tokens = set(re.findall(r"\w+", query.casefold()))
        if query_tokens:
            scored = [
                (
                    len(query_tokens & set(re.findall(r"\w+", frame.search_text.casefold()))),
                    frame,
                )
                for frame in document.keyframes
            ]
            best_overlap, best_frame = max(scored, key=lambda item: (item[0], -item[1].frame_idx))
            if best_overlap:
                return best_frame
        midpoint = (document.start_sec + document.end_sec) / 2
        return min(document.keyframes, key=lambda frame: abs(frame.timestamp_sec - midpoint))

    async def _hydrate(self, candidates: list[Candidate], query: str = "") -> list[SearchHit]:
        documents = {
            item.scene_id: item
            for item in await self.repository.get_many(
                [candidate.scene_id for candidate in candidates if candidate.scene_id]
            )
        }
        hits: list[SearchHit] = []
        for rank, candidate in enumerate(candidates, start=1):
            document = documents.get(candidate.scene_id or "")
            if not document:
                continue
            payload = candidate.payload
            evidence = [Evidence.model_validate(item) for item in payload.get("evidence", [])]
            frame = self._select_frame(document, candidate, query)
            warnings: list[str] = []
            if frame is None:
                # Scene canonical luôn có >= 1 keyframe; nếu tới đây thì export
                # bị hỏng. Vẫn trả một frame_idx hợp lệ (giữa scene) để không
                # chặn cả response, nhưng nói rõ là suy ra chứ không phải evidence.
                best_frame_idx = (document.start_frame + document.end_frame_exclusive - 1) // 2
                warnings.append(
                    f"scene {document.scene_id} has no keyframe evidence; "
                    f"best_frame_idx={best_frame_idx} was derived from the scene midpoint"
                )
            else:
                best_frame_idx = frame.frame_idx
            hits.append(
                SearchHit(
                    rank=rank,
                    candidate_id=candidate.candidate_id,
                    scene_id=document.scene_id,
                    video_id=document.video_id,
                    video_path=document.video_path,
                    event_id=candidate.event_id or document.event_id,
                    scene_idx=document.scene_idx,
                    start_frame=document.start_frame,
                    end_frame_exclusive=document.end_frame_exclusive,
                    start_sec=document.start_sec,
                    end_sec=document.end_sec,
                    best_frame_idx=best_frame_idx,
                    best_keyframe_id=frame.keyframe_id if frame else None,
                    best_keyframe_path=frame.image_path if frame else None,
                    best_timestamp_sec=frame.timestamp_sec if frame else None,
                    score=candidate.raw_score,
                    keyframes=document.keyframes,
                    matched_modalities=[
                        Modality(item) for item in payload.get("matched_modalities", [])
                    ],
                    matched_branches=list(payload.get("matched_branches", [])),
                    evidence=evidence[:10],
                    component_scores=payload.get("component_scores", {}),
                    branch_contributions=payload.get("branch_contributions", {}),
                    warnings=warnings,
                )
            )
        return hits

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = perf_counter()
        query_id = str(uuid4())
        # `task=None` nghĩa là caller không chỉ định (vd gọi service trực tiếp
        # từ script/test); mặc định về task chính của cuộc thi. Route convenience
        # đã điền task của path trước khi tới đây.
        task = request.task or TaskType.TEXTUAL_KIS
        plan = await self.planner.plan(request)
        if task == TaskType.TRAKE and len(plan.events) >= 2:
            event_hit_lists: list[list[SearchHit]] = []
            statuses: list[BranchStatus] = []
            for event in plan.events:
                event_plan = plan.model_copy(
                    update={"normalized_query": event.text, "events": [event]}
                )
                candidates, step_statuses = await self._retrieve(
                    event_plan, self.candidate_limit
                )
                statuses = _merge_statuses(statuses, step_statuses)
                event_hit_lists.append(await self._hydrate(candidates, event.text))
            sequences = link_event_hits(event_hit_lists, limit=request.top_k)
            warnings = _status_warnings(statuses)
            return SearchResponse(
                query_id=query_id,
                task=task,
                took_ms=(perf_counter() - started) * 1000,
                status="COMPLETED_WITH_WARNINGS" if warnings else "COMPLETED",
                sequences=sequences,
                branch_status=statuses,
                query_plan=plan if request.debug else None,
                warnings=warnings,
            )

        candidates, statuses = await self._retrieve(plan, self.candidate_limit)
        hits = await self._hydrate(candidates[: request.top_k], plan.normalized_query)
        if task == TaskType.AVS:
            per_video = plan.search_options.fusion.max_results_per_video or 3
            hits = _diversify_avs(hits, request.top_k, per_video=per_video)
        results = hits[: request.top_k]
        warnings = _status_warnings(statuses) + [
            warning for hit in results for warning in hit.warnings
        ]
        return SearchResponse(
            query_id=query_id,
            task=task,
            took_ms=(perf_counter() - started) * 1000,
            status="COMPLETED_WITH_WARNINGS" if warnings else "COMPLETED",
            results=results,
            branch_status=statuses,
            query_plan=plan if request.debug else None,
            warnings=warnings,
        )


def _status_warnings(statuses: list[BranchStatus]) -> list[str]:
    return [
        f"{status.execution_id}: {status.state} — {status.warning}"
        for status in statuses
        if status.is_degraded
    ]


def _merge_statuses(
    accumulated: list[BranchStatus], new: list[BranchStatus]
) -> list[BranchStatus]:
    """Gộp trạng thái qua nhiều bước TRAKE.

    Mỗi step chạy lại toàn bộ branch; giữ lại kết quả xấu nhất của mỗi
    execution để một lần timeout ở step 2 không bị step 3 thành công che mất.
    """

    merged = {status.execution_id: status for status in accumulated}
    for status in new:
        previous = merged.get(status.execution_id)
        if previous is None or (status.is_degraded and not previous.is_degraded):
            merged[status.execution_id] = status
        elif previous.state == status.state:
            merged[status.execution_id] = previous.model_copy(
                update={
                    "latency_ms": max(previous.latency_ms, status.latency_ms),
                    "candidate_count": previous.candidate_count + status.candidate_count,
                }
            )
    return [merged[key] for key in sorted(merged)]


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
