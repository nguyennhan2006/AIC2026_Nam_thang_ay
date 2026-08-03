"""Hybrid retrieval orchestration for KIS, AVS, and ordered visual sequences."""

from __future__ import annotations

import re
from time import perf_counter
from typing import AsyncIterator
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
from online.domain.search_config import ResultOptions
from online.domain.session import SearchExecutionTrace
from online.ports.interfaces import Retriever, SceneRepository, SessionStore
from online.services.avs import AvsProcessor
from online.services.deduplication import deduplicate_for_task
from online.services.evidence_builder import EvidenceBuilder
from online.services.fusion import fuse_candidates
from online.services.kis import KisProcessor
from online.services.negative_constraints import apply_negative_constraints, extract_negative_constraints
from online.services.qa import QaProcessor
from online.services.query_planner import RuleBasedQueryPlanner
from online.services.registry import RetrieverRegistry
from online.services.rerank_pipeline import RerankPipeline
from online.services.score_normalization import normalize_all
from online.services.retrieval_orchestrator import RetrievalOrchestrator, _branch_identity
from online.services.rules import RuleConfig, apply_bonus_penalty
from online.services.thresholding import apply_thresholds
from online.services.temporal import link_event_hits
from online.services.trake import TrakeProcessor


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
        rerank_pipeline: RerankPipeline | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        session_store: SessionStore | None = None,
        dataset_version: str | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.repository = repository
        self.retrievers = retrievers
        self.registry = RetrieverRegistry(retrievers)
        self.orchestrator = RetrievalOrchestrator(retrievers)
        # None = không lưu trace (vd script/test không cần replay). Có store
        # thì MỌI lần search (kể cả qua endpoint convenience) đều ghi lại
        # được, vì tất cả đều đi qua đúng một hàm `search()` này.
        self.session_store = session_store
        self.dataset_version = dataset_version
        self.evidence_builder = evidence_builder or EvidenceBuilder(repository)
        # None = không có tầng rerank nào; cascade vẫn chạy được và chỉ ghi
        # warning "chưa cấu hình", nên hành vi mặc định không đổi.
        self.rerank_pipeline = rerank_pipeline or RerankPipeline(self.evidence_builder)
        # Bốn processor chuyên biệt (PR-07). Chúng chạy SAU lõi retrieval dùng
        # chung, đúng kiến trúc "một lõi + bốn task processor".
        self.kis_processor = KisProcessor()
        self.qa_processor = QaProcessor()
        self.trake_processor = TrakeProcessor()
        self.avs_processor = AvsProcessor()
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
        # Chuẩn hóa trước, cắt ngưỡng sau, rồi mới fuse. Thứ tự này bắt buộc:
        # RRF dùng *hạng* trong danh sách của branch, nên cắt sau khi fuse sẽ
        # không cho các candidate còn lại lên hạng (xem services/thresholding.py).
        lists = normalize_all(lists, method=fusion_options.normalized_score_method)
        lists, thresholded = apply_thresholds(lists, plan)
        statuses = _annotate_thresholds(statuses, thresholded)
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

    async def _attach_events(self, candidates: list[Candidate]) -> list[Candidate]:
        """Gắn `event_id` từ metadata scene để dedup theo event chạy được.

        Candidate từ nhánh event đã có sẵn `event_id`; nhánh khác thì không, và
        nếu không gắn ở đây thì dedup scope=event sẽ im lặng thoái hóa thành
        dedup theo scene.
        """

        missing = [item.scene_id for item in candidates if item.scene_id and not item.event_id]
        if not missing:
            return candidates
        documents = {
            document.scene_id: document
            for document in await self.repository.get_many(missing)
        }
        output: list[Candidate] = []
        for candidate in candidates:
            document = documents.get(candidate.scene_id or "")
            if candidate.event_id or document is None or not document.event_id:
                output.append(candidate)
                continue
            output.append(candidate.model_copy(update={"event_id": document.event_id}))
        return output

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
        """Chạy một lần search đầy đủ và (nếu có `session_store`) lưu trace."""

        response = await self._search_impl(request)
        await self._record_trace(response, request)
        return response

    async def _record_trace(self, response: SearchResponse, request: SearchRequest) -> None:
        if self.session_store is None:
            return
        await self.session_store.put(SearchExecutionTrace(
            session_id=response.query_id,
            task=response.task,
            raw_request=request,
            branch_status=response.branch_status,
            status=response.status,
            warnings=response.warnings,
            took_ms=response.took_ms,
            dataset_version=self.dataset_version,
            model_versions=self.evidence_builder.model_versions,
        ))

    async def replay(self, session_id: str) -> SearchResponse | None:
        """Chạy lại đúng request đã lưu của `session_id`; None nếu không có trace.

        Kết quả là một session MỚI (query_id mới), đánh dấu `replayed_from`
        trỏ về session gốc — để so sánh gốc-vs-replay thay vì ghi đè lẫn nhau.
        Không tái sử dụng response cũ: dataset/index có thể đã đổi từ lúc đó,
        nên "replay" nghĩa là "chạy lại y hệt input", không phải "trả cache".
        """

        if self.session_store is None:
            return None
        trace = await self.session_store.get(session_id)
        if trace is None:
            return None
        if hasattr(self.session_store, "update"):
            await self.session_store.update(session_id, replay_count=trace.replay_count + 1)
        response = await self.search(trace.raw_request)
        return response.model_copy(update={"replayed_from": session_id})

    async def _search_impl(self, request: SearchRequest) -> SearchResponse:
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
            documents = await self._documents_for(
                [hit for hits in event_hit_lists for hit in hits]
            )
            # Stage A khóa video trước, rồi mới beam search trong video đó và
            # tinh chỉnh frame — thay cho link_event_hits chỉ nối scene.
            trake = self.trake_processor.run(
                [event.text for event in plan.events],
                event_hit_lists,
                documents,
                limit=request.top_k,
            )
            sequences = link_event_hits(event_hit_lists, limit=request.top_k)
            warnings = _status_warnings(statuses)
            if not trake:
                warnings.append(
                    "TRAKE: không dựng được chuỗi nào — không có video nào phủ "
                    "đủ số step tối thiểu"
                )
            return SearchResponse(
                query_id=query_id,
                task=task,
                took_ms=(perf_counter() - started) * 1000,
                status="COMPLETED_WITH_WARNINGS" if warnings else "COMPLETED",
                sequences=sequences,
                trake=trake,
                branch_status=statuses,
                query_plan=plan if request.debug else None,
                warnings=warnings,
            )

        candidates, statuses = await self._retrieve(plan, self.candidate_limit)
        candidates = await self._attach_events(candidates)
        fusion_options = plan.search_options.fusion
        candidates = deduplicate_for_task(
            candidates,
            task,
            scope_override=(
                fusion_options.dedup_scope
                if "dedup_scope" in fusion_options.model_fields_set
                else None
            ),
            max_per_video_override=fusion_options.max_results_per_video,
        )
        # Rerank chạy SAU dedup: đưa 10 scene liền kề của cùng một sự kiện vào
        # cross-encoder là đốt ngân sách model cho cùng một nội dung.
        rerank = await self.rerank_pipeline.run(
            plan.normalized_query, candidates, plan.search_options.rerank
        )
        candidates = rerank.candidates
        hits = await self._hydrate(candidates[: request.top_k], plan.normalized_query)
        results = _format_results(hits, request.top_k, plan.search_options.results)
        warnings = (
            _status_warnings(statuses)
            + rerank.warnings
            + [warning for hit in results for warning in hit.warnings]
        )
        task_results = await self._run_task_processor(
            task, plan, request, results, candidates, rerank.packs
        )
        return SearchResponse(
            query_id=query_id,
            task=task,
            took_ms=(perf_counter() - started) * 1000,
            status="COMPLETED_WITH_WARNINGS" if warnings else "COMPLETED",
            results=results,
            branch_status=statuses,
            query_plan=plan if request.debug else None,
            warnings=warnings,
            **task_results,
        )

    async def search_stream(self, request: SearchRequest) -> AsyncIterator[dict]:
        """Như `search`, nhưng phát sự kiện thật theo từng giai đoạn (PR-09).

        Chỉ branch retrieval là thứ có thể stream đúng nghĩa (chạy song song,
        xong lúc nào phát lúc đó — dùng `RetrievalOrchestrator.stream`).
        Fusion/dedup/rerank/hydrate/task-processor là các bước tính một lần,
        nên chỉ phát MỘT sự kiện "xong giai đoạn" cho mỗi bước — không giả
        lập tiến độ phần trăm cho những gì không thực sự chia nhỏ được.

        TRAKE nhiều bước (>= 2 event) không stream theo branch: mỗi bước lại
        chạy một vòng retrieval riêng, nên việc dựng progress-per-step cho
        đúng nghĩa cần một luồng sự kiện khác hẳn — trường hợp này chỉ phát
        `search_started` -> `alignment_completed` -> `search_completed`, và
        nói rõ điều đó trong sự kiện để không ai tưởng nhầm là bug.
        """

        started = perf_counter()
        query_id = str(uuid4())
        task = request.task or TaskType.TEXTUAL_KIS
        yield {"type": "search_started", "query_id": query_id, "task": task.value}

        plan = await self.planner.plan(request)
        yield {
            "type": "query_prepared",
            "query_id": query_id,
            "normalized_query": plan.normalized_query,
            "modality_weights": {key.value: value for key, value in plan.modality_weights.items()},
        }

        if task == TaskType.TRAKE and len(plan.events) >= 2:
            response = await self._search_impl(request)
            await self._record_trace(response, request)
            yield {
                "type": "alignment_completed",
                "query_id": response.query_id,
                "sequence_count": len(response.trake),
                "note": "TRAKE nhiều bước không phát sự kiện theo branch — xem docstring search_stream",
            }
            yield {
                "type": "search_completed",
                "query_id": response.query_id,
                "response": response.model_dump(mode="json"),
            }
            return

        for retriever in self.retrievers:
            branch_id, execution_id = _branch_identity(retriever)
            yield {
                "type": "branch_started", "query_id": query_id,
                "branch_id": branch_id, "execution_id": execution_id,
            }

        lists: list[list[Candidate]] = []
        statuses: list[BranchStatus] = []
        async for candidates, status in self.orchestrator.stream(plan, self.candidate_limit):
            lists.append(candidates)
            statuses.append(status)
            yield {
                "type": "branch_failed" if status.is_degraded else "branch_completed",
                "query_id": query_id,
                "branch_id": status.branch_id,
                "execution_id": status.execution_id,
                "state": status.state,
                "latency_ms": status.latency_ms,
                "candidate_count": status.candidate_count,
                "warning": status.warning,
            }

        fusion_options = plan.search_options.fusion
        lists = normalize_all(lists, method=fusion_options.normalized_score_method)
        lists, thresholded = apply_thresholds(lists, plan)
        statuses = _annotate_thresholds(statuses, thresholded)
        rrf_k = fusion_options.rrf_k if "rrf_k" in fusion_options.model_fields_set else self.rrf_k
        candidates = fuse_candidates(
            lists, plan.modality_weights, method=fusion_options.method, rrf_k=rrf_k,
            limit=self.candidate_limit, branches=plan.search_options.branches,
            minimum_matching_branches=fusion_options.minimum_matching_branches,
        )
        candidates = await self._attach_events(candidates)
        candidates = deduplicate_for_task(
            candidates, task,
            scope_override=(
                fusion_options.dedup_scope
                if "dedup_scope" in fusion_options.model_fields_set else None
            ),
            max_per_video_override=fusion_options.max_results_per_video,
        )
        yield {"type": "fusion_completed", "query_id": query_id, "candidate_count": len(candidates)}

        rerank = await self.rerank_pipeline.run(
            plan.normalized_query, candidates, plan.search_options.rerank
        )
        if rerank.stages:
            yield {
                "type": "rerank_completed", "query_id": query_id,
                "stages": [
                    {"stage": item.stage, "applied": item.applied, "warning": item.warning}
                    for item in rerank.stages
                ],
            }
        candidates = rerank.candidates
        hits = await self._hydrate(candidates[: request.top_k], plan.normalized_query)
        results = _format_results(hits, request.top_k, plan.search_options.results)
        yield {"type": "evidence_ready", "query_id": query_id, "count": len(results)}

        warnings = (
            _status_warnings(statuses)
            + rerank.warnings
            + [warning for hit in results for warning in hit.warnings]
        )
        task_results = await self._run_task_processor(
            task, plan, request, results, candidates, rerank.packs
        )
        response = SearchResponse(
            query_id=query_id, task=task, took_ms=(perf_counter() - started) * 1000,
            status="COMPLETED_WITH_WARNINGS" if warnings else "COMPLETED",
            results=results, branch_status=statuses,
            query_plan=plan if request.debug else None, warnings=warnings, **task_results,
        )
        await self._record_trace(response, request)
        yield {"type": "search_completed", "query_id": query_id, "response": response.model_dump(mode="json")}

    async def _documents_for(self, hits: list[SearchHit]) -> dict[str, SceneDocument]:
        return {
            document.scene_id: document
            for document in await self.repository.get_many(
                sorted({hit.scene_id for hit in hits})
            )
        }

    async def _run_task_processor(
        self,
        task: TaskType,
        plan: QueryPlan,
        request: SearchRequest,
        results: list[SearchHit],
        candidates: list[Candidate],
        packs: dict,
    ) -> dict:
        """Chạy processor chuyên biệt của task trên kết quả đã rerank."""

        if not results:
            return {}
        if task == TaskType.TEXTUAL_KIS:
            documents = await self._documents_for(results)
            return {
                "kis": self.kis_processor.rank(
                    plan.original_query, results, documents,
                    packs=packs, limit=request.top_k,
                )
            }

        # QA và AVS đều cần evidence pack đầy đủ; dựng lazy cho đúng phần đầu.
        by_id = {item.candidate_id: item for item in candidates}
        head = [by_id[hit.candidate_id] for hit in results if hit.candidate_id in by_id]
        evidence_packs = []
        for candidate, hit in zip(head, results, strict=False):
            pack = packs.get(candidate.candidate_id)
            if pack is None:
                pack = await self.evidence_builder.build(
                    candidate, best_frame_idx=hit.best_frame_idx
                )
            elif pack.best_frame_idx is None:
                pack = pack.model_copy(update={"best_frame_idx": hit.best_frame_idx})
            if pack is not None:
                evidence_packs.append(pack)
        scores = {hit.candidate_id: hit.score for hit in results}

        if task == TaskType.QA:
            return {
                "qa": self.qa_processor.answer(
                    plan.original_query, evidence_packs,
                    frame_scores=scores, limit=request.top_k,
                )
            }
        if task == TaskType.AVS:
            return {
                "avs": self.avs_processor.rank(
                    plan.original_query, evidence_packs,
                    retrieval_scores=scores, limit=request.top_k,
                )
            }
        return {}


def _annotate_thresholds(
    statuses: list[BranchStatus], thresholded: dict[str, int]
) -> list[BranchStatus]:
    """Ghi rõ branch nào bị ngưỡng cắt bao nhiêu candidate.

    Không có dòng này thì `candidate_count` tụt xuống mà không ai biết là do
    ngưỡng của chính mình hay do index rỗng.
    """

    if not thresholded:
        return statuses
    output: list[BranchStatus] = []
    for status in statuses:
        count = thresholded.get(status.execution_id)
        if not count:
            output.append(status)
            continue
        output.append(
            status.model_copy(
                update={
                    "candidate_count": max(status.candidate_count - count, 0),
                    "warning": f"{count} candidate bị ngưỡng min_score loại hoặc hạ hạng",
                }
            )
        )
    return output


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


_SORT_KEYS: dict[str, tuple[str, ...]] = {
    # "dense_visual" khi có embedding thật (qdrant, hoặc local + PR-13 đã
    # chạy), "lexical_hash_fallback" khi backend local chưa có embedding —
    # container.py chỉ đăng ký MỘT trong hai cho mỗi lần chạy, nên thử theo
    # thứ tự và dùng key đầu tiên có mặt thay vì đoán cố định một cái.
    "visual_score": ("dense_visual.raw", "lexical_hash_fallback.raw"),
    "caption_score": ("bm25_caption.raw",),
    "ocr_score": ("bm25_ocr.raw",),
    "asr_score": ("bm25_asr.raw",),
}


def _component_score(hit: SearchHit, candidate_keys: tuple[str, ...]) -> float:
    for key in candidate_keys:
        if key in hit.component_scores:
            return hit.component_scores[key]
    return 0.0


def _format_results(
    hits: list[SearchHit], top_k: int, options: ResultOptions
) -> list[SearchHit]:
    """Áp `sort_by` / `display_top_k` — hai field trước PR-04 không ai đọc.

    `display_top_k` chỉ được phép THU HẸP so với `top_k` của request: nó là
    thiết lập hiển thị, không phải cách lách giới hạn top_k.
    """

    ordered = hits
    if options.sort_by == "time":
        ordered = sorted(hits, key=lambda hit: (hit.video_id, hit.start_frame))
    elif options.sort_by != "final_score":
        candidate_keys = _SORT_KEYS[options.sort_by]
        ordered = sorted(
            hits, key=lambda hit: _component_score(hit, candidate_keys), reverse=True
        )
    if options.display_min_score is not None:
        ordered = [hit for hit in ordered if hit.score >= options.display_min_score]
    limit = min(top_k, options.display_top_k)
    return [
        hit.model_copy(update={"rank": rank})
        for rank, hit in enumerate(ordered[:limit], start=1)
    ]


# `_diversify_avs` cũ (chỉ giới hạn N kết quả mỗi video) đã được thay bằng
# `online/services/deduplication.py`: trần theo video chỉ là MỘT chính sách dedup,
# và AVS còn cần dedup theo event chứ không chỉ theo video.
