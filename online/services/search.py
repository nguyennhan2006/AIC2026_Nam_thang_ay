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
from online.domain.search_config import BranchRuntimeOptions, ResultOptions
from online.domain.session import SearchExecutionTrace
from online.ports.interfaces import Retriever, SceneRepository, SessionStore
from online.services.avs import AvsProcessor
from online.services.deduplication import deduplicate_for_task
from online.services.evidence_builder import EvidenceBuilder
from online.services.fusion import fuse_candidates
from online.services.kis import KisProcessor
from online.services.negative_constraints import apply_negative_constraints, extract_negative_constraints
from online.services.playback import DEFAULT_PAD_SEC, build_window
from online.services.qa import QaProcessor
from online.services.query_planner import RuleBasedQueryPlanner, compute_modality_weights
from online.services.registry import RetrieverRegistry
from online.services.normalizers import ScoreNormalizers
from online.services.rerank_pipeline import RerankPipeline
from online.services.score_normalization import normalize_all
from online.services.retrieval_orchestrator import RetrievalOrchestrator, _branch_identity
from online.services.rules import RuleConfig, apply_bonus_penalty
from online.services.thresholding import apply_thresholds
from online.services.temporal import link_event_hits
from online.services.temporal_dp import link_event_hits_dp
from online.services.trake import TrakeProcessor, trake_processor_for_request
from online.services.trake.from_sequences import to_trake_results


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
        qa_processor: QaProcessor | None = None,
        session_store: SessionStore | None = None,
        dataset_version: str | None = None,
        weight_recommender=None,
        evidence_selector=None,
        avs_config=None,
        avs_idf=None,
        playback_pad_sec: float = DEFAULT_PAD_SEC,
        fusion_method: str = "rrf",
        branch_weights: dict[str, float] | None = None,
        fusion_method_qa: str | None = None,
        trake_engine: str = "sequences",
        trake_solver: str = "beam",
        trake_gap_penalty: float | None = None,
        trake_candidate_limit: int | None = None,
        trake_allow_missing_steps: bool = False,
        trake_min_covered_steps: int = 2,
        trake_missing_step_penalty: float | None = None,
        trake_beam_size: int | None = None,
        trake_per_video_beam: int | None = None,
        trake_max_chains_per_video: int | None = None,
        trake_free_gap_sec: float | None = None,
        trake_gap_cap: float | None = None,
        media_root=None,
        branch_timeout_ms: int | None = None,
        evidence_select_top_n: int = 10,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.repository = repository
        self.retrievers = retrievers
        self.registry = RetrieverRegistry(retrievers)
        self.orchestrator = RetrievalOrchestrator(
            retrievers,
            **({"default_timeout_ms": branch_timeout_ms} if branch_timeout_ms else {}),
        )
        # None = không lưu trace (vd script/test không cần replay). Có store
        # thì MỌI lần search (kể cả qua endpoint convenience) đều ghi lại
        # được, vì tất cả đều đi qua đúng một hàm `search()` này.
        self.session_store = session_store
        self.dataset_version = dataset_version
        # Hai "cố vấn" LLM: chỉ đọc kết quả rồi nói lại, KHÔNG đụng vào
        # retrieval. Hỏng thì mất lời khuyên chứ không mất kết quả tìm kiếm.
        self.weight_recommender = weight_recommender
        self.evidence_selector = evidence_selector
        self.evidence_select_top_n = evidence_select_top_n
        self.evidence_builder = evidence_builder or EvidenceBuilder(repository)
        # None = không có tầng rerank nào; cascade vẫn chạy được và chỉ ghi
        # warning "chưa cấu hình", nên hành vi mặc định không đổi.
        self.rerank_pipeline = rerank_pipeline or RerankPipeline(self.evidence_builder)
        # Bốn processor chuyên biệt (PR-07). Chúng chạy SAU lõi retrieval dùng
        # chung, đúng kiến trúc "một lõi + bốn task processor".
        self.kis_processor = KisProcessor()
        self.qa_processor = qa_processor or QaProcessor()
        self.trake_processor = TrakeProcessor()
        self.avs_processor = AvsProcessor(avs_config, idf=avs_idf)
        # Nới cửa sổ phát mỗi phía. Scene p50 chỉ 4.1s — xem đúng 4 giây
        # không đủ để người chấm hiểu bối cảnh.
        self.playback_pad_sec = playback_pad_sec
        self.fusion_method = fusion_method
        self.branch_weights = branch_weights or {}
        self.fusion_method_qa = fusion_method_qa
        self.trake_engine = trake_engine
        # Phase A của docs/31. `beam` = `link_event_hits` (hiện tại),
        # `dp` = quy hoạch động kiểu DANTE, tối ưu toàn cục thay vì cắt tỉa
        # theo từng bước. `trake_gap_penalty=None` giữ mặc định của từng solver.
        self.trake_solver = trake_solver
        # Ba tham số của phạt khoảng cách mềm (dead-zone + tuyến tính + trần).
        # None ở tham số nào thì tham số đó giữ mặc định đã hiệu chuẩn trong
        # `online/services/temporal_gap.py` — cùng bộ số cho cả beam lẫn dp.
        self.trake_gap_penalty = trake_gap_penalty
        # TRAKE cần candidate_limit RIÊNG và LỚN hơn hẳn ba task kia. Lý do là
        # cấu trúc, không phải tinh chỉnh: `link_event_hits` chỉ dựng được chuỗi
        # khi MỌI step có candidate trong CÙNG một video, mà mỗi step lại lấy
        # top-K trên toàn corpus. Với 873 video, K=100 nghĩa là 100 slot rải
        # trên 873 video — số video có mặt ở cả 3 step tụt về ~1, và TRAKE trả
        # về đúng một video, thường là video "nam châm" chứ không phải đáp án.
        #
        # Đo trên mô phỏng 873 video, số video trả về theo K:
        #     K=100 -> 0    K=200 -> 2    K=500 -> 13    K=1000 -> 13
        # Gãy giữa 200 và 500, bão hoà trên 500. Quota candidate/video KHÔNG
        # cứu được (đo 3/5/10, số y hệt) — thiếu là thiếu độ phủ tuyệt đối.
        #
        # Tách khỏi `candidate_limit` chung vì ba task kia được chỉnh ở K=100 và
        # TRAKE chạy retrieval MỘT LẦN MỖI STEP, nên nâng chung là nhân chi phí
        # với số step.
        self.trake_candidate_limit = trake_candidate_limit
        # Đổi bản-sao-gần-trùng lấy ĐỘ PHỦ. Chỉ có nghĩa với beam: `dp` giải
        # riêng từng video nên nó không có hiện tượng một video chiếm beam.
        # Chuỗi thiếu step vẫn được trả về. Xem docstring `link_event_hits`.
        self.trake_allow_missing_steps = trake_allow_missing_steps
        self.trake_min_covered_steps = trake_min_covered_steps
        self.trake_missing_step_penalty = trake_missing_step_penalty
        self.trake_beam_size = trake_beam_size
        self.trake_per_video_beam = trake_per_video_beam
        self.trake_max_chains_per_video = trake_max_chains_per_video
        self.trake_free_gap_sec = trake_free_gap_sec
        self.trake_gap_cap = trake_gap_cap
        self.media_root = media_root
        self._last_avs_diagnostics: dict = {}
        self.planner = planner or RuleBasedQueryPlanner()
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        # Phương án E (bonus/penalty sau RRF), optional — None giữ nguyên hành vi
        # cũ; xem online/services/rules.py và docs/15_RESEARCH_AGENDA.md mục 5.
        self.rule_config = rule_config

    def _apply_default_branch_weights(self, plan: QueryPlan) -> QueryPlan:
        """Gắn trọng số MẶC ĐỊNH mức triển khai vào plan, request vẫn thắng.

        Vì sao phải làm ở đây chứ không ở fusion: `effective_weight` vừa quyết
        điểm fusion VỪA quyết nhánh có chạy hay không (weight <= 0 -> trả rỗng).
        Đặt vào `plan.search_options.branches` là cả hai chỗ cùng thấy một giá
        trị, không có đường nào đọc lệch.

        Chỉ điền cho nhánh request KHÔNG khai — `model_fields_set` không dùng
        được ở đây vì `branches` là dict, nên "có khoá" chính là "đã khai".

        Đây là cách nói *"tính năng này trọng số thấp nhưng BẮT BUỘC có mặt"*,
        thay cho việc tắt hẳn nhánh. Nhánh tắt thì không bao giờ cứu được truy
        vấn mà chỉ nó tìm ra.
        """

        if not self.branch_weights:
            return plan
        branches = dict(plan.search_options.branches)
        changed = False
        for branch_id, weight in self.branch_weights.items():
            if branch_id in branches:
                continue
            branches[branch_id] = BranchRuntimeOptions(weight=weight)
            changed = True
        if not changed:
            return plan
        options = plan.search_options.model_copy(update={"branches": branches})
        return plan.model_copy(update={"search_options": options})

    def _fusion_method_for(self, task: TaskType | None) -> str:
        """Fusion method mặc định của deployment, cho phép ghi đè theo TASK.

        Cần tách theo task vì đo được (Phase D, docs/31) là các task muốn hai
        thứ khác nhau, và mâu thuẫn đó nhất quán trên cả holdout::

                       KIS R@1   TRAKE mean_r   AVS nDCG   QA joint
            rrf          0.583          0.254      0.558      0.472
            norm_sum     0.611          0.315      0.606      0.389
            norm_max     0.750          0.354      0.565      0.389

        Giải thích hợp lý: KIS/TRAKE hỏi "khung hình nào ĐÚNG nhất" nên một
        nhánh rất chắc chắn phải được thắng — `norm_max` cho đúng điều đó. QA
        hỏi "scene nào là BẰNG CHỨNG tốt" nên đồng thuận nhiều nhánh đáng tin
        hơn một nhánh đơn độc, và RRF vốn thưởng cho đồng thuận.
        """

        if task == TaskType.QA and self.fusion_method_qa:
            return self.fusion_method_qa
        return self.fusion_method

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
        # Cùng quy ước với `rrf_k`: mặc định mức triển khai (AIC_FUSION_METHOD)
        # chỉ bị ghi đè khi request ĐẶT TƯỜNG MINH `fusion.method`, nên request
        # không kèm search_options vẫn dùng đúng cấu hình server.
        method = (
            fusion_options.method
            if "method" in fusion_options.model_fields_set
            else self._fusion_method_for(plan.task)
        )
        candidates = fuse_candidates(
            lists,
            plan.modality_weights,
            method=method,
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
        plan = self._apply_default_branch_weights(await self.planner.plan(request))
        if task == TaskType.TRAKE and len(plan.events) >= 2:
            step_overrides = plan.search_options.temporal.step_modality_weights
            event_hit_lists: list[list[SearchHit]] = []
            statuses: list[BranchStatus] = []
            for index, event in enumerate(plan.events):
                # PR-14A: trọng số modality riêng cho TỪNG step — trước đây mọi
                # step dùng chung weight suy từ cả câu multi-event, nên step
                # không có OCR/ASR vẫn bị đẩy nhánh sai theo step khác.
                weights = compute_modality_weights(
                    event.text, event.exact_phrases,
                    allow_zero=getattr(self.planner, "allow_zero_modality", True),
                )
                if index < len(step_overrides):
                    weights.update(step_overrides[index])
                event_plan = plan.model_copy(
                    update={
                        "normalized_query": event.text,
                        "events": [event],
                        "modality_weights": weights,
                    }
                )
                candidates, step_statuses = await self._retrieve(
                    event_plan, self.trake_candidate_limit or self.candidate_limit
                )
                statuses = _merge_statuses(statuses, step_statuses)
                event_hit_lists.append(await self._hydrate(candidates, event.text))
            documents = await self._documents_for(
                [hit for hits in event_hit_lists for hit in hits]
            )
            # Stage A khóa video trước, rồi mới beam search trong video đó và
            # tinh chỉnh frame — thay cho link_event_hits chỉ nối scene.
            trake_processor = trake_processor_for_request(
                self.trake_processor, plan.search_options.temporal
            )
            processor_trake = trake_processor.run(
                [event.text for event in plan.events],
                event_hit_lists,
                documents,
                limit=request.top_k,
            )
            linker = link_event_hits_dp if self.trake_solver == "dp" else link_event_hits
            # Chỉ truyền tham số ĐƯỢC ĐẶT TƯỜNG MINH; thiếu thì để mặc định của
            # `temporal_gap` nói lên tiếng nói cuối. Trước đây nhánh `dp` ép
            # `gap_penalty=0.0` khi không cấu hình gì, nên beam và dp chạy hai
            # hàm mục tiêu khác nhau mà không ai thấy.
            #
            # `search_options.temporal` thắng cấu hình deployment. Trước đây ba
            # tham số này CHỈ tới được `TrakeProcessor`, nên chạy ablation bằng
            # `--trake-gap-penalty` trên engine mặc định (`sequences`) là chỉnh
            # một tham số KHÔNG nằm trên đường chạy — cờ im lặng không làm gì và
            # bảng số trông như "đổi gì cũng không ảnh hưởng".
            temporal_set = plan.search_options.temporal.model_fields_set
            overrides = {}
            for argument, option_name, deployment_value in (
                ("gap_penalty", "gap_penalty_per_sec", self.trake_gap_penalty),
                ("free_gap_sec", "free_gap_sec", self.trake_free_gap_sec),
                ("max_gap_penalty", "gap_penalty_cap", self.trake_gap_cap),
            ):
                requested = getattr(plan.search_options.temporal, option_name, None)
                if option_name in temporal_set and requested is not None:
                    overrides[argument] = requested
                elif deployment_value is not None:
                    overrides[argument] = deployment_value
            if linker is link_event_hits:
                if self.trake_allow_missing_steps:
                    overrides["allow_missing_steps"] = True
                    overrides["min_covered_steps"] = self.trake_min_covered_steps
                    if self.trake_missing_step_penalty is not None:
                        overrides["missing_step_penalty"] = self.trake_missing_step_penalty
                for argument, value in (
                    ("beam_size", self.trake_beam_size),
                    ("per_video_beam", self.trake_per_video_beam),
                    ("max_chains_per_video", self.trake_max_chains_per_video),
                ):
                    if value is not None:
                        overrides[argument] = value
            sequences = linker(event_hit_lists, limit=request.top_k, **overrides)
            # Đường CŨ (`link_event_hits`) đo ra TỐT HƠN đường đã thay thế nó:
            # video_recall@1 0.833 so với 0.542, và gấp đôi trên hai video
            # holdout. Bảng số đầy đủ ở `trake/from_sequences.py`.
            #
            # Sai lầm ẩn được lâu vì bộ chấm chỉ chấm `response.trake`; đường
            # cũ vẫn chạy và vẫn nằm trong `response.sequences` nhưng chưa bao
            # giờ ai chấm nó. `AIC_TRAKE_ENGINE=processor` để quay lại khi
            # `TrakeProcessor` được sửa — ý tưởng của nó vẫn đúng.
            trake = (
                processor_trake
                if self.trake_engine == "processor"
                else to_trake_results(
                    sequences, expected_steps=len(plan.events), documents=documents
                )
            )
            await _attach_playback(self.repository, trake, self.playback_pad_sec, self.media_root)
            warnings = _status_warnings(statuses)
            if not trake:
                if not any(event_hit_lists):
                    warnings.append(
                        "TRAKE: không có candidate cho bất kỳ step nào — kiểm tra "
                        "retrieval branch (dense_visual/bm25) có trả kết quả không"
                    )
                else:
                    warnings.append(
                        "TRAKE: có candidate cho step nhưng không dựng được chuỗi "
                        "hợp lệ — kiểm tra ràng buộc thứ tự tăng dần/khoảng cách "
                        "(SequenceConfig.max_gap_sec, min_gap_frames)"
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
        # P2: chụp thứ hạng TRƯỚC rerank. Phải chụp tại đây, không suy ngược
        # được từ kết quả cuối — rerank và task processor đều sắp lại. Thiếu nó
        # thì eval chỉ thấy điểm cuối, không biết candidate đúng rơi ở TẦNG NÀO,
        # mà "recall đủ nhưng xếp sai" với "không tìm ra" cần hai cách sửa khác
        # hẳn nhau.
        trace: dict | None = None
        if request.debug:
            trace = {
                "branch_latency_ms": {st.execution_id: st.latency_ms for st in statuses},
                "branch_state": {st.execution_id: st.state for st in statuses},
                "branch_count": {st.execution_id: st.candidate_count for st in statuses},
                "prefusion_total": sum(st.candidate_count for st in statuses),
                "fused": [
                    {
                        "candidate_id": c.candidate_id,
                        "video_id": c.video_id,
                        "score": round(c.raw_score, 6),
                        "n_branches": len(c.payload.get("matched_branches") or ()),
                    }
                    for c in candidates[:100]
                ],
            }
        fusion_options = plan.search_options.fusion
        # Chuẩn hoá điểm phải chốt TRƯỚC dedup: nếu tính sau, mẫu số đổi theo
        # `max_results_per_video` và nới cap sẽ làm xáo trộn cả thứ hạng đã có
        # (EVAL-01 prefix invariance).
        normalizers = ScoreNormalizers.from_pool(candidates)
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
        if trace is not None:
            trace["post_rerank"] = [
                {"candidate_id": c.candidate_id, "video_id": c.video_id,
                 "score": round(c.raw_score, 6)}
                for c in candidates[:100]
            ]
        hits = await self._hydrate(candidates[: request.top_k], plan.normalized_query)
        results = _format_results(hits, request.top_k, plan.search_options.results)
        warnings = (
            _status_warnings(statuses)
            + rerank.warnings
            + [warning for hit in results for warning in hit.warnings]
        )
        # Xoá TRƯỚC khi chạy processor. Xoá sau là xoá đúng thứ vừa ghi —
        # lỗi đã mắc: mọi số đo pre/post gate về 0 trong khi cơ chế vẫn chạy.
        self._last_avs_diagnostics = {}
        task_results, task_warnings = await self._run_task_processor(
            task, plan, request, results, candidates, rerank.packs, normalizers
        )
        warnings = warnings + task_warnings
        # `results` là danh sách UI hiển thị cho mọi task không phải TRAKE.
        # Không gắn ở đây thì UI phải tự suy cửa sổ phát và tự đoán phần nới.
        await _attach_playback(self.repository, results, self.playback_pad_sec, self.media_root)
        recommended_weights = await self._recommend_weights(request, plan, task)
        selected_evidence = await self._select_evidence(
            request, plan, candidates, rerank.packs
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
            recommended_weights=recommended_weights,
            selected_evidence=selected_evidence,
            avs_diagnostics=self._last_avs_diagnostics or None,
            pipeline_trace=trace,
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

        plan = self._apply_default_branch_weights(await self.planner.plan(request))
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
        normalizers = ScoreNormalizers.from_pool(candidates)
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
        task_results, task_warnings = await self._run_task_processor(
            task, plan, request, results, candidates, rerank.packs, normalizers
        )
        warnings = warnings + task_warnings
        await _attach_playback(self.repository, results, self.playback_pad_sec, self.media_root)
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

    async def _recommend_weights(self, request, plan, task) -> dict | None:
        """Đề xuất trọng số nhánh cho truy vấn này — chỉ khi được hỏi.

        Danh sách nhánh lấy từ retriever ĐANG đăng ký, không phải một bảng
        cứng: cấu hình tắt nhánh nào thì LLM cũng không được đề xuất nhánh đó,
        nếu không người dùng nhận về trọng số cho thứ không tồn tại.
        """

        if not request.recommend_weights or self.weight_recommender is None:
            return None
        branch_ids = sorted(
            {
                getattr(retriever, "branch_id", None) or retriever.name
                for retriever in self.retrievers
            }
        )
        return await self.weight_recommender.recommend(
            plan.normalized_query, task=task.value, branch_ids=branch_ids
        )

    async def _select_evidence(self, request, plan, candidates, packs) -> list[dict]:
        """Lọc bằng chứng thô xuống phần thật sự liên quan — chỉ khi được hỏi.

        Dùng lại pack mà rerank đã dựng nếu có; chỉ dựng thêm cho candidate
        chưa có pack. Dựng lại từ đầu là đọc lại repository cho cùng một scene.
        """

        if not request.select_evidence or self.evidence_selector is None:
            return []
        head = candidates[: self.evidence_select_top_n]
        if not head:
            return []
        resolved = []
        for candidate in head:
            pack = packs.get(candidate.candidate_id) if packs else None
            if pack is None:
                pack = await self.evidence_builder.build(candidate)
            if pack is not None:
                resolved.append(pack)
        if not resolved:
            return []
        selected = await self.evidence_selector.select_many(plan.normalized_query, resolved)
        return [item for item in selected if item is not None]

    async def _run_task_processor(
        self,
        task: TaskType,
        plan: QueryPlan,
        request: SearchRequest,
        results: list[SearchHit],
        candidates: list[Candidate],
        packs: dict,
        normalizers: ScoreNormalizers,
    ) -> tuple[dict, list[str]]:
        """Chạy processor chuyên biệt của task trên kết quả đã rerank.

        Trả kèm `warnings` riêng của processor (vd FPT QA LLM lỗi/fallback) —
        tách khỏi warnings cấp branch/rerank vì processor chạy SAU khi
        `warnings` chính đã được caller tính, xem hai điểm gọi ở `search()`
        và `search_stream()`.
        """

        if not results:
            return {}, []
        if task == TaskType.TEXTUAL_KIS:
            documents = await self._documents_for(results)
            kis_results = self.kis_processor.rank(
                plan.original_query, results, documents,
                packs=packs, limit=request.top_k, normalizers=normalizers,
            )
            await _attach_playback(self.repository, kis_results, self.playback_pad_sec, self.media_root)
            return {"kis": kis_results}, []

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
            qa_results, qa_warnings = await self.qa_processor.answer_async(
                plan.original_query, evidence_packs,
                frame_scores=scores, limit=request.top_k, normalizers=normalizers,
            )
            await _attach_playback(self.repository, qa_results, self.playback_pad_sec, self.media_root)
            return {"qa": qa_results}, qa_warnings
        if task == TaskType.AVS:
            # AVS-GRADE-01: đổ số liệu trước/sau cổng grade vào response. Không
            # có nó thì không phân biệt được "cổng từ vựng loại mất candidate
            # đúng" với "pool vốn không có candidate nào tốt" — hai nguyên nhân
            # của cùng một `zero_result_rate`.
            diagnostics: dict = {}
            avs_results = self.avs_processor.rank(
                plan.original_query, evidence_packs,
                retrieval_scores=scores, limit=request.top_k,
                normalizers=normalizers, diagnostics=diagnostics,
            )
            self._last_avs_diagnostics = diagnostics
            await _attach_playback(self.repository, avs_results, self.playback_pad_sec, self.media_root)
            return {"avs": avs_results}, []
        return {}, []


async def _attach_playback(
    repository: SceneRepository, items: list, pad_sec: float,
    media_root=None,
) -> None:
    """Gắn `playback` vào từng kết quả, tại chỗ.

    Bốn task mang thông tin thời gian theo bốn cách khác nhau, nên phải quy về
    một chỗ thay vì để UI tự đoán:

        KIS/QA   `scene_id` + `frame_idx`
        AVS      `segment_id` (chính là scene_id) + `start_frame`/`end_frame`
        TRAKE    `frame_ids` — trải nhiều scene, phải tra scene của frame đầu

    Thiếu `video_path` (V002/V003 hiện chỉ có ảnh, không có mp4) thì để `None`
    chứ không trả URL hỏng — UI cần phân biệt "chưa có video" với "phát lỗi".
    """

    if not items:
        return
    scene_ids = {
        getattr(item, "scene_id", None) or getattr(item, "segment_id", None)
        for item in items
    }
    scene_ids.discard(None)
    documents = {
        document.scene_id: document
        for document in await repository.get_many(sorted(scene_ids))
    }
    trake_videos = {
        item.video_id for item in items if getattr(item, "frame_ids", None)
    }
    if trake_videos:
        for document in await repository.all():
            if document.video_id in trake_videos:
                documents.setdefault(document.scene_id, document)

    for item in items:
        frame_ids = getattr(item, "frame_ids", None)
        if frame_ids:
            first, last = min(frame_ids), max(frame_ids)
            scene = next(
                (
                    d for d in documents.values()
                    if d.video_id == item.video_id
                    and d.start_frame <= first < d.end_frame_exclusive
                ),
                None,
            )
            if scene is None:
                continue
            item.playback = build_window(
                scene, focus_frame=first, start_frame=first, end_frame=last,
                pad_sec=pad_sec, media_root=media_root,
            )
            continue

        key = getattr(item, "scene_id", None) or getattr(item, "segment_id", None)
        scene = documents.get(key)
        if scene is None:
            continue
        item.playback = build_window(
            scene,
            focus_frame=getattr(item, "frame_idx", None) or getattr(item, "best_frame_idx", None),
            start_frame=getattr(item, "start_frame", None),
            end_frame=getattr(item, "end_frame", None),
            pad_sec=pad_sec, media_root=media_root,
        )


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
