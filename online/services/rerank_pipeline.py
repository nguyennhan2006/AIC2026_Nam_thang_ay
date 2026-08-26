"""Rerank cascade nhiều tầng, có fallback tường minh (PR-06).

    Stage 0  rules          (đã có, online/services/rules.py — chạy trước fusion)
    Stage 1  text reranker  top 300 -> reorder + preserve unscored
    Stage 2  VLM reranker   top 20 -> reorder + preserve unscored
    Stage 3  temporal       chỉ khi query có thứ tự/before-after

Mỗi tầng thu hẹp dần: tầng rẻ lọc cho tầng đắt. Tầng nào không cấu hình hoặc
gọi lỗi thì **giữ nguyên thứ hạng của tầng trước và ghi warning** — không bao
giờ trả về danh sách rỗng hay im lặng bỏ qua (nguyên tắc "no silent
degradation" của docs/11_SERVER_IMPLEMENTATION.md).

**PR-RECALL**: Reranker CHỈ có quyền đổi thứ tự, không có quyền làm mất
candidate. Unscored candidates (thiếu evidence, timeout, etc.) được giữ lại
với combined score = fusion_score. Backfill từ pre-rerank ranking đảm bảo
không bao giờ trả < đầu vào.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from online.domain.candidate import Candidate
from online.domain.evidence import EvidencePack
from online.domain.search_config import RerankOptions
from online.errors import DependencyUnavailableError
from online.services.evidence_builder import EvidenceBuilder


# Trọng số kết hợp: fusion_score * ALPHA + rerank_score * BETA
# Để rerank có ảnh hưởng nhưng không át hoàn toàn fusion score
DEFAULT_ALPHA = 0.7  # fusion score weight
DEFAULT_BETA = 0.3   # rerank score weight


@dataclass(slots=True)
class RerankStageResult:
    stage: str
    applied: bool
    input_count: int
    output_count: int
    latency_ms: int
    rerankable: int = 0      # số candidate được rerank thực sự
    unscored: int = 0        # số candidate giữ nguyên (không có context)
    warning: str | None = None


@dataclass(slots=True)
class RerankOutcome:
    candidates: list[Candidate]
    stages: list[RerankStageResult] = field(default_factory=list)
    packs: dict[str, EvidencePack] = field(default_factory=dict)
    pre_rerank: list[Candidate] = field(default_factory=list)  # PR-RECALL: lưu ranking trước rerank

    @property
    def warnings(self) -> list[str]:
        return [
            f"rerank.{item.stage}: {item.warning}" for item in self.stages if item.warning
        ]


def backfill_top_k(
    ranked: list[Candidate],
    fallback: list[Candidate],
    k: int = 100,
) -> list[Candidate]:
    """PR-RECALL: Backfill từ fallback ranking khi ranked < k.

    Đảm bảo output luôn có ít nhất min(k, len(fallback)) candidates.
    """
    output: list[Candidate] = []
    seen: set[str] = set()

    def add(candidate: Candidate) -> None:
        key = candidate.candidate_id
        if key in seen:
            return
        seen.add(key)
        output.append(candidate)

    # Thêm từ ranked trước
    for candidate in ranked:
        add(candidate)
        if len(output) >= k:
            return output[:k]

    # Backfill từ fallback
    for candidate in fallback:
        add(candidate)
        if len(output) >= k:
            break

    return output[:k]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Chuẩn hóa scores về [0, 1] dựa trên min-max của tập.

    PR-RECALL: Dùng min-max để rerank_score có ảnh hưởng tương đối đúng.
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-12:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _compute_combined_scores(
    candidates: list[Candidate],
    scores: list[float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> list[tuple[Candidate, float]]:
    """Kết hợp fusion_score với rerank_score.

    Candidates có rerank_score: combined = alpha * fusion + beta * rerank_norm
    Candidates không có rerank_score: combined = fusion_score (giữ nguyên vị trí tương đối)

    PR-RECALL: Dùng min-max normalization để rerank_score có ảnh hưởng thực sự.
    """
    if not scores or not candidates:
        return [(c, c.raw_score) for c in candidates]

    # Chuẩn hóa rerank scores về [0, 1] trong phạm vi batch này
    rerank_norm = _normalize_scores(scores)

    # Map candidate_id -> normalized rerank score
    scored_map = {c.candidate_id: n for c, n in zip(candidates, rerank_norm)}
    result: list[tuple[Candidate, float]] = []

    for candidate in candidates:
        if candidate.candidate_id in scored_map:
            rerank_val = scored_map[candidate.candidate_id]
            fusion_score = candidate.raw_score
            # Kết hợp: fusion có base weight, rerank điều chỉnh
            combined = alpha * fusion_score + beta * rerank_val
            result.append((candidate, combined))
        else:
            # Unscored: giữ nguyên fusion_score
            result.append((candidate, candidate.raw_score))

    return result


def _reorder(
    candidates: list[Candidate],
    scores: list[float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> list[Candidate]:
    """Sắp lại theo điểm kết hợp (fusion + rerank).

    PR-RECALL: Unscored candidates được giữ lại với fusion_score nguyên vẹn,
    không bị đẩy xuống cuối hoàn toàn.
    """
    # Build map để lấy rerank_score đúng cho từng candidate
    scored_map = {c.candidate_id: s for c, s in zip(candidates, scores)}

    paired = _compute_combined_scores(candidates, scores, alpha, beta)
    paired = sorted(paired, key=lambda item: -item[1])
    return [
        candidate.model_copy(
            update={
                "rank": rank,
                "payload": {
                    **candidate.payload,
                    "rerank_score": scored_map.get(candidate.candidate_id),
                    "combined_score": score,
                },
            }
        )
        for rank, (candidate, score) in enumerate(paired, start=1)
    ]


class RerankPipeline:
    """Chạy cascade cho một danh sách candidate đã fuse + dedup.

    PR-RECALL: Mỗi stage rerank có thể đổi thứ tự nhưng KHÔNG được làm mất
    candidate. Unscored candidates (thiếu context) được kết hợp với fusion_score
    thay vì bị loại bỏ. Backfill đảm bảo output luôn >= đầu vào.
    """

    def __init__(
        self,
        evidence_builder: EvidenceBuilder,
        *,
        text_reranker=None,
        vlm_reranker=None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self.evidence_builder = evidence_builder
        self.text_reranker = text_reranker
        self.vlm_reranker = vlm_reranker
        self.alpha = alpha
        self.beta = beta

    @property
    def available_stages(self) -> dict[str, bool]:
        return {"text": self.text_reranker is not None, "vlm": self.vlm_reranker is not None}

    async def _packs_for(
        self, candidates: list[Candidate]
    ) -> tuple[list[Candidate], list[EvidencePack]]:
        """Build evidence packs, returning only candidates that got packs.

        Returns:
            Tuple of (valid_candidates, valid_packs) — same length, 1:1 mapping.
        """
        valid_candidates: list[Candidate] = []
        valid_packs: list[EvidencePack] = []
        for candidate in candidates:
            # Không cần neighbor context ở tầng rerank: nó chỉ làm dài prompt.
            pack = await self.evidence_builder.build(candidate, with_neighbors=False)
            if pack is not None:
                valid_candidates.append(candidate)
                valid_packs.append(pack)
        return valid_candidates, valid_packs

    async def run(
        self, query: str, candidates: list[Candidate], options: RerankOptions
    ) -> RerankOutcome:
        # PR-RECALL: Lưu ranking TRƯỚC rerank để backfill nếu cần
        outcome = RerankOutcome(candidates=candidates, pre_rerank=list(candidates))
        if not candidates:
            return outcome

        # PR-RECALL: Lấy alpha/beta từ options hoặc dùng instance defaults
        alpha = options.alpha if hasattr(options, 'alpha') else self.alpha
        beta = options.beta if hasattr(options, 'beta') else self.beta
        preserve_unscored = options.preserve_unscored if hasattr(options, 'preserve_unscored') else True

        # --- Stage 1: text cross-encoder ---
        if options.text.enabled and self.text_reranker is not None:
            head = outcome.candidates[: options.text.input_top_k]
            tail = outcome.candidates[options.text.input_top_k :]
            started = perf_counter()
            try:
                valid_head, valid_packs = await self._packs_for(head)

                # PR-RECALL: Tách rerankable và unscored
                valid_ids = {c.candidate_id for c in valid_head}
                unscored = [c for c in head if c.candidate_id not in valid_ids]

                if not valid_packs:
                    # PR-RECALL: Không có packs nào — giữ tất cả candidates với fusion_score
                    outcome.packs.update({})
                    outcome.candidates = _renumber(head + tail)
                    # Thất bại nhẹ: có evidence packs nhưng reranker không chạy được
                    outcome.stages.append(RerankStageResult(
                        "text",
                        False,
                        len(head),
                        len(outcome.candidates),
                        int((perf_counter() - started) * 1000),
                        rerankable=0,
                        unscored=len(head),
                        warning="no valid evidence packs — candidates kept with fusion_score",
                    ))
                else:
                    scores = await self.text_reranker.score(query, valid_packs)
                    # PR-RECALL: Contract check
                    if len(scores) != len(valid_head):
                        raise RuntimeError(
                            f"text reranker contract violation: "
                            f"rerankable={len(valid_head)}, scores={len(scores)}"
                        )
                    # PR-RECALL: Kết hợp scores, giữ unscored
                    reranked = _reorder(valid_head, scores, alpha=alpha, beta=beta)
                    # PR-RECALL: Unscored giữ nguyên fusion_score
                    if preserve_unscored:
                        outcome.candidates = _renumber(reranked + unscored + tail)
                    else:
                        # Legacy behavior: đẩy unscored xuống cuối
                        outcome.candidates = _renumber(reranked + tail)
                    outcome.packs.update({pack.candidate_id: pack for pack in valid_packs})
                    outcome.stages.append(RerankStageResult(
                        "text",
                        True,
                        len(head),
                        len(outcome.candidates),
                        int((perf_counter() - started) * 1000),
                        rerankable=len(valid_head),
                        unscored=len(unscored),
                        warning=f"{len(unscored)} unscored candidates kept with fusion_score" if unscored else None,
                    ))
            except DependencyUnavailableError as exc:
                outcome.stages.append(RerankStageResult(
                    "text", False, len(head), len(candidates),
                    int((perf_counter() - started) * 1000),
                    rerankable=0,
                    unscored=0,
                    warning=f"{exc} — giữ nguyên thứ hạng sau fusion",
                ))
        elif options.text.enabled:
            outcome.stages.append(RerankStageResult(
                "text", False, len(candidates), len(candidates), 0,
                rerankable=0,
                unscored=0,
                warning="chưa cấu hình text reranker (AIC_RERANK_TEXT_URL)",
            ))

        # --- Stage 2: VLM ---
        if options.vlm.enabled and self.vlm_reranker is not None:
            head = outcome.candidates[: options.vlm.input_top_k]
            tail = outcome.candidates[options.vlm.input_top_k :]
            started = perf_counter()
            try:
                valid_head, valid_packs = await self._packs_for(head)

                # PR-RECALL: Tách rerankable và unscored
                valid_ids = {c.candidate_id for c in valid_head}
                unscored = [c for c in head if c.candidate_id not in valid_ids]

                if not valid_packs:
                    outcome.stages.append(RerankStageResult(
                        "vlm", True, len(head), len(head),
                        int((perf_counter() - started) * 1000),
                        rerankable=0,
                        unscored=len(head),
                        warning="no valid evidence packs for VLM — candidates kept with fusion_score",
                    ))
                    outcome.candidates = _renumber(head + tail)
                else:
                    results = await self.vlm_reranker.score(query, valid_packs)
                    scores = [float(item.get("relevance", 0.0)) for item in results]
                    # PR-RECALL: Contract check
                    if len(scores) != len(valid_head):
                        raise RuntimeError(
                            f"VLM reranker contract violation: "
                            f"rerankable={len(valid_head)}, scores={len(scores)}"
                        )
                    reranked = _reorder(valid_head, scores, alpha=alpha, beta=beta)
                    # PR-RECALL: Enrich và giữ unscored
                    enriched = [
                        c.model_copy(
                            update={
                                "payload": {
                                    **c.payload,
                                    "vlm_verdict": {
                                        key: value
                                        for key, value in results[index].items()
                                        if key != "candidate_id"
                                    },
                                }
                            }
                        )
                        for index, c in enumerate(reranked)
                    ]
                    if preserve_unscored:
                        outcome.candidates = _renumber(enriched + unscored + tail)
                    else:
                        # Legacy behavior: đẩy unscored xuống cuối
                        outcome.candidates = _renumber(enriched + tail)
                    outcome.packs.update({pack.candidate_id: pack for pack in valid_packs})
                    outcome.stages.append(RerankStageResult(
                        "vlm", True, len(head), len(head),
                        int((perf_counter() - started) * 1000),
                        rerankable=len(valid_head),
                        unscored=len(unscored),
                    ))
            except DependencyUnavailableError as exc:
                outcome.stages.append(RerankStageResult(
                    "vlm", False, len(head), len(candidates),
                    int((perf_counter() - started) * 1000),
                    rerankable=0,
                    unscored=0,
                    warning=f"{exc} — rơi về kết quả của stage text",
                ))
        elif options.vlm.enabled:
            outcome.stages.append(RerankStageResult(
                "vlm", False, len(candidates), len(candidates), 0,
                rerankable=0,
                unscored=0,
                warning="chưa cấu hình VLM reranker (AIC_RERANK_VLM_URL)",
            ))

        # PR-RECALL: Backfill nếu output < input
        pre_count = len(outcome.pre_rerank)
        post_count = len(outcome.candidates)
        if post_count < pre_count:
            outcome.candidates = backfill_top_k(
                ranked=outcome.candidates,
                fallback=outcome.pre_rerank,
                k=pre_count,
            )
            outcome.stages.append(RerankStageResult(
                "backfill",
                True,
                pre_count,
                len(outcome.candidates),
                0,
                rerankable=0,
                unscored=0,
                warning=f"backfill: {post_count} -> {len(outcome.candidates)}",
            ))

        return outcome


def _renumber(candidates: list[Candidate]) -> list[Candidate]:
    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(candidates, start=1)
    ]


__all__ = ["RerankOutcome", "RerankPipeline", "RerankStageResult", "backfill_top_k"]
