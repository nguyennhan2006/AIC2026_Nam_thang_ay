"""Rerank cascade nhiều tầng, có fallback tường minh (PR-06).

    Stage 0  rules          (đã có, online/services/rules.py — chạy trước fusion)
    Stage 1  text reranker  top 300 -> 80
    Stage 2  VLM reranker   top 20
    Stage 3  temporal       chỉ khi query có thứ tự/before-after

Mỗi tầng thu hẹp dần: tầng rẻ lọc cho tầng đắt. Tầng nào không cấu hình hoặc
gọi lỗi thì **giữ nguyên thứ hạng của tầng trước và ghi warning** — không bao
giờ trả về danh sách rỗng hay im lặng bỏ qua (nguyên tắc "no silent
degradation" của docs/11_SERVER_IMPLEMENTATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from online.domain.candidate import Candidate
from online.domain.evidence import EvidencePack
from online.domain.search_config import RerankOptions
from online.errors import DependencyUnavailableError
from online.services.evidence_builder import EvidenceBuilder


@dataclass(slots=True)
class RerankStageResult:
    stage: str
    applied: bool
    input_count: int
    output_count: int
    latency_ms: int
    warning: str | None = None


@dataclass(slots=True)
class RerankOutcome:
    candidates: list[Candidate]
    stages: list[RerankStageResult] = field(default_factory=list)
    packs: dict[str, EvidencePack] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        return [
            f"rerank.{item.stage}: {item.warning}" for item in self.stages if item.warning
        ]


def _reorder(candidates: list[Candidate], scores: list[float]) -> list[Candidate]:
    """Sắp lại theo điểm rerank; điểm bằng nhau giữ thứ tự cũ (sort ổn định)."""

    paired = sorted(
        zip(candidates, scores, strict=True), key=lambda item: -item[1]
    )
    return [
        candidate.model_copy(
            update={
                "rank": rank,
                "payload": {**candidate.payload, "rerank_score": score},
            }
        )
        for rank, (candidate, score) in enumerate(paired, start=1)
    ]


class RerankPipeline:
    """Chạy cascade cho một danh sách candidate đã fuse + dedup."""

    def __init__(
        self,
        evidence_builder: EvidenceBuilder,
        *,
        text_reranker=None,
        vlm_reranker=None,
    ) -> None:
        self.evidence_builder = evidence_builder
        self.text_reranker = text_reranker
        self.vlm_reranker = vlm_reranker

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
        outcome = RerankOutcome(candidates=candidates)
        if not candidates:
            return outcome

        # --- Stage 1: text cross-encoder ---
        if options.text.enabled and self.text_reranker is not None:
            head = outcome.candidates[: options.text.input_top_k]
            tail = outcome.candidates[options.text.input_top_k :]
            started = perf_counter()
            try:
                valid_head, valid_packs = await self._packs_for(head)

                if len(valid_head) < len(head):
                    # Some candidates have no evidence — put them at the end, unscored
                    unscored = [c for c in head if c not in valid_head]
                    outcome.stages.append(RerankStageResult(
                        "text",
                        True,
                        len(head),
                        len(valid_head),
                        int((perf_counter() - started) * 1000),
                        warning=f"text reranker skipped {len(unscored)}/{len(head)} candidates without evidence",
                    ))
                else:
                    outcome.stages.append(RerankStageResult(
                        "text", True, len(head), len(valid_head),
                        int((perf_counter() - started) * 1000),
                    ))

                if not valid_packs:
                    # No valid packs at all — keep head as-is
                    outcome.packs.update({pack.candidate_id: pack for pack in valid_packs})
                else:
                    scores = await self.text_reranker.score(query, valid_packs)
                    if len(scores) != len(valid_head):
                        raise ValueError(
                            f"text reranker cardinality mismatch: "
                            f"{len(valid_head)} valid candidates but {len(scores)} scores"
                        )
                    reranked = _reorder(valid_head, scores)
                    # Giữ TẤT CẢ candidates từ head (không cắt ở output_top_k)
                    # unscored candidates được đặt ở cuối phần reranked
                    outcome.candidates = _renumber(reranked + unscored + tail)
                    outcome.packs.update({pack.candidate_id: pack for pack in valid_packs})
            except DependencyUnavailableError as exc:
                outcome.stages.append(RerankStageResult(
                    "text", False, len(head), len(head),
                    int((perf_counter() - started) * 1000),
                    warning=f"{exc} — giữ nguyên thứ hạng sau fusion",
                ))
        elif options.text.enabled:
            outcome.stages.append(RerankStageResult(
                "text", False, len(candidates), len(candidates), 0,
                warning="chưa cấu hình text reranker (AIC_RERANK_TEXT_URL)",
            ))

        # --- Stage 2: VLM ---
        if options.vlm.enabled and self.vlm_reranker is not None:
            head = outcome.candidates[: options.vlm.input_top_k]
            tail = outcome.candidates[options.vlm.input_top_k :]
            started = perf_counter()
            try:
                valid_head, valid_packs = await self._packs_for(head)

                if not valid_packs:
                    outcome.stages.append(RerankStageResult(
                        "vlm", True, len(head), 0,
                        int((perf_counter() - started) * 1000),
                        warning="no valid evidence packs for VLM reranking",
                    ))
                else:
                    results = await self.vlm_reranker.score(query, valid_packs)
                    scores = [float(item.get("relevance", 0.0)) for item in results]
                    if len(scores) != len(valid_head):
                        raise ValueError(
                            f"VLM reranker cardinality mismatch: "
                            f"{len(valid_head)} valid candidates but {len(scores)} scores"
                        )
                    reranked = _reorder(valid_head, scores)
                    enriched = [
                        candidate.model_copy(
                            update={
                                "payload": {
                                    **candidate.payload,
                                    "vlm_verdict": {
                                        key: value
                                        for key, value in results[index].items()
                                        if key != "candidate_id"
                                    },
                                }
                            }
                        )
                        for index, candidate in enumerate(reranked)
                    ]
                    outcome.candidates = _renumber(enriched + tail)
                    outcome.packs.update({pack.candidate_id: pack for pack in valid_packs})
                    outcome.stages.append(RerankStageResult(
                        "vlm", True, len(head), len(valid_head),
                        int((perf_counter() - started) * 1000),
                    ))
            except DependencyUnavailableError as exc:
                outcome.stages.append(RerankStageResult(
                    "vlm", False, len(head), len(head),
                    int((perf_counter() - started) * 1000),
                    warning=f"{exc} — rơi về kết quả của stage text",
                ))
        elif options.vlm.enabled:
            outcome.stages.append(RerankStageResult(
                "vlm", False, len(candidates), len(candidates), 0,
                warning="chưa cấu hình VLM reranker (AIC_RERANK_VLM_URL)",
            ))

        return outcome


def _renumber(candidates: list[Candidate]) -> list[Candidate]:
    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(candidates, start=1)
    ]


__all__ = ["RerankOutcome", "RerankPipeline", "RerankStageResult"]
