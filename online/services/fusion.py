"""Fusion methods across incomparable retrieval scores — Search Mixing Console W5.

Every method reuses the same rank-derived, scale-free per-branch contribution
`weight / (rrf_k + rank)` — never a raw score across branches (BM25 vs.
cosine vs. color-overlap-ratio are not comparable, see the plan's "không dùng
raw score giữa các branch" rule). `weighted_sum`/`max_score` differ from
`rrf` only in how they combine that same contribution across branches
(sum vs. max instead of RRF's harmonic-ish sum); they are NOT a properly
score-normalized weighted sum (that needs per-branch min-max/percentile
calibration — see docs/15_RESEARCH_AGENDA.md, not implemented yet).
`intersection`/`union` differ in which candidates survive: intersection
keeps only candidates seen by at least `minimum_matching_branches` branches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from online.domain.models import Candidate, Modality
from online.domain.search_config import BranchRuntimeOptions

FusionMethod = Literal["rrf", "weighted_sum", "max_score", "intersection", "union"]


def _branch_weight(
    candidate: Candidate, modality_weights: dict[Modality, float], branches: dict[str, BranchRuntimeOptions],
) -> float:
    # `candidate.source` là execution_id (`bm25_caption.expanded`). Tra theo
    # execution trước, rồi tới branch — cấu hình đặt ở mức branch vẫn áp cho
    # mọi biến thể query của nó.
    override = branches.get(candidate.source)
    if override is None and "." in candidate.source:
        override = branches.get(candidate.source.rsplit(".", 1)[0])
    if override is not None:
        return override.weight if override.enabled else 0.0
    return modality_weights.get(candidate.modality, 0.0)


def fuse_candidates(
    ranked_lists: list[list[Candidate]],
    modality_weights: dict[Modality, float],
    *,
    method: FusionMethod = "rrf",
    rrf_k: int = 60,
    limit: int = 100,
    branches: dict[str, BranchRuntimeOptions] | None = None,
    minimum_matching_branches: int = 1,
) -> list[Candidate]:
    """Fuse `ranked_lists`, retain component scores, return deterministic ordering."""

    branches = branches or {}
    totals: defaultdict[str, float] = defaultdict(float)
    representatives: dict[str, Candidate] = {}
    components: defaultdict[str, dict[str, float]] = defaultdict(dict)
    contributions: defaultdict[str, dict[str, float]] = defaultdict(dict)
    modalities: defaultdict[str, set[str]] = defaultdict(set)
    evidence: defaultdict[str, list[dict]] = defaultdict(list)
    matching_branches: defaultdict[str, set[str]] = defaultdict(set)
    frame_hints: dict[str, Candidate] = {}

    for candidates in ranked_lists:
        for fallback_rank, candidate in enumerate(candidates, start=1):
            rank = candidate.rank or fallback_rank
            weight = _branch_weight(candidate, modality_weights, branches)
            contribution = weight / (rrf_k + rank)
            key = candidate.grouping_key
            if method == "max_score":
                totals[key] = max(totals[key], contribution)
            else:
                totals[key] += contribution
            matching_branches[key].add(candidate.source)
            representatives.setdefault(key, candidate)
            components[key][candidate.source] = candidate.raw_score
            # Đóng góp thực tế vào điểm fusion — cùng thang đo giữa mọi branch,
            # khác với component_scores (raw, không so sánh được với nhau).
            contributions[key][candidate.source] = (
                contributions[key].get(candidate.source, 0.0) + contribution
            )
            modalities[key].add(candidate.modality.value)
            # Candidate neo frame mang tọa độ submission chính xác hơn candidate
            # neo scene; giữ lại cái có contribution cao nhất để tầng hydrate ưu
            # tiên frame do retrieval chỉ ra thay vì tự đoán lại.
            if candidate.frame_idx is not None:
                incumbent = frame_hints.get(key)
                if incumbent is None or contribution > contributions[key].get(
                    incumbent.source, 0.0
                ):
                    frame_hints[key] = candidate
            matched_text = candidate.payload.get("matched_text")
            if matched_text:
                evidence[key].append(
                    {
                        "modality": candidate.modality.value,
                        "text": matched_text,
                        "score": candidate.raw_score,
                    }
                )

    eligible = totals.keys()
    if method == "intersection":
        threshold = max(2, minimum_matching_branches)
        eligible = [key for key in totals if len(matching_branches[key]) >= threshold]

    ordered = sorted(eligible, key=lambda item: (-totals[item], item))[:limit]
    output: list[Candidate] = []
    for rank, key in enumerate(ordered, start=1):
        base = representatives[key]
        hint = frame_hints.get(key)
        payload = dict(base.payload)
        payload.update(
            {
                "component_scores": components[key],
                "branch_contributions": contributions[key],
                "matched_modalities": sorted(modalities[key]),
                "evidence": evidence[key],
                "matched_branches": sorted(matching_branches[key]),
            }
        )
        output.append(
            Candidate(
                candidate_id=key,
                entity_type=base.entity_type,
                scene_id=base.scene_id,
                clip_id=base.clip_id,
                event_id=base.event_id or (hint.event_id if hint else None),
                video_id=base.video_id,
                frame_idx=hint.frame_idx if hint else base.frame_idx,
                timestamp_sec=hint.timestamp_sec if hint else base.timestamp_sec,
                start_frame=base.start_frame,
                end_frame=base.end_frame,
                source=f"fusion_{method}",
                modality=base.modality,
                raw_score=totals[key],
                score_kind="fusion",
                rank=rank,
                payload=payload,
            )
        )
    return output


def weighted_rrf(
    ranked_lists: list[list[Candidate]],
    modality_weights: dict[Modality, float],
    *,
    rrf_k: int = 60,
    limit: int = 100,
    branches: dict[str, BranchRuntimeOptions] | None = None,
) -> list[Candidate]:
    """Backward-compatible alias for `fuse_candidates(..., method="rrf")`."""

    return fuse_candidates(
        ranked_lists, modality_weights, method="rrf", rrf_k=rrf_k, limit=limit, branches=branches,
    )


__all__ = ["FusionMethod", "fuse_candidates", "weighted_rrf"]
