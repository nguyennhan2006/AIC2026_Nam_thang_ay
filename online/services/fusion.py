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
    override = branches.get(candidate.source)
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
    modalities: defaultdict[str, set[str]] = defaultdict(set)
    evidence: defaultdict[str, list[dict]] = defaultdict(list)
    matching_branches: defaultdict[str, set[str]] = defaultdict(set)

    for candidates in ranked_lists:
        for fallback_rank, candidate in enumerate(candidates, start=1):
            rank = candidate.rank or fallback_rank
            weight = _branch_weight(candidate, modality_weights, branches)
            contribution = weight / (rrf_k + rank)
            scene_id = candidate.scene_id
            if method == "max_score":
                totals[scene_id] = max(totals[scene_id], contribution)
            else:
                totals[scene_id] += contribution
            matching_branches[scene_id].add(candidate.source)
            representatives.setdefault(scene_id, candidate)
            components[scene_id][candidate.source] = candidate.score
            modalities[scene_id].add(candidate.modality.value)
            matched_text = candidate.payload.get("matched_text")
            if matched_text:
                evidence[scene_id].append(
                    {"modality": candidate.modality.value, "text": matched_text, "score": candidate.score}
                )

    eligible = totals.keys()
    if method == "intersection":
        threshold = max(2, minimum_matching_branches)
        eligible = [scene_id for scene_id in totals if len(matching_branches[scene_id]) >= threshold]

    ordered = sorted(eligible, key=lambda item: (-totals[item], item))[:limit]
    output: list[Candidate] = []
    for rank, scene_id in enumerate(ordered, start=1):
        base = representatives[scene_id]
        payload = dict(base.payload)
        payload.update(
            {
                "component_scores": components[scene_id],
                "matched_modalities": sorted(modalities[scene_id]),
                "evidence": evidence[scene_id],
                "matched_branches": sorted(matching_branches[scene_id]),
            }
        )
        output.append(
            Candidate(
                entity_id=scene_id,
                scene_id=scene_id,
                video_id=base.video_id,
                source=f"fusion_{method}",
                modality=base.modality,
                score=totals[scene_id],
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
