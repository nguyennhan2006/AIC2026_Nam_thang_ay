"""Per-branch runtime override resolution — Search Mixing Console W5.

`weighted_rrf` and every retriever currently gate purely on
`QueryPlan.modality_weights` (5 fixed buckets shared by every retriever in
that bucket — e.g. `bm25_ocr` and `ocr_fuzzy` cannot be weighted
independently even though they are different branches). This module adds an
optional per-branch override layer read from `QueryPlan.search_options.branches
[branch_name]`, without changing the modality-weight behavior when no
override is configured (the default: `SearchOptions().branches == {}`).

Precedence: an explicit `BranchRuntimeOptions` entry for a branch always wins
over its modality's bucket weight — `enabled=False` disables it outright
regardless of modality weight, and `weight`/`top_k` replace the modality
default. With no entry for that branch name, behavior is unchanged from
before this module existed.
"""

from __future__ import annotations

from online.domain.models import Modality, QueryPlan


def effective_weight(plan: QueryPlan, name: str, modality: Modality) -> float:
    """Return the weight a branch should use, or 0.0 if it must not run."""

    override = plan.search_options.branches.get(name)
    if override is not None:
        return override.weight if override.enabled else 0.0
    return plan.modality_weights.get(modality, 0.0)


def effective_limit(plan: QueryPlan, name: str, default_limit: int) -> int:
    """Return the candidate limit a branch should request from its index."""

    override = plan.search_options.branches.get(name)
    if override is not None and override.enabled:
        return min(default_limit, override.top_k)
    return default_limit


__all__ = ["effective_weight", "effective_limit"]
