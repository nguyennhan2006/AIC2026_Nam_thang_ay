from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[Sequence[str], float]],
    rrf_k: int = 60,
    item_multipliers: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ids without trying to calibrate heterogeneous raw scores."""

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    item_multipliers = item_multipliers or {}
    for ids, weight in ranked_lists:
        for rank, item_id in enumerate(ids, start=1):
            multiplier = float(item_multipliers.get(item_id, 1.0))
            scores[item_id] += float(weight) * multiplier / (rrf_k + rank)
            best_rank[item_id] = min(rank, best_rank.get(item_id, rank))
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
