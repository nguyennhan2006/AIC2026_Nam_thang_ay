"""Branch-specific query selection for Structured KIS."""

from __future__ import annotations

from online.domain.models import Modality, QueryPlan


def get_branch_query(
    plan: QueryPlan,
    branch_id: str,
    modality: Modality,
    fallback_query: str,
    *,
    execution_id: str | None = None,
) -> str | None:
    """Return the query a retriever should run.

    Structured KIS populates `branch_queries`/`modality_queries` explicitly. If a
    key exists with an empty string, the caller should skip that branch. Without
    those fields, every retriever receives the same fallback as before.
    """

    keys = [branch_id]
    if execution_id:
        keys.insert(0, execution_id)
    for key in keys:
        if key in plan.branch_queries:
            value = plan.branch_queries[key].strip()
            return value or None
    if modality in plan.modality_queries:
        value = plan.modality_queries[modality].strip()
        return value or None
    return fallback_query.strip() or None


__all__ = ["get_branch_query"]
