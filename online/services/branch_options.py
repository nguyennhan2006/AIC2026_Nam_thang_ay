"""Phân giải cấu hình runtime cho một branch (Search Mixing Console W5 + PR-03).

`QueryPlan.modality_weights` chỉ có 9 bucket dùng chung cho mọi retriever
trong bucket đó — `bm25_ocr` và `ocr_fuzzy` không chỉnh trọng số độc lập được.
Module này thêm lớp override theo từng branch, đọc từ
`QueryPlan.search_options.branches[...]`.

Thứ tự tra cứu (PR-03): `execution_id` trước, rồi `branch_id`. Nhờ vậy có thể
chỉnh riêng `caption_bm25.expanded` mà không đụng `caption_bm25.raw`, nhưng
cấu hình đặt ở mức `caption_bm25` vẫn áp cho mọi execution của nó. Không có
entry nào -> hành vi y như trước khi có module này.
"""

from __future__ import annotations

from online.domain.candidate import Modality
from online.domain.models import QueryPlan
from online.domain.search_config import BranchRuntimeOptions


def resolve_options(
    plan: QueryPlan, execution_id: str, branch_id: str | None = None
) -> BranchRuntimeOptions | None:
    """Trả override cụ thể nhất cho một execution, hoặc None nếu không có."""

    branches = plan.search_options.branches
    override = branches.get(execution_id)
    if override is not None:
        return override
    if branch_id is None and "." in execution_id:
        branch_id = execution_id.rsplit(".", 1)[0]
    return branches.get(branch_id) if branch_id else None


def effective_weight(
    plan: QueryPlan, execution_id: str, modality: Modality, branch_id: str | None = None
) -> float:
    """Trọng số branch nên dùng, hoặc 0.0 nếu nó không được phép chạy."""

    override = resolve_options(plan, execution_id, branch_id)
    if override is not None:
        return override.weight if override.enabled else 0.0
    return plan.modality_weights.get(modality, 0.0)


def effective_limit(
    plan: QueryPlan, execution_id: str, default_limit: int, branch_id: str | None = None
) -> int:
    """Số candidate branch nên xin từ index của nó."""

    override = resolve_options(plan, execution_id, branch_id)
    if override is not None and override.enabled:
        return min(default_limit, override.top_k)
    return default_limit


__all__ = ["effective_limit", "effective_weight", "resolve_options"]
