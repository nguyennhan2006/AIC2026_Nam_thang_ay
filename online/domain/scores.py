"""Per-branch score provenance (PR-01 contract, PR-05 fills it in).

BM25, cosine, fuzzy-ratio và color-overlap không cùng thang đo, nên một
`float` trần không đủ để fuse hay threshold đúng. `BranchScore` gắn kèm
*score sinh ra ở không gian nào* và *đã chuẩn hóa bằng cách gì*.

PR-01 chỉ định nghĩa contract và cho mỗi retriever khai báo `score_kind`
của mình; `normalized_score`/`percentile_score` còn None cho tới khi
`online/services/score_normalization.py` (PR-05) điền.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from online.domain.base import StrictModel

ScoreKind = Literal[
    "cosine",
    "inner_product",
    "bm25",
    "fuzzy_ratio",
    "confidence",
    "histogram_similarity",
    "overlap_ratio",
    "reranker",
    "fusion",
]


class BranchScore(StrictModel):
    """Điểm của một branch cho một candidate, kèm đủ provenance để tái lập."""

    raw_score: float
    score_kind: ScoreKind
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    percentile_score: float | None = Field(default=None, ge=0.0, le=1.0)
    normalization_method: Literal["none", "minmax", "percentile", "affine", "sigmoid", "identity"] = "none"
    calibration_version: str | None = None


__all__ = ["BranchScore", "ScoreKind"]
