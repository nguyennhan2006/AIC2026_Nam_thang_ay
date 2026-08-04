"""Đưa điểm của mọi branch về cùng thang [0, 1] (PR-04).

BM25 (không chặn trên), cosine ([-1, 1]), fuzzy ratio ([0, 1]) và tỉ lệ
overlap màu không so sánh được với nhau. Trước module này `min_score` trong
`BranchRuntimeOptions` không có consumer nào — API nhận rồi bỏ qua, nên UI
chỉnh ngưỡng mà backend không đổi gì.

Chuẩn hóa được tính TRONG PHẠM VI một danh sách kết quả của một branch:
`normalized_score` là vị trí tương đối trong chính danh sách đó, không phải
xác suất tuyệt đối. Calibration tuyệt đối (cần dev set) là việc sau, và khi
có thì chỉ thay `calibration_version` chứ không đổi contract.
"""

from __future__ import annotations

import math

from online.domain.candidate import Candidate
from online.domain.scores import ScoreKind

NormalizationMethod = str
CALIBRATION_VERSION = "listwise_v1"

# Score đã nằm sẵn trong [0, 1] thì giữ nguyên — ép min-max lên chúng sẽ phá
# ý nghĩa tuyệt đối (vd fuzzy 0.9 và 0.91 bị kéo thành 0.0 và 1.0).
_IDENTITY_KINDS: frozenset[ScoreKind] = frozenset(
    {"fuzzy_ratio", "confidence", "histogram_similarity", "overlap_ratio"}
)


def _percentiles(values: list[float]) -> list[float]:
    """Hạng phân vị của từng phần tử; giá trị bằng nhau nhận cùng phân vị."""

    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        # Trung bình hạng của nhóm bằng nhau -> không thiên vị thứ tự đầu vào.
        rank = (position + end) / 2
        share = rank / (len(order) - 1)
        for index in range(position, end + 1):
            result[order[index]] = share
        position = end + 1
    return result


def normalize_branch(
    candidates: list[Candidate], *, method: str = "percentile"
) -> list[Candidate]:
    """Điền `normalized_score` và `percentile_score` cho một danh sách branch."""

    if not candidates:
        return candidates
    raw = [item.raw_score for item in candidates]
    percentile = _percentiles(raw)
    kind = candidates[0].score_kind

    if kind in _IDENTITY_KINDS:
        normalized = [min(max(value, 0.0), 1.0) for value in raw]
        used = "identity"
    elif kind == "cosine":
        # Cosine có miền xác định [-1, 1] nên map affine đúng về [0, 1] mà
        # không phụ thuộc vào các candidate khác trong danh sách.
        normalized = [min(max((value + 1.0) / 2.0, 0.0), 1.0) for value in raw]
        used = "affine"
    elif method == "percentile":
        normalized = list(percentile)
        used = "percentile"
    elif method == "minmax":
        low, high = min(raw), max(raw)
        span = high - low
        normalized = [1.0] * len(raw) if span <= 0 else [(value - low) / span for value in raw]
        used = "minmax"
    else:  # "calibrated" — chưa có dev set để fit; sigmoid là baseline trung tính.
        normalized = [1.0 / (1.0 + math.exp(-value)) for value in raw]
        used = "sigmoid"

    return [
        item.model_copy(
            update={
                "normalized_score": normalized[index],
                "percentile_score": percentile[index],
            }
        )
        for index, item in enumerate(candidates)
    ]


def normalize_all(
    ranked_lists: list[list[Candidate]], *, method: str = "percentile"
) -> list[list[Candidate]]:
    return [normalize_branch(items, method=method) for items in ranked_lists]


__all__ = ["CALIBRATION_VERSION", "normalize_all", "normalize_branch"]
