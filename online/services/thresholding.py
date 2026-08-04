"""Ngưỡng cắt theo từng branch, áp TRƯỚC fusion (PR-04).

`min_score`, `threshold_space` và `threshold_policy` tồn tại trong
`BranchRuntimeOptions` từ W0 nhưng chưa từng có consumer — request đặt ngưỡng
vẫn nhận 200 OK còn backend bỏ qua hoàn toàn.

Ngưỡng phải áp trước fusion chứ không phải sau: RRF dùng *hạng* trong danh
sách của branch, nên loại một candidate sau khi fuse không làm các candidate
còn lại lên hạng — cắt sau fusion cho kết quả khác hẳn cắt trước.

Hai policy:

* ``hard`` — loại hẳn candidate, các candidate còn lại được đánh lại hạng và
  do đó thực sự lên hạng trong RRF.
* ``soft`` — giữ candidate nhưng đẩy xuống cuối danh sách của branch đó, nên
  nó chỉ còn đóng góp rất nhỏ thay vì biến mất. Phù hợp với branch hay bỏ sót
  (detector, caption) — xem docs/15_RESEARCH_AGENDA.md.
"""

from __future__ import annotations

from online.domain.candidate import Candidate
from online.domain.models import QueryPlan
from online.services.branch_options import resolve_options


def _score_in_space(candidate: Candidate, space: str) -> float | None:
    if space == "raw":
        return candidate.raw_score
    if space == "percentile":
        return candidate.percentile_score
    return candidate.normalized_score


def apply_threshold(
    candidates: list[Candidate], plan: QueryPlan
) -> tuple[list[Candidate], int]:
    """Áp ngưỡng cho danh sách của ĐÚNG MỘT branch; trả (kết quả, số bị cắt)."""

    if not candidates:
        return candidates, 0
    source = candidates[0].source
    options = resolve_options(plan, source)
    if options is None or options.min_score is None:
        return candidates, 0

    kept: list[Candidate] = []
    demoted: list[Candidate] = []
    for candidate in candidates:
        score = _score_in_space(candidate, options.threshold_space)
        if score is None:
            # Chưa chuẩn hóa mà lại đòi ngưỡng ở không gian chuẩn hóa: giữ
            # candidate thay vì cắt mù, để không âm thầm bỏ mất kết quả.
            kept.append(candidate)
            continue
        if score >= options.min_score:
            kept.append(candidate)
        elif options.threshold_policy == "soft":
            demoted.append(candidate)

    removed = len(candidates) - len(kept) - len(demoted)
    ordered = kept + demoted
    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(ordered, start=1)
    ], removed + len(demoted)


def apply_thresholds(
    ranked_lists: list[list[Candidate]], plan: QueryPlan
) -> tuple[list[list[Candidate]], dict[str, int]]:
    """Áp ngưỡng cho mọi branch; trả thêm số candidate bị ảnh hưởng mỗi branch."""

    output: list[list[Candidate]] = []
    affected: dict[str, int] = {}
    for candidates in ranked_lists:
        filtered, count = apply_threshold(candidates, plan)
        output.append(filtered)
        if count and candidates:
            affected[candidates[0].source] = count
    return output, affected


__all__ = ["apply_threshold", "apply_thresholds"]
