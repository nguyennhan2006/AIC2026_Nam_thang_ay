"""Hằng số chuẩn hoá điểm, tính MỘT LẦN trên tập candidate trước dedup.

Vì sao cần file này (EVAL-01, tài liệu Experiment Validation §3.3):

Cả bốn task processor đều chuẩn hoá điểm theo `max(...)` của **chính danh sách
được truyền vào**::

    best_score     = max(hit.score for hit in hits)          # kis.py
    branch_ceiling = max(len(hit.matched_branches) ...)      # kis.py
    best_frame_score = max(frame_scores.values())            # qa.py
    best_score     = max(scores.values())                    # avs.py

Danh sách đó nằm SAU `deduplicate_for_task`, nên nó phụ thuộc
`fusion.max_results_per_video`. Hệ quả đã đo được: nới cap 5 -> 20 làm đổi cả
thứ hạng đã có (KIS 12/12 query lệch prefix, có query lệch ngay vị trí thứ 2).
Đó là vi phạm bất biến "nới output cap không được đổi prefix" — và nó khiến
mọi phép so sánh metric giữa hai cấu hình cap trở nên vô nghĩa.

Cách sửa: mẫu số phải là đại lượng của **truy vấn**, không phải của lát cắt
đang hiển thị. `SearchService` tính nó ngay sau fusion (trước dedup) rồi
truyền xuống processor.
"""

from __future__ import annotations

from dataclasses import dataclass

from online.domain.models import Candidate


@dataclass(frozen=True, slots=True)
class ScoreNormalizers:
    """Mẫu số dùng chung cho mọi task processor của MỘT lần search."""

    # Điểm fusion cao nhất trong toàn bộ pool trước dedup.
    best_retrieval_score: float = 1.0
    # Số nhánh tối đa cùng thấy một candidate trong toàn bộ pool trước dedup.
    branch_ceiling: int = 1

    @classmethod
    def from_pool(cls, candidates: list[Candidate]) -> "ScoreNormalizers":
        """Tính trên pool ĐÃ fuse nhưng CHƯA dedup.

        `raw_score` sau fusion chính là điểm RRF tổng; `branch_scores` giữ đủ
        các nhánh đã cùng thấy candidate đó.
        """

        best = max((candidate.raw_score for candidate in candidates), default=0.0) or 1.0
        ceiling = max((len(candidate.branch_scores) for candidate in candidates), default=1) or 1
        return cls(best_retrieval_score=best, branch_ceiling=ceiling)


__all__ = ["ScoreNormalizers"]
