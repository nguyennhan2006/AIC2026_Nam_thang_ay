"""Hằng số luật thi — một nơi duy nhất, không rải rác trong code (PR-08).

Đổi luật (vd BTC nâng giới hạn 100 lên 150) chỉ cần sửa file này.
"""

from __future__ import annotations

MAX_SUBMISSION_ITEMS = 100

# Các mốc chấm điểm chính thức của vòng sơ tuyển.
SCORING_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)

# Vùng chiến thuật khi xếp hạng submission (docs/09_RESEARCH_ALIGNMENT.md,
# 01082026_new_docs.md §17). Dùng để tô màu/ giải thích submission board,
# không ảnh hưởng thứ tự thật — thứ tự do processor quyết định.
RANKING_ZONES: tuple[tuple[str, int, int], ...] = (
    ("rank_1", 1, 1),
    ("ranks_2_5", 2, 5),
    ("ranks_6_20", 6, 20),
    ("ranks_21_50", 21, 50),
    ("ranks_51_100", 51, 100),
)


def zone_for_rank(rank: int) -> str:
    for name, low, high in RANKING_ZONES:
        if low <= rank <= high:
            return name
    return "beyond_100"


__all__ = ["MAX_SUBMISSION_ITEMS", "RANKING_ZONES", "SCORING_CUTOFFS", "zone_for_rank"]
