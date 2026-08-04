"""Gắn vùng chiến thuật (1 / 2–5 / 6–20 / 21–50 / 51–100) vào submission (PR-08).

Chỉ chú thích — KHÔNG sắp lại thứ tự. Thứ tự thật do task processor quyết
định (KIS signature, joint QA score, TRAKE sequence score, AVS MMR); module
này chỉ giúp Submission Board (PR-10) tô màu đúng vùng để người dùng biết nên
ưu tiên chỉnh tay ở đâu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from online.competition.rules import zone_for_rank

Item = TypeVar("Item")


@dataclass(frozen=True, slots=True)
class ZonedItem(Generic[Item]):
    rank: int
    zone: str
    item: Item


def annotate_zones(items: list[Item], *, rank_of=lambda item: item.rank) -> list[ZonedItem[Item]]:
    """Gắn `zone_for_rank` cho từng item theo `rank_of(item)`."""

    return [ZonedItem(rank=rank_of(item), zone=zone_for_rank(rank_of(item)), item=item) for item in items]


def zone_summary(items: list[Item], *, rank_of=lambda item: item.rank) -> dict[str, int]:
    """Đếm số item mỗi vùng — dùng để hiện tổng quan Submission Board."""

    counts: dict[str, int] = {}
    for item in items:
        zone = zone_for_rank(rank_of(item))
        counts[zone] = counts.get(zone, 0) + 1
    return counts


__all__ = ["ZonedItem", "annotate_zones", "zone_summary"]
