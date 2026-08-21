"""Ghép chuỗi TRAKE bằng quy hoạch động (DANTE) — Phase A của docs/31.

Khác `link_event_hits` (beam) ở một điểm quyết định: beam giữ `beam_size` chuỗi
tốt nhất ở MỖI bước rồi mở rộng, nên một chuỗi có bước đầu điểm thấp nhưng các
bước sau rất mạnh có thể bị loại từ sớm. DP không loại gì — nó tính chuỗi tối ưu
cho TỪNG ứng viên kết thúc, nên tối ưu toàn cục theo đúng hàm mục tiêu.

Hàm mục tiêu — GIỐNG HỆT `link_event_hits`, dùng chung `temporal_gap`::

    DP[i][t] = S[i][t] + max   ( DP[i-1][tau] - penalty(gap(tau, t)) )
                        tau dung TRUOC t

    penalty(gap) = min( lambda * max(0, gap - W), cap )

Ràng buộc cứng: cùng video, và `(scene_idx, best_frame_idx)` tăng nghiêm ngặt —
hai bước ĐƯỢC PHÉP nằm trong cùng một scene miễn là frame tiến lên.

**Vì sao O(N·M²) chứ không còn O(N·M).** Bản trước phạt tuyến tính thuần nên tách
được biến (`-lambda*start_t` + `max(DP + lambda*end_tau)`), và một running maximum
quét theo `scene_idx` là đủ. Phạt dead-zone + trần KHÔNG tách được: cùng một
`tau`, phần đóng góp của nó đổi công thức tuỳ `t` rơi vào vùng miễn phạt, vùng
tuyến tính hay vùng chạm trần. Giữ running maximum ở đây là giải một bài toán
khác rồi gọi kết quả là "tối ưu".

Cái giá thật sự nhỏ: M là số ứng viên TRONG MỘT VIDEO ở một bước (hàng chục), và
N <= 6, nên quét thẳng vẫn không đáng kể so với chi phí retrieval. Đổi một hằng
số nhỏ lấy lời giải đúng là đúng hướng ở quy mô này.

Hiệu chuẩn ba tham số phạt: xem `online/services/temporal_gap.py`.
"""

from __future__ import annotations

from online.domain.models import SearchHit, SequenceHit
from online.services.temporal_gap import (
    DEFAULT_FREE_GAP_SEC,
    DEFAULT_GAP_PENALTY_PER_SEC,
    DEFAULT_MAX_GAP_PENALTY,
    gap_penalty_value,
)


def link_event_hits_dp(
    event_hits: list[list[SearchHit]],
    *,
    limit: int = 20,
    gap_penalty: float = DEFAULT_GAP_PENALTY_PER_SEC,
    free_gap_sec: float = DEFAULT_FREE_GAP_SEC,
    max_gap_penalty: float = DEFAULT_MAX_GAP_PENALTY,
) -> list[SequenceHit]:
    """Chuỗi tối ưu toàn cục cho từng ứng viên kết thúc, gộp mọi video.

    Trả về nhiều chuỗi chứ không chỉ chuỗi tốt nhất: mỗi ứng viên của bước CUỐI
    cho một chuỗi (chuỗi tốt nhất kết thúc tại đó). Đủ để `video_recall@3` có
    nghĩa, và không cần beam width — N-best ở đây là hệ quả tự nhiên của DP.
    """

    if len(event_hits) < 2 or any(not hits for hits in event_hits):
        return []

    videos = {hit.video_id for hits in event_hits for hit in hits}
    results: list[SequenceHit] = []
    for video_id in sorted(videos):
        results.extend(
            _solve_one_video(
                [
                    sorted(
                        (hit for hit in hits if hit.video_id == video_id),
                        key=lambda item: (item.scene_idx, item.best_frame_idx, item.scene_id),
                    )
                    for hits in event_hits
                ],
                gap_penalty=gap_penalty,
                free_gap_sec=free_gap_sec,
                max_gap_penalty=max_gap_penalty,
            )
        )
    # Sắp theo điểm, phá hoà bằng scene_id để kết quả TẤT ĐỊNH — cùng quy ước
    # `link_event_hits` dùng, nếu không hai lần chạy cho hai thứ hạng khác nhau.
    results.sort(key=lambda item: (-item.score, [scene.scene_id for scene in item.scenes]))
    return results[:limit]


def _precedes(previous: SearchHit, hit: SearchHit) -> bool:
    """Cùng thứ tự xuất hiện mà `temporal._in_order` áp cho beam."""

    return (previous.scene_idx, previous.best_frame_idx) < (
        hit.scene_idx,
        hit.best_frame_idx,
    )


def _solve_one_video(
    per_event: list[list[SearchHit]],
    *,
    gap_penalty: float,
    free_gap_sec: float,
    max_gap_penalty: float,
) -> list[SequenceHit]:
    """DP trong PHẠM VI một video. Thiếu ứng viên ở bất kỳ bước nào -> bỏ video."""

    if any(not hits for hits in per_event):
        return []

    # `best[j]` = điểm chuỗi tốt nhất kết thúc tại ứng viên j của bước hiện tại;
    # `back[j]` = chỉ số ứng viên bước trước đã chọn.
    best = [hit.score for hit in per_event[0]]
    back: list[list[int | None]] = [[None] * len(per_event[0])]

    for index in range(1, len(per_event)):
        previous, current = per_event[index - 1], per_event[index]
        scores: list[float] = []
        parents: list[int | None] = []
        for hit in current:
            # Quét thẳng mọi ứng viên bước trước. Không dùng running maximum:
            # phạt dead-zone + trần không tách được biến, xem docstring module.
            best_total = float("-inf")
            best_arg: int | None = None
            for position, candidate in enumerate(previous):
                if best[position] == float("-inf"):
                    continue
                if not _precedes(candidate, hit):
                    continue
                total = best[position] - gap_penalty_value(
                    hit.start_sec - candidate.end_sec,
                    penalty_per_sec=gap_penalty,
                    free_gap_sec=free_gap_sec,
                    max_penalty=max_gap_penalty,
                )
                if total > best_total:
                    best_total, best_arg = total, position
            if best_arg is None:
                # Không có ứng viên bước trước nào đứng TRƯỚC hit này.
                scores.append(float("-inf"))
                parents.append(None)
            else:
                scores.append(hit.score + best_total)
                parents.append(best_arg)
        best = scores
        back.append(parents)

    out: list[SequenceHit] = []
    for end_index, total in enumerate(best):
        if total == float("-inf"):
            continue
        chain: list[SearchHit] = []
        cursor: int | None = end_index
        for step in range(len(per_event) - 1, -1, -1):
            if cursor is None:
                break
            chain.append(per_event[step][cursor])
            cursor = back[step][cursor]
        if len(chain) != len(per_event):
            continue
        chain.reverse()
        out.append(SequenceHit(video_id=chain[0].video_id, score=total, scenes=chain))
    return out


__all__ = ["link_event_hits_dp"]
