"""Ghép chuỗi TRAKE bằng quy hoạch động (DANTE) — Phase A của docs/31.

Khác `link_event_hits` (beam) ở một điểm quyết định: beam giữ `beam_size` chuỗi
tốt nhất ở MỖI bước rồi mở rộng, nên một chuỗi có bước đầu điểm thấp nhưng các
bước sau rất mạnh có thể bị loại từ sớm. DP không loại gì — nó tính chuỗi tối ưu
cho TỪNG ứng viên kết thúc, nên tối ưu toàn cục theo đúng hàm mục tiêu.

Hàm mục tiêu::

    DP[i][t] = S[i][t] + max   ( DP[i-1][tau] - lambda * gap(tau, t) )
                        tau: scene_idx(tau) < scene_idx(t)

`gap = start_sec(t) - end_sec(tau)`, luôn >= 0 vì scene trong một video liền kề
nhau và ràng buộc bắt `scene_idx` tăng nghiêm ngặt.

**Chạy được O(N·M) nhờ tách biến.** Phạt tuyến tính nên::

    DP[i][t] = S[i][t] - lambda*start_t + max( DP[i-1][tau] + lambda*end_tau )

Vế `max` không phụ thuộc `t`, nên duyệt ứng viên theo `scene_idx` tăng dần và
giữ một running maximum là đủ. Không tách được biến thì độ phức tạp là O(N·M²),
và với M=100 ứng viên mỗi bước thì vẫn chạy được nhưng chậm hơn nhiều.

Hiệu chuẩn `lambda` — ĐO TRƯỚC, không chép từ paper (đơn vị thời gian khác nhau
thì `lambda` của paper vô nghĩa). Trên gold TRAKE của corpus này::

    khoang cach giua hai buoc gold lien tiep:  p10=5s  p50=10s  p90=21s  max=36s
    diem moi scene sau fusion RRF:             ~0.04

`link_event_hits` đang dùng `gap_penalty=0.002`, tức phạt tại p50 là **0.02** —
bằng NỬA điểm của một scene. Nói cách khác ràng buộc hình thức đang lấn át độ
liên quan. `lambda=0` là biến thể bắt buộc phải đo để biết phạt có ích gì không.
"""

from __future__ import annotations

from online.domain.models import SearchHit, SequenceHit


def link_event_hits_dp(
    event_hits: list[list[SearchHit]],
    *,
    limit: int = 20,
    gap_penalty: float = 0.0,
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
                        key=lambda item: (item.scene_idx, item.scene_id),
                    )
                    for hits in event_hits
                ],
                gap_penalty=gap_penalty,
            )
        )
    # Sắp theo điểm, phá hoà bằng scene_id để kết quả TẤT ĐỊNH — cùng quy ước
    # `link_event_hits` dùng, nếu không hai lần chạy cho hai thứ hạng khác nhau.
    results.sort(key=lambda item: (-item.score, [scene.scene_id for scene in item.scenes]))
    return results[:limit]


def _solve_one_video(
    per_event: list[list[SearchHit]], *, gap_penalty: float
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
        # Running maximum trên `DP[i-1][tau] + lambda*end_tau`, quét theo
        # `scene_idx` tăng dần. `pointer` là vị trí đầu tiên của `previous` CHƯA
        # được nạp vào running max.
        pointer = 0
        running_best = float("-inf")
        running_arg: int | None = None
        scores: list[float] = []
        parents: list[int | None] = []
        for hit in current:
            while pointer < len(previous) and previous[pointer].scene_idx < hit.scene_idx:
                value = best[pointer] + gap_penalty * previous[pointer].end_sec
                if value > running_best:
                    running_best, running_arg = value, pointer
                pointer += 1
            if running_arg is None:
                # Không có ứng viên bước trước nào đứng TRƯỚC scene này.
                scores.append(float("-inf"))
                parents.append(None)
            else:
                scores.append(hit.score - gap_penalty * hit.start_sec + running_best)
                parents.append(running_arg)
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
