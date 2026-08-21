"""Temporal linking of independently retrieved query events."""

from __future__ import annotations

from online.domain.models import SearchHit, SequenceHit
from online.services.temporal_gap import (
    DEFAULT_FREE_GAP_SEC,
    DEFAULT_GAP_PENALTY_PER_SEC,
    DEFAULT_MAX_GAP_PENALTY,
    gap_penalty_value,
)


def _in_order(previous: SearchHit, hit: SearchHit) -> bool:
    """Thứ tự XUẤT HIỆN tăng nghiêm ngặt, không bắt phải khác scene.

    Bản trước đòi `scene_idx` tăng, tức vứt mọi chuỗi có hai bước nằm trong cùng
    một scene — nhưng một scene dài 10 giây chứa thừa chỗ cho nhiều khoảnh khắc,
    và người ra đề không cắt scene theo bước của họ. So bằng cặp
    `(scene_idx, best_frame_idx)` giữ nguyên tính tăng nghiêm ngặt (nên frame nộp
    lên vẫn phân biệt được từng bước) mà không còn ràng buộc thừa đó.
    """

    return (hit.scene_idx, hit.best_frame_idx) > (
        previous.scene_idx,
        previous.best_frame_idx,
    )


def _quota(ranked: list, cap: int, per_video: int | None) -> list:
    """Cắt còn `cap` phần tử, nhưng mỗi video được nhiều nhất `per_video` chỗ.

    `ranked` phải đã sắp giảm dần theo điểm. `per_video=None` giữ nguyên hành vi
    cũ (cắt thuần theo điểm).

    Vì sao cần: đo trên 873 video, 20 dòng output chỉ chứa **1.58 video khác
    nhau** — một video "nam châm" sinh hàng trăm chuỗi gần trùng nhau và chiếm
    sạch beam. Với người dùng tự xem video để chốt, đó là danh sách vô dụng:
    quét sâu hơn không lộ ra video nào mới (video_recall@1 0.750 so với
    video_recall@20 0.792 — gần như bằng nhau). Hạn ngạch đổi các bản sao gần
    trùng lấy độ phủ, thứ mà người xem thật sự dùng được.
    """

    if per_video is None:
        return ranked[:cap]
    kept: list = []
    used: dict[str, int] = {}
    for item in ranked:
        if not item[0]:
            continue  # chuỗi rỗng chưa gắn với video nào
        video_id = item[0][0].video_id
        if used.get(video_id, 0) >= per_video:
            continue
        used[video_id] = used.get(video_id, 0) + 1
        kept.append(item)
        if len(kept) >= cap:
            break
    return kept


def link_event_hits(
    event_hits: list[list[SearchHit]],
    *,
    limit: int = 20,
    beam_size: int = 100,
    per_video_beam: int | None = None,
    max_chains_per_video: int | None = None,
    allow_missing_steps: bool = False,
    min_covered_steps: int = 2,
    missing_step_penalty: float = 0.01,
    gap_penalty: float = DEFAULT_GAP_PENALTY_PER_SEC,
    free_gap_sec: float = DEFAULT_FREE_GAP_SEC,
    max_gap_penalty: float = DEFAULT_MAX_GAP_PENALTY,
) -> list[SequenceHit]:
    """Beam-link hits trong cùng một video theo thứ tự xuất hiện tăng dần.

    Ràng buộc CỨNG chỉ có hai: cùng `video_id`, và `(scene_idx, best_frame_idx)`
    tăng nghiêm ngặt. Thời gian là tín hiệu MỀM — phạt dead-zone + trần, xem
    `online/services/temporal_gap.py` để biết vì sao ba tham số mặc định là như
    vậy. Đặt `gap_penalty=0.0` để tắt hẳn phần thời gian.

    `per_video_beam` giới hạn số chuỗi dở dang mà MỘT video được giữ trong beam;
    `max_chains_per_video` làm điều tương tự ở đầu ra. Cả hai `None` = hành vi
    cũ. Phải áp ở CẢ HAI chỗ: chỉ khử trùng lặp lúc xuất thì vô ích, vì lúc đó
    các video khác đã bị đẩy khỏi beam từ mấy bước trước rồi.

    `allow_missing_steps` cho phép chuỗi **bỏ qua** step không có candidate hợp
    lệ trong video, chịu `missing_step_penalty` mỗi lần bỏ, miễn còn đủ
    `min_covered_steps` step thật. Đây không phải một nới lỏng cho đẹp: đo trên
    873 video, chỉ 14/24 truy vấn có đủ candidate ở MỌI step, và trong số truy
    vấn hỏng có ca video đúng nằm sẵn trong pool ở 4/5 step nhưng bị vứt sạch
    chỉ vì thiếu step còn lại. Với `False` (mặc định) hành vi giữ y như cũ.

    `missing_step_penalty` phải NHỎ so với điểm một scene (~0.04 sau RRF): mục
    tiêu là xếp chuỗi đầy đủ lên trước chuỗi thủng khi cả hai cùng tồn tại,
    KHÔNG phải để dìm chuỗi thủng xuống dưới nhiễu của video khác. Phạt quá tay
    thì tính năng này thành vô dụng đúng theo cách khó thấy nhất.
    """

    total_steps = len(event_hits)
    if total_steps < 2:
        return []
    if not allow_missing_steps and any(not hits for hits in event_hits):
        return []

    # Mỗi beam: (hit đã chọn, số thứ tự step 1-based của chúng, điểm).
    beams: list[tuple[list[SearchHit], list[int], float]] = [
        ([hit], [1], hit.score) for hit in event_hits[0]
    ]
    if allow_missing_steps:
        # Nhánh "chưa chọn gì": cần thiết để step 1 cũng được phép thiếu.
        beams.append(([], [], -missing_step_penalty))
    if not beams:
        return []

    for index, hits in enumerate(event_hits[1:], start=2):
        expanded: list[tuple[list[SearchHit], list[int], float]] = []
        for sequence, steps, score in beams:
            previous = sequence[-1] if sequence else None
            for hit in hits:
                if previous is not None:
                    if hit.video_id != previous.video_id or not _in_order(previous, hit):
                        continue
                    gap = hit.start_sec - previous.end_sec
                    penalty = gap_penalty_value(
                        gap,
                        penalty_per_sec=gap_penalty,
                        free_gap_sec=free_gap_sec,
                        max_penalty=max_gap_penalty,
                    )
                else:
                    penalty = 0.0
                expanded.append(
                    (sequence + [hit], steps + [index], score + hit.score - penalty)
                )
            if allow_missing_steps:
                expanded.append((sequence, steps, score - missing_step_penalty))
        expanded.sort(
            key=lambda item: (
                -item[2],
                [scene.scene_id for scene in item[0]],
                item[1],
            )
        )
        beams = _quota(expanded, beam_size, per_video_beam)
        if not beams:
            return []

    survivors = [item for item in beams if len(item[0]) >= max(1, min_covered_steps)]
    return [
        SequenceHit(
            video_id=scenes[0].video_id,
            score=score,
            scenes=scenes,
            covered_steps=steps,
            total_steps=total_steps,
        )
        for scenes, steps, score in _quota(survivors, limit, max_chains_per_video)
    ]
