"""Cửa sổ phát video cho một kết quả tìm kiếm.

Trước đây UI không phát được đoạn video của kết quả, và có ba lý do riêng biệt:

1. **KIS/QA chỉ có thời gian khi `evidence` được đính kèm.** `KisResultItem`
   mang `frame_idx` và `scene_id` nhưng không mang giây nào.
2. **TRAKE và AVS không có giây nào cả.** `AvsResultItem` chỉ có
   `start_frame`/`end_frame`, `TrakeResultItem` chỉ có `frame_ids`.
3. **Không kết quả nào mang đường dẫn video**, nên UI không biết gọi
   `/v1/media/...` với cái gì.

Thêm nữa, đoạn của một scene rất ngắn — đo trên dữ liệu thật, scene p50 chỉ
4.1 giây, và một kết quả KIS trả về cửa sổ 3.2 giây. Xem 3 giây không đủ để
người chấm hiểu chuyện gì đang xảy ra, nên cửa sổ được NỚI ra hai phía.

`focus_sec` giữ nguyên thời điểm của frame được nộp — UI nên nhảy tới đó, còn
phần nới chỉ là bối cảnh trước/sau.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from online.domain.models import SceneDocument
from online.domain.task_results import PlaybackWindow

DEFAULT_PAD_SEC = 5.0


@lru_cache(maxsize=512)
def _media_exists(media_root: Path, video_path: str) -> bool:
    """Có file video thật không. Cache vì mỗi request hỏi lại cùng vài video."""

    return (media_root / video_path).is_file()


def _fps(scene: SceneDocument) -> float:
    """Suy fps từ chính scene; an toàn hơn là giả định 30."""

    frames = scene.end_frame_exclusive - scene.start_frame
    seconds = scene.end_sec - scene.start_sec
    if frames > 0 and seconds > 0:
        return frames / seconds
    return 30.0


def build_window(
    scene: SceneDocument,
    *,
    focus_frame: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    pad_sec: float = DEFAULT_PAD_SEC,
    video_duration_sec: float | None = None,
    media_root: Path | None = None,
) -> PlaybackWindow | None:
    """Cửa sổ phát cho một kết quả, đã nới `pad_sec` mỗi phía.

    `start_frame`/`end_frame` để AVS và TRAKE truyền khoảng riêng của chúng
    (có thể rộng hơn một scene). Không truyền thì lấy biên của scene.

    Trả `None` khi không phát được, và điều đó KHÁC với "phát lỗi": UI phải
    hiện được "chưa có video" thay vì một player 404.
    """

    if not scene.video_path:
        return None
    if media_root is not None and not _media_exists(media_root, scene.video_path):
        # `videos.jsonl` khai `source_path` cho cả ba video nhưng chỉ V001 có
        # file mp4 thật — V002/V003 chỉ được cấp ảnh keyframe. Không kiểm ở đây
        # thì UI nhận một URL 404 và người dùng tưởng player hỏng.
        return None

    fps = _fps(scene)
    if start_frame is not None and end_frame is not None and end_frame > start_frame:
        raw_start = scene.start_sec + (start_frame - scene.start_frame) / fps
        raw_end = scene.start_sec + (end_frame - scene.start_frame) / fps
    else:
        raw_start, raw_end = scene.start_sec, scene.end_sec

    focus = raw_start
    if focus_frame is not None:
        focus = scene.start_sec + (focus_frame - scene.start_frame) / fps
        focus = min(max(focus, 0.0), max(raw_end, raw_start))

    start = max(raw_start - pad_sec, 0.0)
    end = raw_end + pad_sec
    if video_duration_sec:
        end = min(end, video_duration_sec)
    if end <= start:
        end = start + 1.0

    return PlaybackWindow(
        video_id=scene.video_id,
        media_path=scene.video_path,
        start_sec=round(start, 3),
        end_sec=round(end, 3),
        focus_sec=round(max(focus, 0.0), 3),
        pad_sec=pad_sec,
    )


__all__ = ["DEFAULT_PAD_SEC", "build_window"]
