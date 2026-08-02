"""TRAKE Stage C — tinh chỉnh frame trong cửa sổ hẹp (PR-07).

Cửa sổ GT của một semantic keyframe chỉ rộng **9 frame (±4)**, trong khi
keyframe được trích thưa (thường 1 frame/scene). Nộp thẳng keyframe gần như
chắc chắn trượt cửa sổ, dù đã tìm đúng video và đúng scene.

Module này chấm lại từng frame ứng viên trong một cửa sổ quanh keyframe:

* Có index frame (`aic_frames_v2`) -> chấm mọi keyframe rơi vào cửa sổ.
* Không có -> chỉ còn keyframe của scene, và kết quả được đánh dấu
  `refinement="keyframe_only"` để không ai nhầm là đã tinh chỉnh dày đặc.

Cố tình KHÔNG tự decode video ở tầng online: giải mã trong request path là
đường ngắn nhất tới timeout. Decode dày đặc là việc của offline (notebook
keyframe với stride nhỏ quanh vùng quan tâm) hoặc của một endpoint riêng.
"""

from __future__ import annotations

from dataclasses import dataclass

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument
from online.domain.task_results import TrakeStep
from online.services.safe_frame import SafeFrameConfig, score_frames


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    # Nửa cửa sổ tìm kiếm quanh frame ban đầu, tính theo frame.
    window_frames: int = 45
    safe_frame: SafeFrameConfig = SafeFrameConfig()


def refine_step(
    step: int,
    step_query: str,
    scene: SceneDocument,
    anchor_frame_idx: int,
    *,
    extra_frames: list[FrameEvidence] | None = None,
    config: RefinementConfig | None = None,
) -> TrakeStep:
    """Chọn frame tốt nhất trong cửa sổ quanh `anchor_frame_idx`.

    `extra_frames` là frame lấy từ index mức frame (nếu có). Không truyền thì
    chỉ dùng keyframe của scene và kết quả bị đánh dấu là chưa tinh chỉnh dày.
    """

    config = config or RefinementConfig()
    window = config.window_frames
    pool = list(scene.keyframes)
    dense = False
    if extra_frames:
        known = {frame.frame_idx for frame in pool}
        for frame in extra_frames:
            if frame.frame_idx not in known:
                pool.append(frame)
                known.add(frame.frame_idx)
        dense = True

    in_window = [
        frame for frame in pool if abs(frame.frame_idx - anchor_frame_idx) <= window
    ]
    if not in_window:
        # Cửa sổ rỗng nghĩa là anchor nằm ngoài scene này; giữ nguyên anchor
        # thay vì kéo về một frame xa lắc chỉ vì nó tồn tại.
        return TrakeStep(
            step=step, frame_idx=anchor_frame_idx, scene_id=scene.scene_id,
            confidence=0.2, refinement="dense_window" if dense else "keyframe_only",
        )

    windowed = scene.model_copy(update={"keyframes": in_window})
    scored = score_frames(windowed, step_query, config.safe_frame)
    best = scored[0]
    # Chuẩn hóa về [0, 1]: điểm safe-frame có thể âm khi bị phạt nặng.
    confidence = min(max(best.total, 0.0), 1.0)
    return TrakeStep(
        step=step,
        frame_idx=best.frame.frame_idx,
        scene_id=scene.scene_id,
        confidence=confidence,
        refinement="dense_window" if dense else "keyframe_only",
    )


__all__ = ["RefinementConfig", "refine_step"]
