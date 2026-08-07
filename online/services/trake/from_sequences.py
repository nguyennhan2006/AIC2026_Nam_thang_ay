"""Dựng `TrakeResultItem` từ `link_event_hits` — đường CŨ, đo ra tốt hơn.

`TrakeProcessor` (PR-07) được viết để THAY `link_event_hits`, với lý do đúng:
sai video là 0 điểm nên phải khoá video trước bằng bằng chứng gộp của mọi
step. Nhưng đo trên 24 truy vấn TRAKE gold thì nó kém hơn chính thứ nó thay:

                        TrakeProcessor   link_event_hits
    video_recall@1              0.542            0.833
    video_recall@3              0.833            1.000
    gold_video_missing          0.167            0.000
    mean R-score                0.183            0.254

Và không phải khớp riêng một video — trên hai video holdout, đường cũ gấp đôi
tỉ lệ chọn đúng video:

    V001   1.000 -> 1.000     mean R 0.287 -> 0.281
    V002   0.375 -> 0.750     mean R 0.179 -> 0.306
    V003   0.250 -> 0.750     mean R 0.081 -> 0.175

Theo từng truy vấn: đường cũ tốt hơn ở 9, kém hơn ở 4, hoà 11.

Sai lầm này ẩn được lâu vì `scripts/eval_tasks.py` CHỈ chấm `response.trake`.
Đường cũ vẫn chạy và vẫn được trả về trong `response.sequences`, nhưng chưa
bao giờ có ai chấm nó. Nay `--trake-source` chấm được cả hai.

Module này chỉ ĐỔI HÌNH DẠNG, không tính lại điểm: submission builder, UI và
`_attach_playback` đều đã nói chuyện với `TrakeResultItem`, nên giữ nguyên
contract là giữ nguyên mọi thứ phía sau.
"""

from __future__ import annotations

from online.domain.models import SequenceHit
from online.domain.task_results import TrakeResultItem, TrakeStep


def _confidence(score: float, best: float) -> float:
    """`TrakeStep.confidence` bị ràng [0, 1]; điểm fusion thì không.

    Chuẩn hoá theo điểm cao nhất trong CÙNG một lần trả về, không kẹp cứng —
    kẹp sẽ dồn mọi scene mạnh về đúng 1.0 và mất hết thứ tự.
    """

    if best <= 0:
        return 0.0
    return min(max(score / best, 0.0), 1.0)


def to_trake_results(
    sequences: list[SequenceHit], *, expected_steps: int | None = None
) -> list[TrakeResultItem]:
    """`SequenceHit` -> `TrakeResultItem`, giữ nguyên thứ hạng sẵn có."""

    if not sequences:
        return []
    best = max((scene.score for item in sequences for scene in item.scenes), default=0.0)

    results: list[TrakeResultItem] = []
    for rank, item in enumerate(sequences, start=1):
        steps = [
            TrakeStep(
                step=index,
                frame_idx=scene.best_frame_idx,
                scene_id=scene.scene_id,
                confidence=_confidence(scene.score, best),
                refinement="keyframe_only",
                image_path=scene.best_keyframe_path,
                timestamp_sec=scene.best_timestamp_sec,
            )
            for index, scene in enumerate(item.scenes, start=1)
        ]
        total = expected_steps or len(steps)
        # `link_event_hits` chỉ dựng chuỗi từ những step CÓ candidate, nên số
        # step thiếu suy ra từ chênh lệch với số step của truy vấn. Không ghi
        # lại thì output không phân biệt được "chuỗi đủ 3 bước" với "chuỗi 3
        # bước nhưng truy vấn hỏi 5".
        missing = list(range(len(steps) + 1, total + 1)) if total > len(steps) else []
        results.append(
            TrakeResultItem(
                rank=rank,
                video_id=item.video_id,
                frame_ids=[scene.best_frame_idx for scene in item.scenes],
                sequence_score=item.score,
                steps=steps,
                step_coverage=len(steps) / total if total else 0.0,
                # Chuỗi do `link_event_hits` dựng LUÔN tăng dần theo thời gian
                # (nó nối theo thứ tự event và ràng buộc frame tăng), nên thứ
                # tự đã đúng theo cấu trúc — không bịa thêm một số đo khác.
                ordering_score=1.0 if len(steps) > 1 else 0.0,
                missing_steps=missing,
            )
        )
    return results


__all__ = ["to_trake_results"]
