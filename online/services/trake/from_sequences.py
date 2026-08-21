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

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument, SequenceHit
from online.domain.task_results import TrakeResultItem, TrakeStep


def _confidence(score: float, best: float) -> float:
    """`TrakeStep.confidence` bị ràng [0, 1]; điểm fusion thì không.

    Chuẩn hoá theo điểm cao nhất trong CÙNG một lần trả về, không kẹp cứng —
    kẹp sẽ dồn mọi scene mạnh về đúng 1.0 và mất hết thứ tự.
    """

    if best <= 0:
        return 0.0
    return min(max(score / best, 0.0), 1.0)


def _fill_holes(
    item: SequenceHit,
    steps: list[TrakeStep],
    total: int,
    documents: dict[str, SceneDocument] | None,
) -> list[TrakeStep]:
    """Lấp step thiếu bằng keyframe THẬT, đặt đúng VỊ TRÍ step của nó.

    Thứ tự ưu tiên — **độ khớp trước, hình học sau**:

    1. Chỉ xét keyframe của những scene ĐÃ CÓ BẰNG CHỨNG cho truy vấn này, tức
       nằm trong `documents` vì chúng lọt vào pool candidate của một step nào
       đó. Đây là phần "độ khớp": không rải đều toàn video mà chỉ lấy vùng mà
       hệ đã thấy liên quan.
    2. Trong vùng đó, chọn keyframe gần điểm nội suy theo TỈ LỆ VỊ TRÍ STEP
       nhất — step 2 của khoảng (1, 5) phải nằm gần đầu khoảng, không phải
       chính giữa.
    3. Không có gì trong khoảng -> để trống. KHÔNG bịa số frame: luật chỉ chấm
       frame có thật trong `keyframes.jsonl`, nên frame bịa vừa chắc chắn mất
       điểm vừa làm người dùng tin nhầm là đã tìm được.

    Vì sao không lấy candidate của CHÍNH step thiếu: nếu step đó còn candidate
    hợp lệ trong khoảng thì beam đã dùng rồi — lấy một hit được +0.04 điểm
    scene, trong khi bỏ qua chỉ mất 0.01 tiền phạt. Lỗ thủng theo định nghĩa là
    chỗ không còn lựa chọn nào có bằng chứng riêng.
    """

    if not documents:
        return steps
    covered = {step.step: step for step in steps}
    holes = [number for number in range(1, total + 1) if number not in covered]
    if not holes:
        return steps

    pool: list[FrameEvidence] = []
    for document in documents.values():
        if document.video_id == item.video_id:
            pool.extend(document.keyframes)
    if not pool:
        return steps
    pool.sort(key=lambda frame: frame.frame_idx)
    taken = {step.frame_idx for step in steps}

    for number in holes:
        before = max((n for n in covered if n < number), default=None)
        after = min((n for n in covered if n > number), default=None)
        low = covered[before].frame_idx if before is not None else -1
        high = covered[after].frame_idx if after is not None else float("inf")
        window = [
            frame for frame in pool
            if low < frame.frame_idx < high and frame.frame_idx not in taken
        ]
        if not window:
            continue
        if before is not None and after is not None:
            ratio = (number - before) / (after - before)
            target = low + (high - low) * ratio
        elif after is not None:
            target = window[0].frame_idx
        else:
            target = window[-1].frame_idx
        chosen = min(window, key=lambda frame: abs(frame.frame_idx - target))
        taken.add(chosen.frame_idx)
        covered[number] = TrakeStep(
            step=number,
            frame_idx=chosen.frame_idx,
            scene_id=chosen.scene_id,
            confidence=0.1,
            refinement="interpolated",
            image_path=chosen.image_path,
            timestamp_sec=chosen.timestamp_sec,
        )
    return [covered[number] for number in sorted(covered)]


def to_trake_results(
    sequences: list[SequenceHit],
    *,
    expected_steps: int | None = None,
    documents: dict[str, SceneDocument] | None = None,
) -> list[TrakeResultItem]:
    """`SequenceHit` -> `TrakeResultItem`, giữ nguyên thứ hạng sẵn có.

    Truyền `documents` để bật lấp lỗ: chuỗi thiếu step vẫn xuất đủ số frame nên
    nộp được, phần lấp đánh dấu `refinement="interpolated"`.
    """

    if not sequences:
        return []
    best = max((scene.score for item in sequences for scene in item.scenes), default=0.0)

    results: list[TrakeResultItem] = []
    for rank, item in enumerate(sequences, start=1):
        steps = [
            TrakeStep(
                step=step_no,
                frame_idx=scene.best_frame_idx,
                scene_id=scene.scene_id,
                confidence=_confidence(scene.score, best),
                refinement="keyframe_only",
                image_path=scene.best_keyframe_path,
                timestamp_sec=scene.best_timestamp_sec,
            )
            for step_no, scene in zip(item.covered_steps, item.scenes, strict=True)
        ]
        total = expected_steps or item.total_steps or len(steps)
        # Lấy thẳng từ `covered_steps` thay vì suy ra từ chênh lệch số lượng:
        # cách suy ra chỉ đúng khi lỗ thủng nằm ở ĐUÔI. Từ khi
        # `allow_missing_steps` cho phép bỏ step ở GIỮA, suy ra là gán nhầm
        # frame cho step — sai lặng lẽ và không có gì báo.
        covered = set(item.covered_steps)
        missing = [step for step in range(1, total + 1) if step not in covered]
        if missing:
            steps = _fill_holes(item, steps, total, documents)
            filled = {step.step for step in steps}
            missing = [number for number in missing if number not in filled]
        results.append(
            TrakeResultItem(
                rank=rank,
                video_id=item.video_id,
                frame_ids=[step.frame_idx for step in steps],
                sequence_score=item.score,
                steps=steps,
                step_coverage=len(steps) / total if total else 0.0,
                # Chuỗi do `link_event_hits` dựng LUÔN tăng dần theo thời gian
                # (nó nối theo thứ tự event và ràng buộc frame tăng), nên thứ
                # tự đã đúng theo cấu trúc — không bịa thêm một số đo khác.
                ordering_score=1.0 if len(steps) > 1 else 0.0,
                # Chuỗi thủng phải nhìn thấy được ở tầng hiển thị: người dùng
                # cần biết dòng này chỉ bắt được 2/5 step trước khi bỏ công xem.
                #
                # Tính CẢ phần đã lấp: `missing` được tính sau khi lấp, nên chỉ
                # dựa vào nó thì một chuỗi từng thủng rồi được lấp bằng frame
                # nội suy sẽ trông y hệt chuỗi lành. Đó đúng là thứ người dùng
                # cần biết để chọn dòng nào đáng mở video ra xem.
                degraded=bool(missing)
                or any(step.refinement == "interpolated" for step in steps),
                missing_steps=missing,
            )
        )
    return results


__all__ = ["to_trake_results"]
