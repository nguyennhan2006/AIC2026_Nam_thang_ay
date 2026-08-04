"""Kiểm tra submission trước khi xuất — chặn lỗi rẻ nhất có thể chặn (PR-08).

Mọi lỗi ở đây là lỗi *sẽ* làm mất điểm khi nộp thật: quá 100 dòng, frame âm,
frame vượt quá độ dài video thật, answer rỗng, TRAKE thiếu step. Validator
không tự sửa — chỉ báo, vì tự động sửa (vd cắt bớt dòng) có thể xóa mất đúng
dòng người dùng muốn giữ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from online.competition.rules import MAX_SUBMISSION_ITEMS
from online.domain.submission import KisSubmissionItem, QaSubmissionItem, TrakeSubmissionItem

Severity = Literal["error", "warning"]

# Trả về frame_count thật của video, hoặc None nếu không biết (không kiểm
# tra được, không phải "mọi frame đều sai").
FrameCountLookup = Callable[[str], Awaitable[int | None]]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    row_index: int | None = None


def _check_size(count: int) -> list[ValidationIssue]:
    if count == 0:
        return [ValidationIssue("error", "empty_submission", "submission không có dòng nào")]
    if count > MAX_SUBMISSION_ITEMS:
        return [ValidationIssue(
            "error", "too_many_rows",
            f"submission có {count} dòng, vượt giới hạn {MAX_SUBMISSION_ITEMS}",
        )]
    return []


async def _check_frame_bounds(
    video_id: str, frame_idx: int, index: int, lookup: FrameCountLookup | None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if frame_idx < 0:
        issues.append(ValidationIssue(
            "error", "negative_frame", f"frame_idx={frame_idx} < 0", row_index=index,
        ))
        return issues
    if lookup is None:
        return issues
    frame_count = await lookup(video_id)
    if frame_count is None:
        issues.append(ValidationIssue(
            "warning", "unknown_video",
            f"không rõ frame_count của video {video_id!r} — không kiểm tra được biên",
            row_index=index,
        ))
    elif frame_idx >= frame_count:
        issues.append(ValidationIssue(
            "error", "frame_out_of_bounds",
            f"frame_idx={frame_idx} vượt quá frame_count={frame_count} của video {video_id!r} "
            "— có thể đang nộp thứ tự keyframe thay vì true frame index",
            row_index=index,
        ))
    return issues


async def validate_kis(
    items: list[KisSubmissionItem], *, frame_count: FrameCountLookup | None = None
) -> list[ValidationIssue]:
    issues = _check_size(len(items))
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(items):
        issues.extend(await _check_frame_bounds(item.video_id, item.frame_idx, index, frame_count))
        key = (item.video_id, item.frame_idx)
        if key in seen:
            issues.append(ValidationIssue(
                "warning", "duplicate_row", f"dòng {index} trùng {key}", row_index=index,
            ))
        seen.add(key)
    return issues


async def validate_qa(
    items: list[QaSubmissionItem], *, frame_count: FrameCountLookup | None = None
) -> list[ValidationIssue]:
    issues = _check_size(len(items))
    seen: set[tuple[str, int, str]] = set()
    for index, item in enumerate(items):
        issues.extend(await _check_frame_bounds(item.video_id, item.frame_idx, index, frame_count))
        if not item.answer.strip():
            issues.append(ValidationIssue(
                "error", "empty_answer", "answer rỗng — chắc chắn 0 điểm theo luật", row_index=index,
            ))
        elif len(item.answer) > 100:
            # Giới hạn hiển thị của form nộp bài (docs/12_USER_GUIDE.md §6).
            issues.append(ValidationIssue(
                "warning", "answer_too_long",
                f"answer dài {len(item.answer)} ký tự, vượt 100 — có thể bị cắt khi nộp",
                row_index=index,
            ))
        key = (item.video_id, item.frame_idx, item.answer)
        if key in seen:
            issues.append(ValidationIssue(
                "warning", "duplicate_row", f"dòng {index} trùng {key}", row_index=index,
            ))
        seen.add(key)
    return issues


async def validate_trake(
    items: list[TrakeSubmissionItem],
    *,
    frame_count: FrameCountLookup | None = None,
    expected_steps: int | None = None,
) -> list[ValidationIssue]:
    issues = _check_size(len(items))
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for index, item in enumerate(items):
        if expected_steps is not None and len(item.frame_ids) != expected_steps:
            issues.append(ValidationIssue(
                "error", "wrong_step_count",
                f"dòng {index} có {len(item.frame_ids)} frame, cần đúng {expected_steps} step",
                row_index=index,
            ))
        if any(later <= earlier for earlier, later in zip(item.frame_ids, item.frame_ids[1:])):
            issues.append(ValidationIssue(
                "error", "frames_not_increasing",
                f"dòng {index}: frame_ids phải tăng dần nghiêm ngặt, nhận {item.frame_ids}",
                row_index=index,
            ))
        for frame_idx in item.frame_ids:
            issues.extend(await _check_frame_bounds(item.video_id, frame_idx, index, frame_count))
        key = (item.video_id, tuple(item.frame_ids))
        if key in seen:
            issues.append(ValidationIssue(
                "warning", "duplicate_row", f"dòng {index} trùng {key}", row_index=index,
            ))
        seen.add(key)
    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(item.severity == "error" for item in issues)


__all__ = [
    "FrameCountLookup",
    "Severity",
    "ValidationIssue",
    "has_errors",
    "validate_kis",
    "validate_qa",
    "validate_trake",
]
