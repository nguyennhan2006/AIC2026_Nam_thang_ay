"""Submission contracts — đúng format chính thức BTC (PR-08).

Format đã xác nhận trong `docs/12_USER_GUIDE.md` §6 (mục "Xuất CSV nộp bài"):

    KIS / AVS / TRAKE   mỗi dòng  <video_id>, <frame_id>[, <frame_id_2>, ...]
    QA                  mỗi dòng  <video_id>, <frame_id>, "<answer>"

Không có header, tối đa 100 dòng. `frame_idx` là **true frame index** của
video gốc — không phải thứ tự keyframe (keyframe thứ 3 của một scene không
phải frame số 3).
"""

from __future__ import annotations

from pydantic import Field

from online.domain.base import StrictModel
from online.domain.task_results import KisResultItem, QaResultItem, TrakeResultItem
from online.domain.tasks import TaskType


class KisSubmissionItem(StrictModel):
    video_id: str
    frame_idx: int


class QaSubmissionItem(StrictModel):
    video_id: str
    frame_idx: int
    answer: str


class TrakeSubmissionItem(StrictModel):
    video_id: str
    frame_ids: list[int]


# --------------------------------------------------------------------------
# API request/response cho POST /v1/submissions/*
# --------------------------------------------------------------------------


class SubmissionBuildRequest(StrictModel):
    """Đầu vào của `/v1/submissions/build` — kết quả task đã có sẵn từ một
    lần gọi `/v1/search/*` trước đó (client tự truyền lại, không cần session
    lưu phía server cho tới khi PR-09 thêm search-session)."""

    task: TaskType
    kis: list[KisResultItem] = Field(default_factory=list)
    qa: list[QaResultItem] = Field(default_factory=list)
    trake: list[TrakeResultItem] = Field(default_factory=list)


class SubmissionIssue(StrictModel):
    severity: str
    code: str
    message: str
    row_index: int | None = None


class SubmissionBuildResponse(StrictModel):
    task: TaskType
    item_count: int
    csv: str
    has_errors: bool
    issues: list[SubmissionIssue] = Field(default_factory=list)


class GoldIntervalIn(StrictModel):
    start_frame: int
    end_frame: int = Field(description="inclusive")


class EvaluateLocalRequest(SubmissionBuildRequest):
    """Chấm thử submission trên một gold nhỏ người dùng tự dán vào.

    `intervals` dùng cho KIS/QA; `step_windows` dùng cho TRAKE. Không nhằm
    thay thế `scripts/eval_tasks.py` (bộ gold 40 query) — đây là "xem thử
    điểm trước khi nộp" cho một câu cụ thể.
    """

    video_id: str
    intervals: list[GoldIntervalIn] = Field(default_factory=list)
    accepted_answers: list[str] = Field(default_factory=list)
    step_windows: list[GoldIntervalIn] = Field(default_factory=list)


class EvaluateLocalResponse(StrictModel):
    score: float
    best_rank: int | None
    detail: str


__all__ = [
    "EvaluateLocalRequest",
    "EvaluateLocalResponse",
    "GoldIntervalIn",
    "KisSubmissionItem",
    "QaSubmissionItem",
    "SubmissionBuildRequest",
    "SubmissionBuildResponse",
    "SubmissionIssue",
    "TrakeSubmissionItem",
]
