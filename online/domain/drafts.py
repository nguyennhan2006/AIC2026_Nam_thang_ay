"""Bản nháp sắp xếp bài nộp, DÙNG CHUNG cả đội (FB-003).

Trước đây thứ tự sắp tay và đáp án sửa tay chỉ sống trong state React của một
tab trình duyệt: F5 là mất, và người ngồi máy bên cạnh không có cách nào thấy
được bản đã soát của người kia. Trong một buổi thi, hai người soát trùng nhau
một câu và bỏ trắng câu khác là chuyện xảy ra được.

Vì vậy nháp nằm ở SERVER chứ không phải localStorage: cả đội trỏ vào cùng một
backend, nên lưu ở đó là tự động thấy được của nhau.

Nháp KHÔNG phải bài nộp. Nó không đi qua `submission_validator`, không giới
hạn 100 dòng, và có thể chứa dòng còn dở. Chuyển thành bài nộp vẫn phải đi qua
`/v1/submissions/build` như cũ.
"""

from __future__ import annotations

from pydantic import Field

from online.domain.base import StrictModel
from online.domain.tasks import TaskType


class DraftRow(StrictModel):
    """Một dòng trong bản nháp.

    `frame_ids` chỉ dùng cho TRAKE (một dòng là cả chuỗi). `answer` chỉ dùng
    cho QA. Giữ cả hai ở đây thay vì tách ba kiểu dòng: bản nháp cần chở đúng
    thứ bảng nộp đang hiện, và bảng đó dùng chung một kiểu dòng cho mọi task.
    """

    video_id: str
    frame_idx: int
    frame_ids: list[int] = Field(default_factory=list)
    answer: str | None = None


class SubmissionDraft(StrictModel):
    draft_id: str
    name: str
    author: str = ""
    task: TaskType
    query: str = ""
    rows: list[DraftRow] = Field(default_factory=list)
    created_at: str
    updated_at: str


class DraftSaveRequest(StrictModel):
    """Lưu mới hoặc ghi đè.

    `draft_id` rỗng = tạo mới. Có `draft_id` = ghi đè đúng bản đó, để người
    soát bấm "Lưu" nhiều lần không đẻ ra mười bản trùng tên.
    """

    name: str
    author: str = ""
    task: TaskType
    query: str = ""
    rows: list[DraftRow] = Field(default_factory=list)
    draft_id: str | None = None


class DraftListResponse(StrictModel):
    drafts: list[SubmissionDraft] = Field(default_factory=list)


__all__ = [
    "DraftListResponse",
    "DraftRow",
    "DraftSaveRequest",
    "SubmissionDraft",
]
