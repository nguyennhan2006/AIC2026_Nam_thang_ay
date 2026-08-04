"""Search session trace — replay và audit (PR-09).

Mỗi lần search phải để lại đủ dấu vết để: (1) replay đúng y hệt request cũ,
(2) so sánh hai lần chạy khác cấu hình, (3) debug "tại sao kết quả này lại ở
đây" sau khi đã đóng tab. Trước PR-09 không có gì được lưu lại — `query_id`
sinh ra rồi vứt đi ngay sau response.

Lưu **request gốc** (`raw_request`) chứ không lưu lại toàn bộ candidate list:
đủ để replay (chạy lại `search(raw_request)` cho đúng kết quả, giả sử dataset
chưa đổi) mà không phình bộ nhớ theo số candidate mỗi phiên.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from online.domain.base import StrictModel
from online.domain.execution import BranchStatus
from online.domain.models import PipelineStatus, SearchRequest
from online.domain.tasks import TaskType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SearchExecutionTrace(StrictModel):
    """Dấu vết của một lần search — khóa theo `session_id` (= `query_id`)."""

    session_id: str
    task: TaskType
    raw_request: SearchRequest
    branch_status: list[BranchStatus] = Field(default_factory=list)
    status: PipelineStatus = "COMPLETED"
    warnings: list[str] = Field(default_factory=list)
    took_ms: float = 0.0
    dataset_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    # Số lần trace này đã được dùng để replay — không đổi raw_request, chỉ
    # đếm để thấy phiên nào đang được lặp lại nhiều (đáng nghi/đáng debug).
    replay_count: int = 0


__all__ = ["SearchExecutionTrace"]
