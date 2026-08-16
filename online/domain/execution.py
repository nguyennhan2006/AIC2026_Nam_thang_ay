"""Trạng thái thực thi từng branch (PR-03).

Trước PR-03 `SearchService._retrieve` gọi `asyncio.gather` trần: không
timeout, không `return_exceptions`. Một branch chết (Qdrant treo, event index
lỗi) làm cả request trả 500, và không có cách nào biết branch nào đã chạy.

Từ đây mỗi branch tự bắt lỗi của mình rồi trả về một `BranchStatus` có kiểu.
Search vẫn hoàn tất miễn còn ít nhất một branch thành công, và response nói
rõ branch nào timeout/lỗi — không silent degradation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from online.domain.base import StrictModel
from online.domain.candidate import Modality

BranchState = Literal["success", "disabled", "unavailable", "timeout", "failed", "empty"]

# Backend thật sự đứng sau một branch. `lexical_fallback` tồn tại để một
# retriever hash/BM25 không bao giờ được báo cáo như dense vector search —
# nhầm chỗ này làm sai toàn bộ số liệu ablation.
BackendKind = Literal[
    "vector", "lexical", "lexical_fallback", "fuzzy", "metadata", "rule", "remote"
]


class BranchStatus(StrictModel):
    """Kết quả chạy của đúng một execution trong đúng một request."""

    execution_id: str
    branch_id: str
    state: BranchState
    latency_ms: int = Field(ge=0)
    candidate_count: int = Field(default=0, ge=0)
    warning: str | None = None

    @property
    def is_degraded(self) -> bool:
        return self.state in ("timeout", "failed", "unavailable")


class BranchCapabilities(StrictModel):
    """Mô tả một branch cho `/v1/search/capabilities` — UI render từ đây.

    `supported_controls` liệt kê đúng những field của `BranchRuntimeOptions`
    mà branch này THỰC SỰ đọc. UI không được hiện control nằm ngoài danh sách
    (nguyên tắc "không control giả").
    """

    branch_id: str
    execution_ids: list[str] = Field(default_factory=list)
    modality: Modality | None = None
    backend_kind: BackendKind
    available: bool = True
    degraded: bool = False
    degraded_reason: str | None = None
    model_id: str | None = None
    index_id: str | None = None
    supported_controls: list[str] = Field(default_factory=list)
    # Trọng số MẶC ĐỊNH mức triển khai (AIC_BRANCH_WEIGHTS). `None` = không đặt,
    # nhánh rơi về trọng số theo modality mà planner tính cho từng truy vấn.
    #
    # Có mặt ở đây vì trước đó muốn biết "nhánh nào đang chạy ở mức nào" phải
    # đối chiếu SÁU biến bool, một chuỗi trọng số, và mặc định trong code —
    # ba nguồn, không nguồn nào tự nói ra kết quả cuối cùng.
    default_weight: float | None = None


__all__ = ["BackendKind", "BranchCapabilities", "BranchState", "BranchStatus"]
