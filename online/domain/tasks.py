"""Canonical competition task taxonomy (PR-01).

Trước PR-01 domain dùng lẫn lộn `kis`/`vqa`/`sequence` — ba tên không khớp
với luật thi (Textual KIS / Q&A / TRAKE) và không có alias, nên client gửi
`"trake"` bị 422 còn client gửi `{"task": "qa"}` tới `/search/kis` thì bị
ghi đè im lặng.

Từ đây chỉ tồn tại MỘT enum. Alias được chuẩn hóa ở API boundary
(`SearchRequest.task` validator) chứ không lan vào service/adapter — mọi
so sánh bên trong domain đều so với `TaskType.*`.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Ba task chính thức của vòng sơ tuyển + AVS (task nội bộ mở rộng)."""

    TEXTUAL_KIS = "TEXTUAL_KIS"
    QA = "QA"
    TRAKE = "TRAKE"
    AVS = "AVS"


# Tên cũ và các cách viết thường gặp. Key đã casefold; tra cứu qua
# `normalize_task` để không phải nhớ dạng viết hoa.
TASK_ALIASES: dict[str, TaskType] = {
    "kis": TaskType.TEXTUAL_KIS,
    "textual_kis": TaskType.TEXTUAL_KIS,
    "textualkis": TaskType.TEXTUAL_KIS,
    "vqa": TaskType.QA,
    "qa": TaskType.QA,
    "q&a": TaskType.QA,
    "sequence": TaskType.TRAKE,
    "temporal": TaskType.TRAKE,
    "trake": TaskType.TRAKE,
    "avs": TaskType.AVS,
}


def normalize_task(value: object) -> TaskType:
    """Chấp nhận `TaskType`, tên canonical hoặc alias; ném ValueError nếu lạ."""

    if isinstance(value, TaskType):
        return value
    if not isinstance(value, str):
        raise ValueError(f"task must be a string, got {type(value).__name__}")
    key = value.strip().casefold()
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    raise ValueError(
        f"unknown task {value!r}; expected one of "
        + ", ".join(item.value for item in TaskType)
    )


__all__ = ["TASK_ALIASES", "TaskType", "normalize_task"]
