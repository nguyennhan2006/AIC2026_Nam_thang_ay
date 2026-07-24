from __future__ import annotations

import re
from dataclasses import dataclass

from .config import EngineConfig


@dataclass(slots=True)
class QueryPlan:
    task: str
    branch_weights: dict[str, float]
    hints: list[str]


OCR_HINTS = (
    "chữ", "văn bản", "biển", "logo", "tiêu đề", "phụ đề", "màn hình",
    "số điện thoại", "đọc được", "written", "text", "sign", "subtitle",
)
SPEECH_HINTS = (
    "nói", "phát biểu", "trả lời", "hỏi", "âm thanh", "giọng", "nghe",
    "said", "says", "speech", "announces", "mentions",
)
TEMPORAL_HINTS = (
    "sau đó", "trước khi", "tiếp theo", "đầu tiên", "cuối cùng", " rồi ",
    "before", "after", "then", "next", "finally",
)
ACTION_HINTS = (
    "đang", "hành động", "di chuyển", "đi", "chạy", "cầm", "mở", "đóng",
    "action", "moving", "walking", "running", "holding",
)


def _contains(text: str, hints: tuple[str, ...]) -> bool:
    padded = f" {text.casefold()} "
    return any(hint in padded for hint in hints)


def plan_query(text: str, config: EngineConfig, task: str = "auto") -> QueryPlan:
    task = task.casefold().strip()
    if task not in {"auto", "frame", "scene", "temporal", "qa"}:
        raise ValueError(f"Unsupported task: {task}")
    weights = {
        "semantic": config.semantic_weight,
        "ocr": config.ocr_weight,
        "speech": config.speech_weight,
        "tags": config.tags_weight,
        "event": config.event_weight,
        "scene_vector": config.scene_vector_weight,
        "frame_vector": config.frame_vector_weight,
    }
    hints: list[str] = []
    if _contains(text, OCR_HINTS) or bool(re.search(r"\b\d{2,}\b", text)):
        weights["ocr"] *= 1.8
        hints.append("ocr")
    if _contains(text, SPEECH_HINTS):
        weights["speech"] *= 1.8
        hints.append("speech")
    if task == "temporal" or _contains(text, TEMPORAL_HINTS):
        weights["event"] *= 1.8
        weights["tags"] *= 1.25
        hints.append("temporal")
    if _contains(text, ACTION_HINTS):
        weights["tags"] *= 1.45
        weights["event"] *= 1.25
        hints.append("action")
    if task == "frame":
        weights["frame_vector"] *= 1.5
        weights["scene_vector"] *= 0.8
        hints.append("frame")
    elif task in {"scene", "qa"}:
        weights["semantic"] *= 1.2
        weights["scene_vector"] *= 1.2
    return QueryPlan(task=task, branch_weights=weights, hints=hints)


def split_temporal_query(text: str) -> list[str]:
    parts = re.split(
        r"\s+(?:sau đó|tiếp theo|rồi|trước khi|then|next|after that|before)\s+",
        text,
        flags=re.IGNORECASE,
    )
    return [part.strip(" ,.;") for part in parts if part.strip(" ,.;")]
