"""Deterministic Vietnamese/English query normalization and event planning."""

from __future__ import annotations

import re
import unicodedata

from online.domain.models import (
    Modality,
    QueryEvent,
    QueryPlan,
    SearchRequest,
    TaskType,
)
from online.errors import InvalidQueryError


QUOTED_RE = re.compile(r'["“”]([^"“”]{2,})["“”]')
SPACE_RE = re.compile(r"\s+")
TEMPORAL_RE = re.compile(
    r"\b(?:sau đó|tiếp theo|kế tiếp|cuối cùng|rồi|then|next|finally)\b",
    flags=re.IGNORECASE,
)
SPEECH_HINTS = {
    "nói",
    "phát biểu",
    "trình bày",
    "nghe thấy",
    "lời thoại",
    "says",
    "speaks",
    "speech",
    "heard",
}
TEXT_HINTS = {"chữ", "dòng chữ", "biển", "bảng", "khẩu hiệu", "text", "sign"}


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).strip()
    normalized = SPACE_RE.sub(" ", normalized)
    if not normalized:
        raise InvalidQueryError("query is empty after normalization")
    return normalized


class RuleBasedQueryPlanner:
    """Safe V1 planner; an LLM planner can replace it through the same output model."""

    async def plan(self, request: SearchRequest) -> QueryPlan:
        normalized = normalize_query(request.query)
        exact_phrases = [item.strip() for item in QUOTED_RE.findall(normalized)]
        parts = [item.strip(" ,.;:") for item in TEMPORAL_RE.split(normalized)]
        parts = [item for item in parts if item]
        if request.task != TaskType.SEQUENCE or len(parts) < 2:
            parts = [normalized]
        events = [
            QueryEvent(
                event_idx=index,
                text=text,
                exact_phrases=[phrase for phrase in exact_phrases if phrase in text],
            )
            for index, text in enumerate(parts)
        ]
        lowered = normalized.casefold()
        weights = {
            Modality.VISUAL: 1.0,
            Modality.CAPTION: 1.0,
            Modality.OCR: 0.35,
            Modality.ASR: 0.25,
            Modality.KEYWORD: 0.65,
        }
        if exact_phrases or any(hint in lowered for hint in TEXT_HINTS):
            weights[Modality.OCR] = 2.0
        if any(hint in lowered for hint in SPEECH_HINTS):
            weights[Modality.ASR] = 1.7
        return QueryPlan(
            task=request.task,
            original_query=request.query,
            normalized_query=normalized,
            events=events,
            modality_weights=weights,
            filters=request.filters,
        )

