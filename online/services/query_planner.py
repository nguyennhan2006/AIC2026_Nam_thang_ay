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
from online.domain.search_config import SearchOptions
from online.errors import InvalidQueryError


QUOTED_RE = re.compile(r'["“”]([^"“”]{2,})["“”]')
SPACE_RE = re.compile(r"\s+")
TEMPORAL_RE = re.compile(
    r"\b(?:sau đó|tiếp theo|kế tiếp|cuối cùng|rồi|then|next|finally)\b",
    flags=re.IGNORECASE,
)
# Gold TRAKE query dùng format liệt kê đánh số "(1) ...; (2) ...; (3) ..."
# (xem examples/AIC2026_L21_V001_queries_4tasks.jsonl) — KHÔNG dùng từ nối
# tiếp diễn kiểu "sau đó"/"cuối cùng" mà TEMPORAL_RE bắt. Không có nhánh này,
# mọi query TRAKE thật rơi về đúng 1 event, khiến `len(plan.events) >= 2` ở
# search.py luôn False và TrakeProcessor không bao giờ chạy.
NUMBERED_STEP_RE = re.compile(r"\(\d+\)\s*")
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


def compute_modality_weights(text: str, exact_phrases: list[str]) -> dict[Modality, float]:
    """Suy modality weight từ MỘT đoạn text — dùng cho cả full query lẫn
    từng event riêng của TRAKE (PR-14A: trước đây mọi step TRAKE dùng chung
    weight của cả câu, nên step không có OCR/ASR vẫn bị đẩy nhánh sai)."""

    lowered = text.casefold()
    weights = {
        Modality.VISUAL: 1.0,
        Modality.CAPTION: 1.0,
        Modality.OCR: 0.35,
        Modality.ASR: 0.25,
        Modality.KEYWORD: 0.65,
        # Buckets mới (W3) chỉ có hiệu lực khi container thực sự đăng ký
        # retriever tương ứng (mặc định tắt, xem AIC_ENABLE_* ở online/config.py)
        # — giá trị ở đây chỉ là default hợp lý cho lúc retriever được bật.
        Modality.OBJECT: 0.5,
        Modality.ACTION: 0.5,
        Modality.COLOR: 0.4,
        Modality.EVENT: 0.3,
    }
    if exact_phrases or any(hint in lowered for hint in TEXT_HINTS):
        weights[Modality.OCR] = 2.0
    if any(hint in lowered for hint in SPEECH_HINTS):
        weights[Modality.ASR] = 1.7
    return weights


class RuleBasedQueryPlanner:
    """Safe V1 planner; an LLM planner can replace it through the same output model."""

    async def plan(self, request: SearchRequest) -> QueryPlan:
        task = request.task or TaskType.TEXTUAL_KIS
        normalized = normalize_query(request.query)
        exact_phrases = [item.strip() for item in QUOTED_RE.findall(normalized)]
        parts = [normalized]
        if task == TaskType.TRAKE:
            # Ưu tiên format đánh số "(1)...(2)..." — đúng format gold thật.
            # `[1:]` bỏ phần dẫn trước "(1)" (vd "...căn chỉnh bốn khoảnh khắc:").
            numbered = [
                item.strip(" ,.;:") for item in NUMBERED_STEP_RE.split(normalized)[1:]
            ]
            numbered = [item for item in numbered if item]
            if len(numbered) >= 2:
                parts = numbered
            else:
                temporal = [item.strip(" ,.;:") for item in TEMPORAL_RE.split(normalized)]
                temporal = [item for item in temporal if item]
                if len(temporal) >= 2:
                    parts = temporal
        events = [
            QueryEvent(
                event_idx=index,
                text=text,
                exact_phrases=[phrase for phrase in exact_phrases if phrase in text],
            )
            for index, text in enumerate(parts)
        ]
        weights = compute_modality_weights(normalized, exact_phrases)
        return QueryPlan(
            task=task,
            original_query=request.query,
            normalized_query=normalized,
            events=events,
            modality_weights=weights,
            filters=request.filters,
            search_options=request.search_options or SearchOptions(),
        )

