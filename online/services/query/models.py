"""Data models for Query Routing V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class QueryIntent(Enum):
    """Query intent classification — determines which retrieval branches get priority."""

    VISUAL = "visual"           # "chiếc xe màu gì", "người đang làm gì"
    OCR = "ocr"                # "biển ghi gì", "số điện thoại là gì"
    ASR = "asr"               # "người đàn ông nói gì", "ai được nhắc đến"
    MIXED = "mixed"            # mixed modality
    NUMERIC_OCR = "numeric_ocr"  # "số trên cân là bao nhiêu", "giá bao nhiêu"
    TEMPORAL = "temporal"      # "đầu tiên..., sau đó..., cuối cùng..."


class AnswerType(Enum):
    """Expected answer type for QA/AVS tasks."""

    UNKNOWN = "unknown"
    NUMERIC = "numeric"        # số: kg, điểm, phút, tuổi, giá
    TEXT = "text"              # chữ: tên, biển, nhãn
    COLOR = "color"            # màu sắc
    OBJECT = "object"          # đồ vật: xe, cá, người
    ACTION = "action"          # hành động: đang làm gì
    LOCATION = "location"      # vị trí: ở đâu
    PERSON = "person"          # người: ai
    TIME = "time"              # thời gian: khi nào
    BOOLEAN = "boolean"        # có/không


@dataclass(frozen=True, slots=True)
class SearchQueryBundle:
    """Query bundle với nhiều specialized queries cho từng retriever.

    Thay vì một query chung cho mọi engine, mỗi retriever nhận query phù hợp
    với modality của nó:

        - Jina CLIP nhận visual_query (tập trung objects/actions/scenes)
        - BM25 Caption nhận caption_query (giữ nouns/verbs, expansion)
        - BM25 OCR nhận ocr_query (tập trung expected text/keywords)
        - BM25 ASR nhận asr_query (semantic question + speech keywords)

    Attributes:
        raw_query: Query gốc từ user.
        normalized_query: Query đã normalize (NFC, lowercase, strip).
        visual_query: Query cho visual retrieval — thiên về objects, actions,
            scenes, spatial relations. LOẠI BỎ câu hỏi trừu tượng ("bao nhiêu",
            "là gì", "tại sao").
        visual_query_en: English augmentation cho Jina CLIP v2.
        caption_query: Query cho BM25 caption — giữ nouns/verbs, có expansion,
            có thể giữ context ngắn.
        ocr_query: Query cho OCR retrieval — chỉ expected text/keywords,
            không visual description.
        asr_query: Query cho ASR retrieval — semantic question + speech keywords,
            không visual description.
        context_query: Phần context của query (cho multi-event queries).
        target_query: Phần target/moment cần tìm.
        exact_phrases: Các cụm trong ngoặc kép — cho OCR exact matching.
        events: Danh sách event nếu query có temporal structure.
        intent: Query intent classification.
        answer_type: Expected answer type cho QA tasks.
        expected_units: Các đơn vị mong đợi (kg, g, điểm, phút...).
        visual_entities: Các entity quan trọng trích xuất từ query.
        actions: Các action trong query.
        attributes: Các attribute quan trọng (màu sắc, kích thước...).
        complexity_score: Điểm phức tạp của query (0=simple, 3+=complex).
        debug_info: Thông tin debug cho UI.
    """

    # Core queries
    raw_query: str = ""
    normalized_query: str = ""

    # Engine-specific queries
    visual_query: str = ""
    visual_query_en: str = ""
    caption_query: str = ""
    ocr_query: str = ""
    asr_query: str = ""

    # Structure
    context_query: str = ""
    target_query: str = ""
    exact_phrases: list[str] = field(default_factory=list)

    # Events (for TRAKE/multi-event)
    events: list[str] = field(default_factory=list)

    # Classification
    intent: QueryIntent = QueryIntent.MIXED
    answer_type: AnswerType = AnswerType.UNKNOWN
    expected_units: list[str] = field(default_factory=list)

    # Extracted entities
    visual_entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)

    # Complexity
    complexity_score: int = 0

    # Debug
    debug_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        # Ensure empty strings are empty, not None
        if self.raw_query is None:
            object.__setattr__(self, "raw_query", "")
        if self.normalized_query is None:
            object.__setattr__(self, "normalized_query", "")

    @property
    def has_visual_query(self) -> bool:
        return bool(self.visual_query.strip())

    @property
    def has_en_visual_query(self) -> bool:
        return bool(self.visual_query_en.strip())

    @property
    def has_caption_query(self) -> bool:
        return bool(self.caption_query.strip())

    @property
    def has_ocr_query(self) -> bool:
        return bool(self.ocr_query.strip())

    @property
    def has_asr_query(self) -> bool:
        return bool(self.asr_query.strip())

    @property
    def is_complex(self) -> bool:
        return self.complexity_score >= 2

    @property
    def is_numeric_qa(self) -> bool:
        return self.intent == QueryIntent.NUMERIC_OCR or self.answer_type == AnswerType.NUMERIC

    @property
    def needs_ocr(self) -> bool:
        return self.intent in (QueryIntent.OCR, QueryIntent.NUMERIC_OCR, QueryIntent.MIXED)

    @property
    def needs_asr(self) -> bool:
        return self.intent in (QueryIntent.ASR, QueryIntent.MIXED)

    @property
    def is_temporal(self) -> bool:
        return self.intent == QueryIntent.TEMPORAL or len(self.events) >= 2

    def to_debug_dict(self) -> dict:
        """Convert to debug-friendly dict for UI."""
        return {
            "task": self.debug_info.get("task", "unknown"),
            "intent": self.intent.value,
            "answer_type": self.answer_type.value,
            "raw": self.raw_query,
            "normalized": self.normalized_query,
            "visual": self.visual_query,
            "visual_en": self.visual_query_en,
            "caption": self.caption_query,
            "ocr": self.ocr_query,
            "asr": self.asr_query,
            "exact_phrases": self.exact_phrases,
            "expected_units": self.expected_units,
            "events": self.events,
            "complexity": self.complexity_score,
            "entities": self.visual_entities,
            "actions": self.actions,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, slots=True)
class EngineQueries:
    """Queries cho một retriever engine cụ thể."""

    retriever_id: str
    primary_query: str = ""
    secondary_queries: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        """Tất cả query cần encode/search."""
        queries = [self.primary_query] if self.primary_query else []
        queries.extend(self.secondary_queries)
        return [q for q in queries if q.strip()]

    @property
    def is_valid(self) -> bool:
        return bool(self.all_queries)
