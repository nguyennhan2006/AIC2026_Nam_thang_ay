"""Online read models. Canonical perception metadata remains in datasection."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from online.domain.base import StrictModel  # noqa: F401 - re-export cho chỗ import cũ
from online.domain.search_config import SearchOptions


class TaskType(StrEnum):
    KIS = "kis"
    AVS = "avs"
    SEQUENCE = "sequence"
    VQA = "vqa"


class Modality(StrEnum):
    VISUAL = "visual"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"
    KEYWORD = "keyword"
    # Search Mixing Console W3 — mỗi bucket mới có 1 retriever tương ứng, mặc định
    # KHÔNG được container đăng ký (xem AIC_ENABLE_* trong online/config.py) nên
    # thêm bucket ở đây không tự thay đổi hành vi search hiện tại.
    OBJECT = "object"
    ACTION = "action"
    COLOR = "color"
    EVENT = "event"


class SearchFilters(StrictModel):
    video_ids: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    has_ocr: bool | None = None
    has_asr: bool | None = None
    start_sec_gte: float | None = Field(default=None, ge=0)
    end_sec_lte: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "SearchFilters":
        if (
            self.start_sec_gte is not None
            and self.end_sec_lte is not None
            and self.end_sec_lte < self.start_sec_gte
        ):
            raise ValueError("end_sec_lte must be >= start_sec_gte")
        return self


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    task: TaskType = TaskType.KIS
    top_k: int = Field(default=20, ge=1, le=200)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    debug: bool = False
    # Search Mixing Console (W0) — None = hành vi search hiện tại, không đổi mặc
    # định. Chưa đọc bởi SearchService/container.py (đó là W3/W5).
    search_options: SearchOptions | None = None


class QueryEvent(StrictModel):
    event_idx: int = Field(ge=0)
    text: str = Field(min_length=1)
    exact_phrases: list[str] = Field(default_factory=list)


class QueryPlan(StrictModel):
    task: TaskType
    original_query: str
    normalized_query: str
    events: list[QueryEvent] = Field(min_length=1)
    modality_weights: dict[Modality, float]
    filters: SearchFilters
    # Search Mixing Console (W5) — per-branch override, copied verbatim from
    # SearchRequest.search_options. Default SearchOptions() has empty `branches`,
    # so every retriever's per-branch lookup misses and falls back to
    # modality_weights exactly like before this field existed.
    search_options: SearchOptions = Field(default_factory=SearchOptions)

    @model_validator(mode="after")
    def validate_weights(self) -> "QueryPlan":
        if any(weight < 0 for weight in self.modality_weights.values()):
            raise ValueError("modality weights must be non-negative")
        if not any(self.modality_weights.values()):
            raise ValueError("at least one modality weight must be positive")
        return self


class SceneDocument(StrictModel):
    """Read-only projection built from one canonical datasection Scene."""

    scene_id: str
    video_id: str
    video_path: str | None = None
    scene_idx: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    keyframe_ids: list[str] = Field(default_factory=list)
    keyframe_paths: list[str] = Field(default_factory=list)
    keyframe_timestamps: list[float] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    keyframe_evidence: list[dict[str, Any]] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    asr_texts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # W1 action tags / color features, projected read-only (xem project_scene).
    action_tags: list[str] = Field(default_factory=list)
    color_names: list[str] = Field(default_factory=list)

    def field_text(self, field: str) -> str:
        values = {
            "caption": self.captions,
            "ocr": self.ocr_texts,
            "asr": self.asr_texts,
            "keyword": self.keywords,
            "object": self.object_labels,
            "action": self.action_tags,
        }.get(field, [])
        return " ".join(values)


class EventDocument(StrictModel):
    """Read-only projection built from one canonical datasection Event."""

    event_id: str
    video_id: str
    scene_ids: list[str] = Field(default_factory=list)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    event_caption: str | None = None
    keywords: list[str] = Field(default_factory=list)
    action_tags: list[str] = Field(default_factory=list)
    previous_event_id: str | None = None
    next_event_id: str | None = None

    def field_text(self, field: str = "text") -> str:
        # `field` is ignored — kept so EventDocument can be reused with the same
        # BM25Index used for scene fields (online/adapters/bm25.py), which is
        # built generically around `item.field_text(field)`.
        return " ".join([self.event_caption or "", *self.keywords, *self.action_tags])


class Candidate(StrictModel):
    entity_id: str
    scene_id: str
    video_id: str
    source: str
    modality: Modality
    score: float
    rank: int = Field(ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class Evidence(StrictModel):
    modality: Modality
    text: str
    score: float


class SearchHit(StrictModel):
    scene_id: str
    video_id: str
    video_path: str | None = None
    scene_idx: int = Field(ge=0)
    start_sec: float
    end_sec: float
    score: float
    keyframe_ids: list[str] = Field(default_factory=list)
    keyframe_paths: list[str] = Field(default_factory=list)
    keyframe_timestamps: list[float] = Field(default_factory=list)
    best_keyframe_id: str | None = None
    best_keyframe_path: str | None = None
    best_timestamp_sec: float | None = None
    matched_modalities: list[Modality] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)


class SequenceHit(StrictModel):
    video_id: str
    score: float
    scenes: list[SearchHit] = Field(min_length=2)


class SearchResponse(StrictModel):
    query_id: str
    task: TaskType
    took_ms: float
    results: list[SearchHit] = Field(default_factory=list)
    sequences: list[SequenceHit] = Field(default_factory=list)
    query_plan: QueryPlan | None = None


class VQARequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k_evidence: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    debug: bool = False


class VQAResponse(StrictModel):
    query_id: str
    answer: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[SearchHit]
    took_ms: float


EntityType = Literal["scene", "keyframe"]
