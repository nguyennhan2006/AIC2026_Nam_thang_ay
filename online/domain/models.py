"""Online read models. Canonical perception metadata remains in datasection."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator

from online.domain.base import StrictModel  # noqa: F401 - re-export cho chỗ import cũ
from online.domain.candidate import (  # noqa: F401 - re-export: nhiều adapter import từ đây
    Candidate,
    EntityType,
    FrameEvidence,
    FrameQuality,
    Modality,
)
from online.domain.scores import BranchScore, ScoreKind  # noqa: F401 - re-export
from online.domain.search_config import SearchOptions
from online.domain.tasks import TaskType, normalize_task  # noqa: F401 - re-export


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
    # None = "để endpoint quyết định". Convenience endpoint điền task của path;
    # body khai báo task KHÁC path là lỗi tường minh (TaskConflictError), không
    # còn bị ghi đè im lặng như trước PR-01.
    task: TaskType | None = None
    top_k: int = Field(default=20, ge=1, le=200)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    debug: bool = False
    search_options: SearchOptions | None = None

    @field_validator("task", mode="before")
    @classmethod
    def accept_task_aliases(cls, value: object) -> object:
        """Chấp nhận alias (`kis`, `vqa`, `sequence`, ...) ở API boundary."""

        if value is None or isinstance(value, TaskType):
            return value
        return normalize_task(value)


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
    # Khoảng frame nửa mở [start_frame, end_frame_exclusive) — giữ đúng
    # convention của canonical `datasection.schemas.scene.Scene`.
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    event_id: str | None = None
    artifact_version: str | None = None

    keyframes: list[FrameEvidence] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    asr_texts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    action_tags: list[str] = Field(default_factory=list)
    color_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_frame_interval(self) -> "SceneDocument":
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("scene requires end_frame_exclusive > start_frame")
        for frame in self.keyframes:
            if not self.start_frame <= frame.frame_idx < self.end_frame_exclusive:
                raise ValueError(
                    f"keyframe {frame.keyframe_id} (frame_idx={frame.frame_idx}) is "
                    f"outside scene interval [{self.start_frame}, {self.end_frame_exclusive})"
                )
        return self

    # -- Derived views ----------------------------------------------------
    # Trước PR-01 ba list này được lưu song song và có thể lệch nhau. Giờ
    # chúng chỉ là view của `keyframes`, không thể desync.

    @property
    def keyframe_ids(self) -> list[str]:
        return [frame.keyframe_id for frame in self.keyframes]

    @property
    def keyframe_paths(self) -> list[str]:
        return [frame.image_path for frame in self.keyframes]

    @property
    def keyframe_timestamps(self) -> list[float]:
        return [frame.timestamp_sec for frame in self.keyframes]

    @property
    def frame_indices(self) -> list[int]:
        return [frame.frame_idx for frame in self.keyframes]

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


class Evidence(StrictModel):
    modality: Modality
    text: str
    score: float


class SearchHit(StrictModel):
    """Một kết quả đã hydrate, sẵn sàng để hiển thị và để build submission."""

    rank: int = Field(default=1, ge=1)
    candidate_id: str
    scene_id: str
    video_id: str
    video_path: str | None = None
    event_id: str | None = None
    scene_idx: int = Field(ge=0)

    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_sec: float
    end_sec: float

    # Tọa độ submission. Bắt buộc: KIS/QA nộp đúng cặp (video_id, frame_idx).
    best_frame_idx: int = Field(ge=0)
    best_keyframe_id: str | None = None
    best_keyframe_path: str | None = None
    best_timestamp_sec: float | None = None
    safe_frame_score: float | None = None

    score: float
    keyframes: list[FrameEvidence] = Field(default_factory=list)
    matched_modalities: list[Modality] = Field(default_factory=list)
    matched_branches: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    # Điểm gốc từng branch (không cùng thang đo giữa các branch — chỉ để debug).
    component_scores: dict[str, float] = Field(default_factory=dict)
    # Đóng góp thực tế của từng branch vào điểm fusion (cùng thang đo).
    branch_contributions: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def keyframe_ids(self) -> list[str]:
        return [frame.keyframe_id for frame in self.keyframes]

    @property
    def keyframe_paths(self) -> list[str]:
        return [frame.image_path for frame in self.keyframes]

    @property
    def keyframe_timestamps(self) -> list[float]:
        return [frame.timestamp_sec for frame in self.keyframes]


class SequenceHit(StrictModel):
    """Chuỗi scene theo đúng thứ tự cho TRAKE.

    `frame_ids` là thứ được nộp — mỗi step một frame. Nó được suy ra từ
    `scenes[*].best_frame_idx` nên không thể lệch với evidence hiển thị.
    """

    video_id: str
    score: float
    scenes: list[SearchHit] = Field(min_length=2)

    # computed_field (không phải property thường): `frame_ids` là thứ được nộp
    # nên nó phải nằm trong JSON response, kể cả khi FastAPI serialize qua
    # response_model.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def frame_ids(self) -> list[int]:
        return [scene.best_frame_idx for scene in self.scenes]


PipelineStatus = Literal["COMPLETED", "COMPLETED_WITH_WARNINGS"]


class SearchResponse(StrictModel):
    query_id: str
    task: TaskType
    took_ms: float
    status: PipelineStatus = "COMPLETED"
    results: list[SearchHit] = Field(default_factory=list)
    sequences: list[SequenceHit] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    warnings: list[str] = Field(default_factory=list)


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
