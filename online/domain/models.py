"""Online read models. Canonical perception metadata remains in datasection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, computed_field, field_validator, model_validator

from online.domain.base import StrictModel  # noqa: F401 - re-export cho chỗ import cũ
from online.domain.candidate import (  # noqa: F401 - re-export: nhiều adapter import từ đây
    Candidate,
    EntityType,
    FrameEvidence,
    FrameQuality,
    Modality,
)
from online.domain.execution import BranchStatus  # noqa: F401 - re-export
from online.domain.scores import BranchScore, ScoreKind  # noqa: F401 - re-export
from online.domain.search_config import SearchOptions
from online.domain.task_results import PlaybackWindow
from online.domain.task_results import (  # noqa: F401 - re-export
    AvsResultItem,
    KisResultItem,
    QaResultItem,
    TrakeResultItem,
)
from online.domain.tasks import TaskType, normalize_task  # noqa: F401 - re-export

if TYPE_CHECKING:
    # Avoid circular import - only used for type hints
    from online.services.query.models import SearchQueryBundle


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
    # Nhờ LLM đề xuất trọng số cho từng nhánh theo truy vấn này. Đề xuất đi
    # kèm response ở `recommended_weights`, KHÔNG được tự áp: trọng số đổi ngầm
    # giữa hai lần tìm thì không ai tái lập được kết quả.
    recommend_weights: bool = False
    # Nhờ LLM lọc bằng chứng thô xuống phần thật sự liên quan tới truy vấn.
    # Không bật thì `evidence` giữ nguyên bản gộp máy móc, trong đó logo đài và
    # đồng hồ trên màn hình đứng ngang hàng với nội dung cảnh.
    select_evidence: bool = False

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
    # Query Routing V2 — specialized queries per retrieval engine.
    # These are populated by QueryRouter.prepare() when using the new query prep.
    # Retriever adapters check these first, fall back to normalized_query if empty.
    visual_query: str = ""
    visual_query_en: str = ""
    caption_query: str = ""
    ocr_query: str = ""
    asr_query: str = ""
    # Classification metadata from query routing
    query_intent: str = ""
    answer_type: str = ""

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

    #: Đoạn video cần phát, ĐÃ nới bối cảnh. `None` = chưa có file video nguồn
    #: (khác với "phát lỗi"). UI đọc trường này thay vì tự tính lại phần nới.
    playback: "PlaybackWindow | None" = None
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
    """Chuỗi scene theo đúng thứ tự cho TRAKE, CÓ THỂ thiếu step.

    `frame_ids` là thứ được nộp — nó được suy ra từ `scenes[*].best_frame_idx`
    nên không thể lệch với evidence hiển thị.

    **Chuỗi thiếu step là công dân hạng nhất, không phải trường hợp lỗi.** Đo
    trên corpus 873 video: chỉ 14/24 truy vấn có đủ candidate cho MỌI step, nên
    đòi chuỗi đầy đủ là vứt 10/24 truy vấn về không — kể cả những truy vấn mà
    video đúng đang nằm sẵn trong pool ở 4/5 step. Một chuỗi bắt được 2/5 step
    vẫn chỉ đúng video là đủ để người dùng xem và chốt.

    `covered_steps` là số thứ tự step (1-based) của từng phần tử trong `scenes`,
    cùng độ dài với `scenes`. Không có nó thì lỗ thủng ở GIỮA chuỗi không phân
    biệt được với lỗ thủng ở ĐUÔI, và mọi thứ hạ nguồn sẽ gán nhầm frame cho
    step — sai lặng lẽ, đúng loại lỗi tệ nhất.
    """

    video_id: str
    score: float
    scenes: list[SearchHit] = Field(min_length=1)
    covered_steps: list[int] = Field(default_factory=list)
    total_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _default_and_check_step_mapping(self) -> "SequenceHit":
        # Chuỗi đầy đủ dựng trước khi có field này vẫn hợp lệ: suy ra 1..N.
        if not self.covered_steps:
            object.__setattr__(self, "covered_steps", list(range(1, len(self.scenes) + 1)))
        if len(self.covered_steps) != len(self.scenes):
            raise ValueError(
                f"covered_steps ({len(self.covered_steps)}) phải cùng độ dài với "
                f"scenes ({len(self.scenes)})"
            )
        if any(
            later <= earlier
            for earlier, later in zip(self.covered_steps, self.covered_steps[1:])
        ):
            raise ValueError(f"covered_steps phải tăng dần, nhận {self.covered_steps}")
        if not self.total_steps:
            object.__setattr__(self, "total_steps", max(self.covered_steps, default=0))
        return self

    @property
    def missing_steps(self) -> list[int]:
        covered = set(self.covered_steps)
        return [step for step in range(1, self.total_steps + 1) if step not in covered]

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
    # Kết quả theo đúng đơn vị nộp bài của từng task (PR-07). `results` vẫn là
    # danh sách retrieval chung để UI hiển thị; bốn field dưới mới là thứ đi
    # thẳng vào submission.
    kis: list[KisResultItem] = Field(default_factory=list)
    qa: list[QaResultItem] = Field(default_factory=list)
    trake: list[TrakeResultItem] = Field(default_factory=list)
    avs: list[AvsResultItem] = Field(default_factory=list)
    # PR-09: có giá trị khi response này là kết quả của POST
    # /v1/search-sessions/{id}/replay — trỏ về session gốc.
    replayed_from: str | None = None
    results: list[SearchHit] = Field(default_factory=list)
    sequences: list[SequenceHit] = Field(default_factory=list)
    # Trạng thái từng branch: UI phải thấy được branch nào timeout/lỗi thay vì
    # nhận một danh sách ngắn đi mà không biết vì sao (PR-03).
    branch_status: list[BranchStatus] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    warnings: list[str] = Field(default_factory=list)
    # Chỉ có khi request đặt `recommend_weights=true`. Là ĐỀ XUẤT để người dùng
    # xem rồi tự quyết, không phải trọng số đã dùng cho lần tìm này.
    recommended_weights: dict | None = None
    # Chỉ có khi request đặt `select_evidence=true`. Cùng thứ tự với `results`.
    selected_evidence: list[dict] = Field(default_factory=list)
    # AVS-GRADE-01: đếm candidate trước/sau cổng grade + danh sách bị loại.
    avs_diagnostics: dict | None = None
    # P2: dấu vết TỪNG TẦNG, chỉ khi `debug=true`. Không có nó thì eval chỉ
    # thấy điểm cuối và không biết candidate đúng rơi ở tầng nào —
    # "recall đủ nhưng xếp sai" và "không tìm ra" cần hai cách sửa khác hẳn.
    pipeline_trace: dict | None = None


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
