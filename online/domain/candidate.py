"""Frame-anchored retrieval contract (PR-01).

Đây là thay đổi nền của PR-01: trước đó toàn bộ tầng online xoay quanh
`scene`, và `project_scene()` làm mất `frame_idx` dù canonical
`datasection.schemas.Keyframe` đã có sẵn. Kết quả là không thể xuất
submission `<video_id>, <frame_idx>` mà không join ngược ra datasection.

Từ đây:

* `FrameEvidence` là đơn vị nhỏ nhất, luôn mang `frame_idx` — tọa độ
  submission chuẩn (timestamp chỉ để debug/hiển thị).
* `Candidate` khai báo rõ mình neo ở entity nào (`entity_type`) và luôn
  truy ngược được về video + model/index đã sinh ra nó.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from online.domain.base import StrictModel
from online.domain.scores import BranchScore, ScoreKind

EntityType = Literal["frame", "scene", "clip", "event", "video"]


class Modality(StrEnum):
    VISUAL = "visual"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"
    KEYWORD = "keyword"
    OBJECT = "object"
    ACTION = "action"
    COLOR = "color"
    EVENT = "event"


class FrameQuality(StrictModel):
    """Tín hiệu chất lượng frame, tên giữ nguyên theo canonical
    `datasection.schemas.keyframe.QualitySignals` để không đổi nghĩa khi
    chiếu sang online. Safe-frame selection (PR-07) đọc trực tiếp ở đây."""

    sharpness: float | None = Field(default=None, ge=0.0)
    brightness: float | None = Field(default=None, ge=0.0, le=1.0)
    contrast: float | None = Field(default=None, ge=0.0, le=1.0)
    black_frame_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    duplicate_score: float | None = Field(default=None, ge=0.0, le=1.0)


class FrameEvidence(StrictModel):
    """Một frame tìm kiếm được, đầy đủ để chấm safe-frame và để submit.

    Thay cho ba list song song `keyframe_ids` / `keyframe_paths` /
    `keyframe_timestamps` của `SceneDocument` cũ — ba list đó không có
    `frame_idx` và dễ lệch index với nhau.
    """

    keyframe_id: str
    video_id: str
    scene_id: str

    frame_idx: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    image_path: str

    selection_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality: FrameQuality = Field(default_factory=FrameQuality)
    # Khoảng cách (số frame) tới biên scene gần nhất. Frame sát biên hay dính
    # transition/cut nên bị phạt khi chọn safe-frame.
    boundary_distance_frames: int | None = Field(default=None, ge=0)

    captions: list[str] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    # bbox của TỪNG chuỗi trong `ocr_texts`, cùng thứ tự và cùng độ dài.
    # `None` cho chuỗi không biết vị trí (phần bù OCR bằng prompt text-only).
    #
    # Trường này tồn tại vì bộ lọc lớp phủ (`AIC_OCR_OVERLAY_DF`) cần vị trí,
    # mà trước đây nó đi tìm `keyframe.ocr_instances` — thuộc tính CHỈ có trên
    # canonical Keyframe, không có trên projection này. `getattr` trả `[]`, rồi
    # `zip(texts, [])` cho ra rỗng, nên bộ lọc xoá SẠCH text OCR của cả 765
    # scene thay vì bỏ riêng phần lớp phủ. Đo được lúc phát hiện:
    #   có lọc  ->   0/765 scene còn chữ OCR (nhánh bm25_ocr chết hẳn)
    #   ko lọc  -> 674/765 scene, 12490 từ
    ocr_boxes: list[dict[str, float] | None] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    action_tags: list[str] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)
    # Tên các embedding đã sinh cho frame này (vd "openclip_l14"). Vector thật
    # nằm trong vector store; ở đây chỉ giữ tên để biết frame có mặt ở index nào.
    embedding_names: list[str] = Field(default_factory=list)

    @property
    def search_text(self) -> str:
        """Toàn bộ text của frame, dùng để chấm độ khớp query ở mức frame."""

        return " ".join(
            [*self.captions, *self.ocr_texts, *self.object_labels, *self.action_tags]
        )


class Candidate(StrictModel):
    """Một kết quả từ đúng một branch (hoặc từ fusion, `source="fusion_*"`).

    `raw_score` là điểm trong không gian gốc của branch (BM25 / cosine /
    tỷ lệ overlap...). Không bao giờ cộng trực tiếp `raw_score` giữa các
    branch — xem `online/services/fusion.py`.
    """

    candidate_id: str
    entity_type: EntityType = "scene"

    video_id: str
    scene_id: str | None = None
    clip_id: str | None = None
    event_id: str | None = None

    frame_idx: int | None = Field(default=None, ge=0)
    timestamp_sec: float | None = Field(default=None, ge=0.0)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)

    source: str
    modality: Modality

    raw_score: float
    score_kind: ScoreKind = "cosine"
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    percentile_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rank: int = Field(ge=1)

    model_id: str | None = None
    index_id: str | None = None
    branch_scores: dict[str, BranchScore] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_anchor(self) -> "Candidate":
        if self.entity_type == "frame" and self.frame_idx is None:
            raise ValueError("frame candidates require frame_idx")
        if self.entity_type == "scene" and not self.scene_id:
            raise ValueError("scene candidates require scene_id")
        if self.entity_type == "event" and not self.event_id and not self.scene_id:
            raise ValueError("event candidates require event_id or scene_id")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame < self.start_frame
        ):
            raise ValueError("end_frame must be >= start_frame")
        return self

    @property
    def grouping_key(self) -> str:
        """Khóa gom candidate ở tầng fusion.

        Mặc định gom theo scene (hành vi từ trước tới nay). Candidate không
        neo scene — vd frame rời hoặc event-level — gom theo chính id của nó
        thay vì bị bỏ rơi. Dedup scope thật do PR-05 xử lý.
        """

        return self.scene_id or self.candidate_id


__all__ = [
    "Candidate",
    "EntityType",
    "FrameEvidence",
    "FrameQuality",
    "Modality",
]
