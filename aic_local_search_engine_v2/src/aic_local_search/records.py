from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SceneDocument:
    scene_id: str
    video_id: str
    scene_no: int
    start_sec: float
    end_sec: float
    clip_path: str
    start_frame: int = 0
    end_frame: int = 0
    representative_keyframe_id: str = ""
    vector_row: int = -1
    ocr_text: str = ""
    transcript: str = ""
    caption_vi: str = ""
    caption_en: str = ""
    speech_summary: str = ""
    scene_type: str = "other"
    visible_text: str = ""
    keywords: str = ""
    entities: str = ""
    actions: str = ""
    attributes: str = ""
    relations: str = ""
    event_text: str = ""
    temporal_events: list[dict[str, Any]] = field(default_factory=list)
    semantic_status: str = "missing"
    quality_status: str = "passed"
    quality_penalty: float = 1.0
    quality_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KeyframeDocument:
    keyframe_id: str
    scene_id: str
    frame_idx: int
    timestamp_sec: float
    image_path: str
    vector_row: int
    quality_score: float = 0.0
    ocr_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedComponents:
    scenes: list[SceneDocument]
    keyframes: list[KeyframeDocument]
    scene_embeddings: Any
    keyframe_embeddings: Any | None
    scene_embedding_model: str
    keyframe_embedding_model: str
    embedding_dimension: int
    source_root: Path
    stats: dict[str, Any]
    warnings: list[str]


@dataclass(slots=True)
class BuildReport:
    index_dir: str
    database_path: str
    vector_backend: str
    embedding_dimension: int
    scene_count: int
    keyframe_count: int
    video_count: int
    warnings: list[str]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
