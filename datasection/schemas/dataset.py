"""Version manifest that makes an exported dataset reproducible and auditable."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator

from .common import ArtifactURI, NonEmptyStr, SHA256Checksum, StrictModel, utc_now


class ModelArtifact(StrictModel):
    task: Literal["scene", "keyframe", "caption", "ocr", "object", "asr", "embedding", "reranker", "color"]
    model_name: NonEmptyStr
    revision: NonEmptyStr
    config_checksum: SHA256Checksum | None = None


class IndexArtifact(StrictModel):
    backend: Literal["faiss", "qdrant", "bm25", "file"]
    name: NonEmptyStr
    entity: Literal["scene", "keyframe"]
    vector_name: str | None = None
    dimension: Annotated[int, Field(gt=0)] | None = None
    location: ArtifactURI
    checksum: SHA256Checksum | None = None


class DatasetManifest(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    dataset_id: NonEmptyStr
    build_id: NonEmptyStr
    pipeline_version: NonEmptyStr
    video_count: Annotated[int, Field(ge=0)]
    scene_count: Annotated[int, Field(ge=0)]
    keyframe_count: Annotated[int, Field(ge=0)]
    # Optional/mặc định 0 để tương thích ngược với manifest cũ chưa có clip (seed_demo,
    # export không chạy clip pooling) — không đổi hành vi mặc định.
    clip_count: Annotated[int, Field(ge=0)] = 0
    models: list[ModelArtifact] = Field(default_factory=list)
    indexes: list[IndexArtifact] = Field(default_factory=list)
    export_checksums: dict[str, SHA256Checksum] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value
