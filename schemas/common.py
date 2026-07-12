"""Shared schema primitives and repository-wide metadata conventions.

ID hierarchy (fixed-width and case-sensitive):
    video    L01_V001
    scene    L01_V001_S0003
    keyframe L01_V001_S0003_F001234

Media paths are POSIX paths relative to ``AIC_DATA_ROOT``. Absolute paths and
``..`` traversal are rejected. Checksums use ``sha256:<64 lowercase hex>``.
Artifact URIs may be relative paths or use an explicitly allowed URI scheme.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


VIDEO_ID_PATTERN = r"^L[0-9]{2}_V[0-9]{3}$"
SCENE_ID_PATTERN = r"^L[0-9]{2}_V[0-9]{3}_S[0-9]{4}$"
KEYFRAME_ID_PATTERN = r"^L[0-9]{2}_V[0-9]{3}_S[0-9]{4}_F[0-9]{6}$"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
ALLOWED_ARTIFACT_URI_SCHEMES = frozenset(
    {"az", "file", "gs", "https", "qdrant", "s3"}
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
VideoId = Annotated[str, Field(pattern=VIDEO_ID_PATTERN)]
SceneId = Annotated[str, Field(pattern=SCENE_ID_PATTERN)]
KeyframeId = Annotated[str, Field(pattern=KEYFRAME_ID_PATTERN)]
SHA256Checksum = Annotated[str, Field(pattern=SHA256_PATTERN)]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def normalize_relative_artifact_path(value: str) -> str:
    """Normalize and validate a path relative to ``AIC_DATA_ROOT``."""

    candidate = value.strip().replace("\\", "/")
    path = PurePosixPath(candidate)
    if not candidate or candidate == ".":
        raise ValueError("artifact path must not be empty")
    if path.is_absolute():
        raise ValueError("artifact path must be relative to AIC_DATA_ROOT")
    if ".." in path.parts:
        raise ValueError("artifact path must not contain '..' traversal")
    if "://" in candidate:
        raise ValueError("artifact path must not be a URI")
    return str(path)


def normalize_artifact_uri(value: str) -> str:
    """Validate a supported URI or normalize a relative artifact path."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("artifact URI must not be empty")
    if re.search(r"\s", candidate):
        raise ValueError("artifact URI must not contain whitespace")
    if "://" not in candidate:
        return normalize_relative_artifact_path(candidate)

    parsed = urlsplit(candidate)
    if parsed.scheme not in ALLOWED_ARTIFACT_URI_SCHEMES:
        raise ValueError(f"unsupported artifact URI scheme: {parsed.scheme}")
    if not parsed.netloc and not parsed.path:
        raise ValueError("artifact URI must contain a location")
    return candidate


RelativeArtifactPath = Annotated[str, AfterValidator(normalize_relative_artifact_path)]
ArtifactURI = Annotated[str, AfterValidator(normalize_artifact_uri)]


class StrictModel(BaseModel):
    """Strict base model shared by all metadata entities."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class BoundingBox(StrictModel):
    """Normalized XYXY box, independent of image resolution."""

    x1: Annotated[float, Field(ge=0.0, le=1.0)]
    y1: Annotated[float, Field(ge=0.0, le=1.0)]
    x2: Annotated[float, Field(ge=0.0, le=1.0)]
    y2: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_corner_order(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox requires x2 > x1 and y2 > y1")
        return self


class ModelProvenance(StrictModel):
    """Information required to reproduce a generated model output."""

    model_name: NonEmptyStr
    model_revision: str | None = None
    pipeline_version: NonEmptyStr
    prompt_version: str | None = None
    device: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value


class VectorLocation(StrictModel):
    """One physical location of a vector in FAISS, Qdrant, or a file."""

    backend: Literal["faiss", "qdrant", "file"]
    vector_id: NonEmptyStr
    index_name: NonEmptyStr
    vector_uri: ArtifactURI | None = None

    @model_validator(mode="after")
    def validate_backend_fields(self) -> VectorLocation:
        if self.backend == "file" and not self.vector_uri:
            raise ValueError("file embedding references require vector_uri")
        return self


class EmbeddingReference(StrictModel):
    """Logical embedding and every backend in which it is available."""

    embedding_name: NonEmptyStr
    modality: Literal["image", "caption", "ocr", "multimodal"]
    model_name: NonEmptyStr
    model_revision: str | None = None
    dimension: Annotated[int, Field(gt=0)]
    normalized: bool = True
    storage_locations: list[VectorLocation] = Field(min_length=1)

    @field_validator("storage_locations")
    @classmethod
    def require_unique_locations(
        cls, values: list[VectorLocation]
    ) -> list[VectorLocation]:
        keys = [(item.backend, item.index_name) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "an embedding cannot have duplicate backend/index locations"
            )
        return values


__all__ = [
    "ALLOWED_ARTIFACT_URI_SCHEMES",
    "ArtifactURI",
    "BoundingBox",
    "EmbeddingReference",
    "KEYFRAME_ID_PATTERN",
    "KeyframeId",
    "ModelProvenance",
    "NonEmptyStr",
    "Probability",
    "RelativeArtifactPath",
    "SCENE_ID_PATTERN",
    "SHA256Checksum",
    "SceneId",
    "StrictModel",
    "VIDEO_ID_PATTERN",
    "VectorLocation",
    "VideoId",
    "normalize_artifact_uri",
    "normalize_relative_artifact_path",
    "utc_now",
]
