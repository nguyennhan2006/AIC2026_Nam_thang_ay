"""Canonical keyframe metadata contract for AIC 2026.

This module contains data definitions only. It deliberately does not run OCR,
captioning, object detection, or vector search. Model outputs are attached to a
keyframe together with provenance, while large embedding arrays remain in a
dedicated vector store and are represented here by stable references.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    ArtifactURI,
    BoundingBox,
    EmbeddingReference,
    KeyframeId,
    ModelProvenance,
    NonEmptyStr,
    Probability,
    RelativeArtifactPath,
    SHA256Checksum,
    SceneId,
    StrictModel,
    VectorLocation,
    VideoId,
    utc_now,
)


class KeyframeRole(StrEnum):
    """Reason why a frame was selected from its parent scene."""

    REPRESENTATIVE = "representative"
    MIDDLE = "middle"
    BOUNDARY_START = "boundary_start"
    BOUNDARY_END = "boundary_end"
    OCR_RICH = "ocr_rich"
    MOTION_CHANGE = "motion_change"
    MANUAL = "manual"


class CaptionRecord(StrictModel):
    """One model-generated textual view of the keyframe."""

    language: NonEmptyStr = "en"
    caption_type: Literal["short", "detailed", "tags", "crop"]
    text: NonEmptyStr
    confidence: Probability | None = None
    crop_bbox: BoundingBox | None = None
    provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_crop_caption(self) -> CaptionRecord:
        if self.caption_type == "crop" and self.crop_bbox is None:
            raise ValueError("crop captions require crop_bbox")
        if self.caption_type != "crop" and self.crop_bbox is not None:
            raise ValueError("crop_bbox is only valid for crop captions")
        return self


class OCRInstance(StrictModel):
    """A recognized text instance and its exact location in the keyframe."""

    text: NonEmptyStr
    normalized_text: str | None = None
    language: str | None = None
    confidence: Probability
    bbox: BoundingBox
    provenance: ModelProvenance


class ObjectInstance(StrictModel):
    """A detected object used as soft search evidence, not a hard truth."""

    label: NonEmptyStr
    confidence: Probability
    bbox: BoundingBox
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    provenance: ModelProvenance


class ColorFeature(StrictModel):
    """Compact color metadata suitable for filtering and result explanation."""

    dominant_hex: list[str] = Field(default_factory=list, max_length=8)
    mean_hsv: tuple[
        Annotated[float, Field(ge=0.0, le=360.0)],
        Annotated[float, Field(ge=0.0, le=1.0)],
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] | None = None
    histogram_uri: ArtifactURI | None = None
    provenance: ModelProvenance | None = None

    @field_validator("dominant_hex")
    @classmethod
    def validate_hex_colors(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            color = value.upper()
            if len(color) != 7 or not color.startswith("#"):
                raise ValueError(f"invalid HEX color: {value}")
            try:
                int(color[1:], 16)
            except ValueError as exc:
                raise ValueError(f"invalid HEX color: {value}") from exc
            normalized.append(color)
        return normalized


class QualitySignals(StrictModel):
    """Optional signals used to choose or down-rank weak keyframes."""

    sharpness: Annotated[float, Field(ge=0.0)] | None = None
    brightness: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    contrast: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    black_frame_ratio: Probability | None = None
    duplicate_score: Probability | None = None


class Keyframe(StrictModel):
    """Complete metadata for one searchable frame inside a scene."""

    schema_version: Literal["1.0.0"] = "1.0.0"

    keyframe_id: KeyframeId
    video_id: VideoId
    scene_id: SceneId

    frame_idx: Annotated[int, Field(ge=0, le=999_999)]
    timestamp_sec: Annotated[float, Field(ge=0.0)]
    image_path: RelativeArtifactPath
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]

    roles: list[KeyframeRole] = Field(min_length=1)
    selection_score: Probability | None = None
    quality: QualitySignals = Field(default_factory=QualitySignals)

    captions: list[CaptionRecord] = Field(default_factory=list)
    ocr_instances: list[OCRInstance] = Field(default_factory=list)
    objects: list[ObjectInstance] = Field(default_factory=list)
    color: ColorFeature | None = None
    embedding_refs: list[EmbeddingReference] = Field(default_factory=list)

    source_checksum: SHA256Checksum | None = None
    created_at: datetime = Field(default_factory=utc_now)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def require_unique_roles(cls, values: list[KeyframeRole]) -> list[KeyframeRole]:
        if len(values) != len(set(values)):
            raise ValueError("roles must not contain duplicates")
        return values

    @field_validator("created_at")
    @classmethod
    def require_created_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value

    @model_validator(mode="after")
    def validate_identity_and_embeddings(self) -> Keyframe:
        if self.scene_id != f"{self.video_id}_{self.scene_id.rsplit('_', 1)[-1]}":
            raise ValueError("scene_id must belong to video_id")
        expected_keyframe_id = f"{self.scene_id}_F{self.frame_idx:06d}"
        if self.keyframe_id != expected_keyframe_id:
            raise ValueError(
                f"keyframe_id must equal {expected_keyframe_id} for this scene/frame"
            )
        names = [item.embedding_name for item in self.embedding_refs]
        if len(names) != len(set(names)):
            raise ValueError("embedding_name must be unique within a keyframe")
        return self

    @property
    def ocr_text(self) -> str:
        """Return OCR text in instance order for display or later aggregation."""

        return " ".join(item.text for item in self.ocr_instances)

    def qdrant_payload(self) -> dict[str, Any]:
        """Build a compact filterable payload without embedding large content.

        Full captions and OCR bounding boxes remain in the canonical metadata
        store. Qdrant receives only fields useful for filtering, tracing, and
        returning an initial search result.
        """

        return {
            "entity_type": "keyframe",
            "schema_version": self.schema_version,
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "scene_id": self.scene_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "image_path": self.image_path,
            "roles": [role.value for role in self.roles],
            "has_ocr": bool(self.ocr_instances),
            "has_caption": bool(self.captions),
            "object_labels": sorted({item.label for item in self.objects}),
            "pipeline_versions": sorted(
                {
                    record.provenance.pipeline_version
                    for record in [*self.captions, *self.ocr_instances, *self.objects]
                }
            ),
        }


__all__ = [
    "CaptionRecord",
    "ColorFeature",
    "Keyframe",
    "KeyframeRole",
    "ObjectInstance",
    "OCRInstance",
    "QualitySignals",
]
