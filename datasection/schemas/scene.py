"""Canonical scene metadata contract for AIC 2026.

A scene is the aggregate root returned by most retrieval operations. It embeds
its complete keyframe children and owns scene-level outputs such as temporal
boundaries, ASR projections, summaries, keywords, and aggregate embeddings.

Frame and time intervals are half-open: ``[start, end_exclusive)``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    ASRSourceId,
    EmbeddingReference,
    KeyframeId,
    ModelProvenance,
    NonEmptyStr,
    Probability,
    RelativeArtifactPath,
    SHA256Checksum,
    SceneId,
    SceneASRSegmentId,
    StrictModel,
    VideoId,
    utc_now,
)
from .keyframe import Keyframe


class TransitionType(StrEnum):
    """Visual transition observed at a scene boundary."""

    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    UNKNOWN = "unknown"


class SceneCaptionRecord(StrictModel):
    """One generated textual description of the entire scene."""

    language: NonEmptyStr = "en"
    caption_type: Literal["visual", "audio_visual", "summary", "tags"]
    text: NonEmptyStr
    confidence: Probability | None = None
    evidence_keyframe_ids: list[KeyframeId] = Field(default_factory=list)
    provenance: ModelProvenance

    @field_validator("evidence_keyframe_ids")
    @classmethod
    def require_unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_keyframe_ids must not contain duplicates")
        return values


class ASRSegment(StrictModel):
    """Scene-clipped projection of an ASR segment on the video timeline."""

    segment_id: SceneASRSegmentId
    source_segment_id: ASRSourceId
    start_sec: Annotated[float, Field(ge=0.0)]
    end_sec: Annotated[float, Field(gt=0.0)]
    text: NonEmptyStr
    normalized_text: str | None = None
    language: str | None = None
    confidence: Probability | None = None
    speaker_id: str | None = None
    provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_interval(self) -> ASRSegment:
        if self.end_sec <= self.start_sec:
            raise ValueError("ASR segment requires end_sec > start_sec")
        return self


class SceneKeyword(StrictModel):
    """Searchable keyword with its origin retained for weighting and auditing."""

    text: NonEmptyStr
    normalized_text: NonEmptyStr
    language: str | None = None
    sources: list[Literal["caption", "ocr", "asr", "object", "manual"]] = Field(
        min_length=1
    )
    confidence: Probability | None = None
    provenance: ModelProvenance | None = None

    @field_validator("sources")
    @classmethod
    def require_unique_sources(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("keyword sources must not contain duplicates")
        return values

    @model_validator(mode="after")
    def require_provenance_for_automatic_keywords(self) -> SceneKeyword:
        if any(source != "manual" for source in self.sources) and not self.provenance:
            raise ValueError("automatically derived keywords require provenance")
        return self


class Scene(StrictModel):
    """Complete searchable scene with nested keyframes and temporal evidence."""

    schema_version: Literal["1.0.0"] = "1.0.0"

    scene_id: SceneId
    video_id: VideoId
    scene_idx: Annotated[int, Field(ge=0, le=9_999)]

    start_frame: Annotated[int, Field(ge=0)]
    end_frame_exclusive: Annotated[int, Field(gt=0)]
    start_sec: Annotated[float, Field(ge=0.0)]
    end_sec: Annotated[float, Field(gt=0.0)]

    transition_in: TransitionType = TransitionType.UNKNOWN
    transition_out: TransitionType = TransitionType.UNKNOWN
    boundary_confidence_in: Probability | None = None
    boundary_confidence_out: Probability | None = None
    segmentation_provenance: ModelProvenance

    keyframes: list[Keyframe] = Field(min_length=1)
    captions: list[SceneCaptionRecord] = Field(default_factory=list)
    asr_segments: list[ASRSegment] = Field(default_factory=list)
    keywords: list[SceneKeyword] = Field(default_factory=list)
    embedding_refs: list[EmbeddingReference] = Field(default_factory=list)

    scene_clip_path: RelativeArtifactPath | None = None
    scene_clip_checksum: SHA256Checksum | None = None
    created_at: datetime = Field(default_factory=utc_now)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def require_created_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value

    @model_validator(mode="after")
    def validate_scene_consistency(self) -> Scene:
        expected_scene_id = f"{self.video_id}_S{self.scene_idx:04d}"
        if self.scene_id != expected_scene_id:
            raise ValueError(
                f"scene_id must equal {expected_scene_id} for video/scene_idx"
            )
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("scene requires end_frame_exclusive > start_frame")
        if self.end_sec <= self.start_sec:
            raise ValueError("scene requires end_sec > start_sec")

        keyframe_ids: list[str] = []
        keyframe_indices: list[int] = []
        for keyframe in self.keyframes:
            if keyframe.video_id != self.video_id:
                raise ValueError(
                    f"keyframe {keyframe.keyframe_id} belongs to another video"
                )
            if keyframe.scene_id != self.scene_id:
                raise ValueError(
                    f"keyframe {keyframe.keyframe_id} belongs to another scene"
                )
            if not self.start_frame <= keyframe.frame_idx < self.end_frame_exclusive:
                raise ValueError(
                    f"keyframe {keyframe.keyframe_id} is outside scene frame interval"
                )
            if not self.start_sec <= keyframe.timestamp_sec < self.end_sec:
                raise ValueError(
                    f"keyframe {keyframe.keyframe_id} is outside scene time interval"
                )
            keyframe_ids.append(keyframe.keyframe_id)
            keyframe_indices.append(keyframe.frame_idx)

        if len(keyframe_ids) != len(set(keyframe_ids)):
            raise ValueError("scene keyframe_id values must be unique")
        if len(keyframe_indices) != len(set(keyframe_indices)):
            raise ValueError("scene keyframe frame_idx values must be unique")
        if keyframe_indices != sorted(keyframe_indices):
            raise ValueError("scene keyframes must be ordered by frame_idx")

        known_keyframe_ids = set(keyframe_ids)
        for caption in self.captions:
            unknown_ids = set(caption.evidence_keyframe_ids) - known_keyframe_ids
            if unknown_ids:
                raise ValueError(
                    "scene caption references unknown keyframes: "
                    + ", ".join(sorted(unknown_ids))
                )

        segment_ids: list[str] = []
        for segment in self.asr_segments:
            if not segment.segment_id.startswith(f"{self.scene_id}_A"):
                raise ValueError(
                    f"ASR segment {segment.segment_id} belongs to another scene"
                )
            if not segment.source_segment_id.startswith(f"{self.video_id}_ASR"):
                raise ValueError(
                    f"ASR source {segment.source_segment_id} belongs to another video"
                )
            if segment.start_sec < self.start_sec or segment.end_sec > self.end_sec:
                raise ValueError(
                    f"ASR segment {segment.segment_id} is outside scene time interval"
                )
            segment_ids.append(segment.segment_id)
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("ASR segment_id values must be unique within a scene")
        segment_starts = [item.start_sec for item in self.asr_segments]
        if segment_starts != sorted(segment_starts):
            raise ValueError("ASR segments must be ordered by start_sec")

        keyword_keys = [
            (item.normalized_text, item.language) for item in self.keywords
        ]
        if len(keyword_keys) != len(set(keyword_keys)):
            raise ValueError(
                "normalized keyword text must be unique per language within a scene"
            )

        if self.scene_clip_checksum and not self.scene_clip_path:
            raise ValueError("scene_clip_checksum requires scene_clip_path")

        embedding_names = [item.embedding_name for item in self.embedding_refs]
        if len(embedding_names) != len(set(embedding_names)):
            raise ValueError("embedding_name must be unique within a scene")
        return self

    @property
    def duration_frames(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def ocr_text(self) -> str:
        """Derive OCR text from children without storing a second source of truth."""

        return " ".join(
            text
            for keyframe in self.keyframes
            if (text := keyframe.ocr_text)
        )

    @property
    def asr_text(self) -> str:
        """Return projected ASR text in timeline order."""

        return " ".join(
            segment.text
            for segment in sorted(self.asr_segments, key=lambda item: item.start_sec)
        )

    def qdrant_payload(self) -> dict[str, Any]:
        """Build a compact, filterable scene payload for online retrieval."""

        return {
            "entity_type": "scene",
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "video_id": self.video_id,
            "scene_idx": self.scene_idx,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "keyframe_ids": [item.keyframe_id for item in self.keyframes],
            "keyframe_count": len(self.keyframes),
            "has_ocr": bool(self.ocr_text),
            "has_asr": bool(self.asr_segments),
            "has_caption": bool(self.captions),
            "keywords": sorted({item.normalized_text for item in self.keywords}),
        }


__all__ = [
    "ASRSegment",
    "Scene",
    "SceneCaptionRecord",
    "SceneKeyword",
    "TransitionType",
]
