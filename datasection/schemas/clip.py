"""Canonical clip-level metadata contract — Search Mixing Console W1 (clip pooling).

A clip is a short sliding-time-window aggregate INSIDE one scene (finer-grained
than the scene itself, coarser than a single keyframe) — used for
`dense_visual_clip`/action/temporal search branches. Clip and scene have
different lifecycle/index needs (a scene re-segmentation should not force every
clip to be rebuilt), so clips are **not** nested inside `Scene` — they live as a
sibling collection on `Video`, referencing `scene_id` like keyframes do.

Baseline V1 pools embeddings of keyframes ALREADY produced by the scene
pipeline (no dedicated clip frame extraction) — see
`offline/clip_pooling.py`. With sparse keyframe density
(`AIC_KEYFRAMES_PER_SCENE=1` default), many clips degenerate to a single
pooled frame; that is valid baseline behavior, not an error (see
docs/15_RESEARCH_AGENDA.md "Clip embedding density").
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import (
    ClipId,
    EmbeddingReference,
    KeyframeId,
    ModelProvenance,
    SceneId,
    StrictModel,
    VideoId,
)


class ClipSegment(StrictModel):
    """One pooled clip window inside a scene."""

    schema_version: Literal["1.0.0"] = "1.0.0"

    clip_id: ClipId
    video_id: VideoId
    scene_id: SceneId

    start_sec: Annotated[float, Field(ge=0.0)]
    end_sec: Annotated[float, Field(gt=0.0)]
    duration_sec: Annotated[float, Field(gt=0.0)]

    start_frame: Annotated[int, Field(ge=0)]
    # Half-open like Scene.end_frame_exclusive: frame at end_frame is NOT included.
    end_frame: Annotated[int, Field(gt=0)]

    sampled_frame_ids: list[KeyframeId] = Field(min_length=1)
    representative_frame_id: KeyframeId

    sampling_method: Literal["in_window", "nearest_scene_keyframe"]
    sampling_degraded: bool = False
    fallback_distance_sec: Annotated[float, Field(ge=0.0)] | None = None

    embedding_refs: list[EmbeddingReference] = Field(default_factory=list)

    caption: str | None = None
    action_tags: list[str] = Field(default_factory=list)

    clip_config_id: str = Field(min_length=1)
    provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_identity(self) -> "ClipSegment":
        if not self.scene_id.startswith(f"{self.video_id}_"):
            raise ValueError("scene_id must belong to video_id")
        if not self.clip_id.startswith(f"{self.scene_id}_C"):
            raise ValueError("clip_id must be derived from scene_id")
        expected_clip_id = f"{self.scene_id}_C{self.start_frame:08d}_{self.end_frame:08d}"
        if self.clip_id != expected_clip_id:
            raise ValueError(f"clip_id must equal {expected_clip_id} for this frame range")
        return self

    @model_validator(mode="after")
    def validate_interval(self) -> "ClipSegment":
        if self.end_sec <= self.start_sec:
            raise ValueError("clip requires end_sec > start_sec")
        if self.end_frame <= self.start_frame:
            raise ValueError("clip requires end_frame > start_frame")
        tolerance = 0.05
        if abs(self.duration_sec - (self.end_sec - self.start_sec)) > tolerance:
            raise ValueError("duration_sec must equal end_sec - start_sec")
        return self

    @model_validator(mode="after")
    def validate_sampling(self) -> "ClipSegment":
        if self.representative_frame_id not in self.sampled_frame_ids:
            raise ValueError("representative_frame_id must be one of sampled_frame_ids")
        if len(self.sampled_frame_ids) != len(set(self.sampled_frame_ids)):
            raise ValueError("sampled_frame_ids must not contain duplicates")
        if self.sampling_method == "in_window":
            if self.sampling_degraded or self.fallback_distance_sec is not None:
                raise ValueError("in_window sampling must not be marked degraded")
        else:  # nearest_scene_keyframe
            if not self.sampling_degraded or self.fallback_distance_sec is None:
                raise ValueError("nearest_scene_keyframe sampling must be marked degraded with a distance")
        return self

    @model_validator(mode="after")
    def validate_embedding_refs_unique(self) -> "ClipSegment":
        names = [item.embedding_name for item in self.embedding_refs]
        if len(names) != len(set(names)):
            raise ValueError("embedding_name must be unique within a clip")
        if len(self.action_tags) != len(set(self.action_tags)):
            raise ValueError("action_tags must not contain duplicates")
        return self


__all__ = ["ClipSegment"]
