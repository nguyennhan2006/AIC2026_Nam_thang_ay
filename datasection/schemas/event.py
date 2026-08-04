"""Canonical event-level metadata contract — Search Mixing Console W1 (event
grouping + event aggregation).

An event is a greedy partition of one video's *consecutive* scenes (every
scene belongs to exactly one event — see offline/event_grouping.py). Like
clips, events have a different lifecycle from scenes (regrouping events
should not force scene re-segmentation), so they live as a sibling
collection on `Video`, not nested inside `Scene`.

Baseline V1 groups scenes using temporal gap and a duration cap only — no
visual similarity model, since no scene-level visual embedding is persisted
yet (see docs/14_TECHNICAL_PREPARATION.md tech debt "persistent
frame-embedding cache"). `event_caption`/`keywords`/`action_tags` are plain
aggregation of the constituent scenes' own fields, not a new model output.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import (
    EventId,
    KeyframeId,
    ModelProvenance,
    NonEmptyStr,
    SceneId,
    StrictModel,
    VideoId,
)


class Event(StrictModel):
    """One greedily-grouped run of consecutive scenes inside a video."""

    schema_version: Literal["1.0.0"] = "1.0.0"

    event_id: EventId
    video_id: VideoId
    event_idx: Annotated[int, Field(ge=0, le=9_999)]

    scene_ids: list[SceneId] = Field(min_length=1)

    start_frame: Annotated[int, Field(ge=0)]
    end_frame_exclusive: Annotated[int, Field(gt=0)]
    start_sec: Annotated[float, Field(ge=0.0)]
    end_sec: Annotated[float, Field(gt=0.0)]

    event_caption: str | None = None
    representative_frame_ids: list[KeyframeId] = Field(default_factory=list)
    keywords: list[NonEmptyStr] = Field(default_factory=list)
    action_tags: list[NonEmptyStr] = Field(default_factory=list)

    previous_event_id: EventId | None = None
    next_event_id: EventId | None = None

    event_config_id: str = Field(min_length=1)
    provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_identity(self) -> "Event":
        expected_event_id = f"{self.video_id}_E{self.event_idx:04d}"
        if self.event_id != expected_event_id:
            raise ValueError(f"event_id must equal {expected_event_id} for this video/event_idx")
        for scene_id in self.scene_ids:
            if not scene_id.startswith(f"{self.video_id}_"):
                raise ValueError(f"event {self.event_id} references scene from another video: {scene_id}")
        if len(self.scene_ids) != len(set(self.scene_ids)):
            raise ValueError("scene_ids must not contain duplicates")
        return self

    @model_validator(mode="after")
    def validate_interval(self) -> "Event":
        if self.end_sec <= self.start_sec:
            raise ValueError("event requires end_sec > start_sec")
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("event requires end_frame_exclusive > start_frame")
        return self

    @model_validator(mode="after")
    def validate_neighbors(self) -> "Event":
        if self.previous_event_id is not None and not self.previous_event_id.startswith(f"{self.video_id}_"):
            raise ValueError("previous_event_id must belong to the same video")
        if self.next_event_id is not None and not self.next_event_id.startswith(f"{self.video_id}_"):
            raise ValueError("next_event_id must belong to the same video")
        if self.previous_event_id == self.event_id or self.next_event_id == self.event_id:
            raise ValueError("event cannot be its own neighbor")
        return self

    @model_validator(mode="after")
    def validate_aggregate_fields_unique(self) -> "Event":
        if len(self.keywords) != len(set(self.keywords)):
            raise ValueError("keywords must not contain duplicates")
        if len(self.action_tags) != len(set(self.action_tags)):
            raise ValueError("action_tags must not contain duplicates")
        if len(self.representative_frame_ids) != len(set(self.representative_frame_ids)):
            raise ValueError("representative_frame_ids must not contain duplicates")
        return self


__all__ = ["Event"]
