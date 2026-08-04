"""Canonical video aggregate and cross-checks for scene timing."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .clip import ClipSegment
from .common import ModelProvenance, RelativeArtifactPath, SHA256Checksum, StrictModel, VideoId, utc_now
from .event import Event
from .scene import Scene


class Video(StrictModel):
    """A source video containing ordered, non-overlapping scenes."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    video_id: VideoId
    source_path: RelativeArtifactPath
    source_checksum: SHA256Checksum | None = None
    fps: Annotated[float, Field(gt=0, le=1000)]
    frame_count: Annotated[int, Field(gt=0)]
    duration_sec: Annotated[float, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    codec: str | None = None
    audio_present: bool = False
    probe_provenance: ModelProvenance
    scenes: list[Scene] = Field(default_factory=list)
    # Clip không nhét vào Scene: clip/scene có lifecycle và index khác nhau (đổi
    # scene boundary không nên buộc rebuild toàn bộ clip) — xem datasection/schemas/clip.py.
    clips: list[ClipSegment] = Field(default_factory=list)
    # Event cũng là sibling collection như clip — xem datasection/schemas/event.py.
    events: list[Event] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return value

    @model_validator(mode="after")
    def validate_children(self) -> "Video":
        expected_duration = self.frame_count / self.fps
        tolerance = max(0.1, 2 / self.fps)
        if abs(self.duration_sec - expected_duration) > tolerance:
            raise ValueError("duration_sec must agree with frame_count/fps")
        previous_end_frame = 0
        previous_end_sec = 0.0
        ids: set[str] = set()
        for scene in self.scenes:
            if scene.video_id != self.video_id:
                raise ValueError(f"scene {scene.scene_id} belongs to another video")
            if scene.scene_id in ids:
                raise ValueError("scene IDs must be unique")
            if scene.start_frame < previous_end_frame or scene.start_sec < previous_end_sec - tolerance:
                raise ValueError("scenes must be ordered and non-overlapping")
            if scene.end_frame_exclusive > self.frame_count or scene.end_sec > self.duration_sec + tolerance:
                raise ValueError(f"scene {scene.scene_id} exceeds video duration")
            for frame in scene.keyframes:
                frame_time = frame.frame_idx / self.fps
                if abs(frame.timestamp_sec - frame_time) > tolerance:
                    raise ValueError(f"keyframe {frame.keyframe_id} timestamp disagrees with fps")
            ids.add(scene.scene_id)
            previous_end_frame = scene.end_frame_exclusive
            previous_end_sec = scene.end_sec
        scene_by_id = {scene.scene_id: scene for scene in self.scenes}
        clip_ids: set[str] = set()
        for clip in self.clips:
            if clip.video_id != self.video_id:
                raise ValueError(f"clip {clip.clip_id} belongs to another video")
            if clip.clip_id in clip_ids:
                raise ValueError("clip IDs must be unique")
            scene = scene_by_id.get(clip.scene_id)
            if scene is None:
                raise ValueError(f"clip {clip.clip_id} references unknown scene {clip.scene_id}")
            if clip.start_frame < scene.start_frame or clip.end_frame > scene.end_frame_exclusive:
                raise ValueError(f"clip {clip.clip_id} exceeds its scene's frame range")
            known_frame_ids = {frame.keyframe_id for frame in scene.keyframes}
            unknown = set(clip.sampled_frame_ids) - known_frame_ids
            if unknown:
                raise ValueError(f"clip {clip.clip_id} references keyframes outside its scene: {sorted(unknown)}")
            clip_ids.add(clip.clip_id)
        known_scene_ids = set(scene_by_id)
        event_ids: set[str] = set()
        scene_ids_in_events: set[str] = set()
        for event in self.events:
            if event.video_id != self.video_id:
                raise ValueError(f"event {event.event_id} belongs to another video")
            if event.event_id in event_ids:
                raise ValueError("event IDs must be unique")
            unknown_scenes = set(event.scene_ids) - known_scene_ids
            if unknown_scenes:
                raise ValueError(f"event {event.event_id} references unknown scenes: {sorted(unknown_scenes)}")
            already_grouped = set(event.scene_ids) & scene_ids_in_events
            if already_grouped:
                raise ValueError(f"scene assigned to more than one event: {sorted(already_grouped)}")
            scene_ids_in_events.update(event.scene_ids)
            event_ids.add(event.event_id)
        return self

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)
