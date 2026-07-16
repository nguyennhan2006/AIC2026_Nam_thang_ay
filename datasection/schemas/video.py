"""Canonical video aggregate and cross-checks for scene timing."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import ModelProvenance, RelativeArtifactPath, SHA256Checksum, StrictModel, VideoId, utc_now
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
        return self

    @property
    def scene_count(self) -> int:
        return len(self.scenes)
