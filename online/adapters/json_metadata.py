"""Read canonical Scene JSONL exported by the sibling datasection package."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from online.domain.models import FrameEvidence, FrameQuality, SceneDocument
from online.errors import MetadataNotFoundError


def _texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(item["text"]) for item in records if item.get("text")]


def project_frame(
    raw: dict[str, Any], *, start_frame: int, end_frame_exclusive: int
) -> FrameEvidence:
    """Chiếu một canonical Keyframe sang FrameEvidence, GIỮ NGUYÊN `frame_idx`.

    Trước PR-01 hàm `project_scene` chỉ lấy id/path/timestamp và vứt
    `frame_idx`, khiến online không xuất được submission. `frame_idx` giờ là
    field bắt buộc của `FrameEvidence`, nên regression tương tự sẽ fail ngay
    ở tầng validate chứ không âm thầm.
    """

    frame_idx = int(raw["frame_idx"])
    color = raw.get("color") or {}
    quality_raw = raw.get("quality") or {}
    return FrameEvidence(
        keyframe_id=raw["keyframe_id"],
        video_id=raw["video_id"],
        scene_id=raw["scene_id"],
        frame_idx=frame_idx,
        timestamp_sec=float(raw["timestamp_sec"]),
        image_path=raw["image_path"],
        selection_score=raw.get("selection_score"),
        quality=FrameQuality(
            sharpness=quality_raw.get("sharpness"),
            brightness=quality_raw.get("brightness"),
            contrast=quality_raw.get("contrast"),
            black_frame_ratio=quality_raw.get("black_frame_ratio"),
            duplicate_score=quality_raw.get("duplicate_score"),
        ),
        boundary_distance_frames=min(
            frame_idx - start_frame, end_frame_exclusive - 1 - frame_idx
        ),
        captions=_texts(raw.get("captions", [])),
        ocr_texts=_texts(raw.get("ocr_instances", [])),
        object_labels=[
            str(item["label"]) for item in raw.get("objects", []) if item.get("label")
        ],
        action_tags=[str(item) for item in raw.get("action_tags", [])],
        dominant_colors=[
            str(item["name"])
            for item in color.get("dominant_colors", [])
            if item.get("name")
        ],
        embedding_names=[
            str(item["embedding_name"])
            for item in raw.get("embedding_refs", [])
            if item.get("embedding_name")
        ],
    )


def project_scene(raw: dict[str, Any], video_path: str | None = None) -> SceneDocument:
    """Build an online read projection without mutating canonical metadata."""

    start_frame = int(raw["start_frame"])
    end_frame_exclusive = int(raw["end_frame_exclusive"])
    keyframes = [
        project_frame(
            item, start_frame=start_frame, end_frame_exclusive=end_frame_exclusive
        )
        for item in raw.get("keyframes", [])
    ]
    keyframe_captions = [text for frame in keyframes for text in frame.captions]
    ocr_texts = [text for frame in keyframes for text in frame.ocr_texts]
    object_labels = [label for frame in keyframes for label in frame.object_labels]
    color_names = {name for frame in keyframes for name in frame.dominant_colors}
    scene_captions = _texts(raw.get("captions", []))
    asr_texts = _texts(raw.get("asr_segments", []))
    keywords = [
        str(item.get("normalized_text") or item.get("text"))
        for item in raw.get("keywords", [])
        if item.get("normalized_text") or item.get("text")
    ]
    action_tags = [str(item) for item in raw.get("action_tags", [])]
    return SceneDocument(
        scene_id=raw["scene_id"],
        video_id=raw["video_id"],
        video_path=video_path,
        scene_idx=raw["scene_idx"],
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        start_sec=raw["start_sec"],
        end_sec=raw["end_sec"],
        # Scene canonical không mang event_id (Event trỏ ngược tới scene_ids);
        # exporter/assemble ghi lại quan hệ này vào `extensions` khi có.
        event_id=(raw.get("extensions") or {}).get("event_id"),
        artifact_version=raw.get("schema_version"),
        keyframes=keyframes,
        object_labels=object_labels,
        captions=scene_captions + keyframe_captions,
        ocr_texts=ocr_texts,
        asr_texts=asr_texts,
        keywords=keywords,
        action_tags=sorted(set(action_tags)),
        color_names=sorted(color_names),
    )


class JsonlSceneRepository:
    """In-memory read repository loaded atomically from a JSONL export."""

    def __init__(self, path: Path, scenes: dict[str, SceneDocument]) -> None:
        self.path = path
        self._scenes = scenes

    @classmethod
    async def load(cls, path: Path) -> "JsonlSceneRepository":
        def read() -> dict[str, SceneDocument]:
            if not path.exists():
                raise MetadataNotFoundError(f"scene JSONL not found: {path}")
            scenes: dict[str, SceneDocument] = {}
            video_paths: dict[str, str] = {}
            videos_path = path.with_name("videos.jsonl")
            if videos_path.exists():
                with videos_path.open(encoding="utf-8") as videos:
                    for row in videos:
                        if row.strip():
                            video = json.loads(row)
                            video_paths[video["video_id"]] = video["source_path"]
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    # Canonical validation is the integration gate. Import is
                    # local to keep the Online projection independently usable.
                    try:
                        from datasection.schemas import Scene
                        Scene.model_validate(raw)
                    except ImportError:
                        pass
                    scene = project_scene(raw, video_paths.get(raw["video_id"]))
                    if scene.scene_id in scenes:
                        raise ValueError(
                            f"duplicate scene_id {scene.scene_id} at line {line_number}"
                        )
                    scenes[scene.scene_id] = scene
            if not scenes:
                raise MetadataNotFoundError(
                    f"no scenes loaded from {path}; the export is empty — "
                    "run the offline pipeline/exporter before starting online"
                )
            return scenes

        return cls(path, await asyncio.to_thread(read))

    async def get(self, scene_id: str) -> SceneDocument | None:
        return self._scenes.get(scene_id)

    async def get_many(self, scene_ids: Sequence[str]) -> list[SceneDocument]:
        return [self._scenes[item] for item in scene_ids if item in self._scenes]

    async def all(self) -> list[SceneDocument]:
        return list(self._scenes.values())
