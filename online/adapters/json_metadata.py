"""Read canonical Scene JSONL exported by the sibling datasection package."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from online.domain.models import SceneDocument
from online.errors import MetadataNotFoundError


def _texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(item["text"]) for item in records if item.get("text")]


def project_scene(raw: dict[str, Any], video_path: str | None = None) -> SceneDocument:
    """Build an online read projection without mutating canonical metadata."""

    keyframes = raw.get("keyframes", [])
    keyframe_captions: list[str] = []
    ocr_texts: list[str] = []
    object_labels: list[str] = []
    for keyframe in keyframes:
        keyframe_captions.extend(_texts(keyframe.get("captions", [])))
        ocr_texts.extend(_texts(keyframe.get("ocr_instances", [])))
        object_labels.extend(str(item["label"]) for item in keyframe.get("objects", []) if item.get("label"))
    scene_captions = _texts(raw.get("captions", []))
    asr_texts = _texts(raw.get("asr_segments", []))
    keywords = [
        str(item.get("normalized_text") or item.get("text"))
        for item in raw.get("keywords", [])
        if item.get("normalized_text") or item.get("text")
    ]
    return SceneDocument(
        scene_id=raw["scene_id"],
        video_id=raw["video_id"],
        video_path=video_path,
        scene_idx=raw["scene_idx"],
        start_sec=raw["start_sec"],
        end_sec=raw["end_sec"],
        keyframe_ids=[item["keyframe_id"] for item in keyframes],
        keyframe_paths=[item["image_path"] for item in keyframes],
        keyframe_timestamps=[float(item["timestamp_sec"]) for item in keyframes],
        object_labels=object_labels,
        keyframe_evidence=[{
            "keyframe_id": item["keyframe_id"],
            "image_path": item["image_path"],
            "timestamp_sec": item["timestamp_sec"],
            "text": " ".join(
                [x.get("text", "") for x in item.get("captions", [])]
                + [x.get("text", "") for x in item.get("ocr_instances", [])]
                + [x.get("label", "") for x in item.get("objects", [])]
            ),
        } for item in keyframes],
        captions=scene_captions + keyframe_captions,
        ocr_texts=ocr_texts,
        asr_texts=asr_texts,
        keywords=keywords,
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
