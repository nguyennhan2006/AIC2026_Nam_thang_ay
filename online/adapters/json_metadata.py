"""Read canonical Scene JSONL exported by the sibling datasection package."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

from online.domain.models import FrameEvidence, FrameQuality, SceneDocument
from online.errors import MetadataNotFoundError


# Ngưỡng tỉ lệ để một màu được coi là màu TRỘI của frame.
#
# `dominant_colors` giữ top-8 màu bất kể tỉ lệ, mà ảnh thật nào cũng có chút của
# mọi sắc. Đo trên 765 scene sau khi bù color: "đỏ" xuất hiện ở **99.6%** scene,
# xám 99.1%, trắng 93.5%, xanh 92.8%. Một tín hiệu có mặt khắp nơi thì không
# phân biệt được gì — đúng cùng một loại lỗi với chữ lớp phủ trong OCR.
#
#     nguong   so mau/scene   mau pho bien nhat
#      0.00        8.08        red   100%
#      0.10        3.11        gray   72%
#      0.15        2.23        gray   61%
#      0.20        1.70        gray   51%
#
# 0.0 = giữ nguyên hành vi cũ (không lọc).
_COLOR_MIN_RATIO = float(os.getenv("AIC_COLOR_MIN_RATIO", "0.0"))


def _texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(item["text"]) for item in records if item.get("text")]


def _boxes(records: list[dict[str, Any]]) -> list[dict[str, float] | None]:
    """bbox song song với `_texts` — CÙNG bộ lọc, nên cùng độ dài.

    Phải dùng đúng điều kiện `if item.get("text")` như `_texts`: lệch một phần
    tử là mọi chuỗi sau đó bị gán nhầm bbox, và bộ lọc lớp phủ sẽ bỏ nhầm chữ
    nội dung trong khi giữ lại lớp phủ.

    `None` khi bbox là khung TOÀN ẢNH: đó là quy ước của phần bù OCR text-only
    (`scripts/ocr_backfill.py`) nghĩa là "không biết vị trí", chứ không phải
    "chuỗi này phủ kín khung hình".
    """

    out: list[dict[str, float] | None] = []
    for item in records:
        if not item.get("text"):
            continue
        box = item.get("bbox")
        if not isinstance(box, dict):
            out.append(None)
            continue
        try:
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
        except (KeyError, TypeError, ValueError):
            out.append(None)
            continue
        full = x1 <= 0.001 and y1 <= 0.001 and x2 >= 0.999 and y2 >= 0.999
        out.append(None if full else {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


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
        ocr_boxes=_boxes(raw.get("ocr_instances", [])),
        object_labels=[
            str(item["label"]) for item in raw.get("objects", []) if item.get("label")
        ],
        action_tags=[str(item) for item in raw.get("action_tags", [])],
        dominant_colors=[
            str(item["name"])
            for item in color.get("dominant_colors", [])
            if item.get("name") and float(item.get("ratio") or 0.0) >= _COLOR_MIN_RATIO
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


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata mức video mà UI cần để quy đổi frame <-> giây CHÍNH XÁC.

    Trước đây loader chỉ giữ `source_path` (để gắn vào scene) và `frame_count`
    (cho submission validator), còn `fps` thì vứt. Hệ quả: mọi chỗ cần đổi
    frame sang giây phải suy fps từ biên của một scene — được với scene đủ
    dài, nhưng scene p50 chỉ 4 giây nên sai số làm tròn đủ để tua lệch, và
    hoàn toàn bất khả thi khi cần tua tới một frame KHÔNG nằm trong scene nào
    đã biết (đúng việc mà tab chỉnh submission bằng tay phải làm).
    """

    video_id: str
    source_path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int | None = None
    height: int | None = None


class JsonlSceneRepository:
    """In-memory read repository loaded atomically from a JSONL export."""

    def __init__(
        self,
        path: Path,
        scenes: dict[str, SceneDocument],
        *,
        video_frame_counts: dict[str, int] | None = None,
        videos: dict[str, VideoInfo] | None = None,
    ) -> None:
        self.path = path
        self._scenes = scenes
        # Cần cho submission_validator (PR-08): "frame thuộc video" chỉ kiểm
        # tra được nếu biết frame_count thật của video, không suy từ scene.
        self._video_frame_counts = video_frame_counts or {}
        self._videos = videos or {}

    @classmethod
    async def load(cls, path: Path) -> "JsonlSceneRepository":
        def read() -> tuple[dict[str, SceneDocument], dict[str, int], dict[str, "VideoInfo"]]:
            if not path.exists():
                raise MetadataNotFoundError(f"scene JSONL not found: {path}")
            scenes: dict[str, SceneDocument] = {}
            video_paths: dict[str, str] = {}
            video_frame_counts: dict[str, int] = {}
            video_infos: dict[str, VideoInfo] = {}
            videos_path = path.with_name("videos.jsonl")
            if videos_path.exists():
                with videos_path.open(encoding="utf-8") as videos:
                    for row in videos:
                        if not row.strip():
                            continue
                        video = json.loads(row)
                        video_id = video["video_id"]
                        video_paths[video_id] = video["source_path"]
                        if "frame_count" in video:
                            video_frame_counts[video_id] = int(video["frame_count"])
                        fps = float(video.get("fps") or 0.0)
                        frames = int(video.get("frame_count") or 0)
                        if fps > 0 and frames > 0:
                            video_infos[video_id] = VideoInfo(
                                video_id=video_id,
                                source_path=str(video["source_path"]),
                                fps=fps,
                                frame_count=frames,
                                duration_sec=float(video.get("duration_sec") or frames / fps),
                                width=video.get("width"),
                                height=video.get("height"),
                            )
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
            return scenes, video_frame_counts, video_infos

        scenes, video_frame_counts, video_infos = await asyncio.to_thread(read)
        return cls(
            path, scenes, video_frame_counts=video_frame_counts, videos=video_infos
        )

    async def get(self, scene_id: str) -> SceneDocument | None:
        return self._scenes.get(scene_id)

    async def video_frame_count(self, video_id: str) -> int | None:
        """Số frame thật của video, hoặc None nếu videos.jsonl không có nó.

        `None` (chứ không phải 0) khi thiếu thông tin: validator coi đây là
        "không kiểm tra được" chứ không phải "mọi frame đều vượt biên".
        """

        return self._video_frame_counts.get(video_id)

    async def video_info(self, video_id: str) -> VideoInfo | None:
        """Metadata video, hoặc None nếu `videos.jsonl` thiếu fps/frame_count."""

        return self._videos.get(video_id)

    async def all_videos(self) -> list[VideoInfo]:
        return [self._videos[key] for key in sorted(self._videos)]

    async def get_many(self, scene_ids: Sequence[str]) -> list[SceneDocument]:
        return [self._scenes[item] for item in scene_ids if item in self._scenes]

    async def all(self) -> list[SceneDocument]:
        return list(self._scenes.values())
