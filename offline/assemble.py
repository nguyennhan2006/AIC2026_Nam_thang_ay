"""Hợp nhất stage pack -> canonical dataset (PR-02).

Đây là đường canonical để dựng dữ liệu THẬT. `offline/pipeline.py` tự cắt
scene đều 8 giây bằng vòng for và không đọc pack nào — nó chỉ còn dùng cho
smoke test local với mock provider.

Ba việc module này làm mà không chỗ nào khác làm:

1. **Nối notebook với canonical.** Notebook Kaggle ghi manifest theo contract
   riêng của nó; ở đây chúng được dịch sang `datasection.schemas`.
2. **Sửa hai lệch contract đã biết** (xem `_scene_from_row`): `scene_id` của
   notebook dùng 5 chữ số còn canonical dùng 4, và `end_frame` của notebook
   là *inclusive* còn canonical là `end_frame_exclusive`.
3. **Cách ly record hỏng thay vì bỏ qua im lặng.** Mọi record không map được
   đi vào `quarantine.jsonl` kèm lý do, và `AssembleReport` nêu rõ stage nào
   vắng mặt — dataset thiếu OCR phải là điều nhìn thấy được, không phải điều
   phát hiện ra lúc thi.

Khóa join giữa các stage: xem docstring `offline/stagepack.py`. Stage mức
keyframe join theo `(video_id, frame_idx)`; `keyframe_id` canonical do CHÍNH
module này dựng, vì id nhúng `scene_idx` mà notebook không biết trước.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from datasection.exporter import atomic_jsonl, export_dataset
from datasection.schemas import (
    ASRSegment,
    BoundingBox,
    CaptionRecord,
    ClipSegment,
    ColorFeature,
    DatasetManifest,
    Event,
    Keyframe,
    ModelArtifact,
    ModelProvenance,
    NamedColorRatio,
    ObjectInstance,
    OCRInstance,
    Scene,
    SceneCaptionRecord,
    SceneKeyword,
    Video,
)
from offline.action_tags import extract_action_tags
from offline.clip_pooling import build_clip_segment, build_clip_windows, select_clip_frames
from offline.config import OfflineSettings
from offline.event_grouping import build_event, group_scenes_into_events, link_event_neighbors
from offline.stagepack import REQUIRED_STAGES, StagePack, StagePackError

MAX_SCENES_PER_VIDEO = 10_000  # canonical: scene_idx <= 9999
# Stage được ghi vào DatasetManifest.models (khớp Literal của ModelArtifact.task).
MODEL_ARTIFACT_TASKS = frozenset(
    {"scene", "keyframe", "caption", "ocr", "object", "asr", "embedding", "color"}
)


@dataclass(slots=True)
class AssembleReport:
    """Kết quả assemble — dùng để biết dataset còn thiếu gì trước khi thi."""

    stages_present: list[str] = field(default_factory=list)
    stages_missing: list[str] = field(default_factory=list)
    video_count: int = 0
    scene_count: int = 0
    keyframe_count: int = 0
    clip_count: int = 0
    event_count: int = 0
    asr_segment_count: int = 0
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def quarantine(self, stage: str, reason: str, record: dict[str, Any]) -> None:
        self.quarantined.append({"stage": stage, "reason": reason, "record": record})

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages_present": self.stages_present,
            "stages_missing": self.stages_missing,
            "video_count": self.video_count,
            "scene_count": self.scene_count,
            "keyframe_count": self.keyframe_count,
            "clip_count": self.clip_count,
            "event_count": self.event_count,
            "asr_segment_count": self.asr_segment_count,
            "quarantined_count": len(self.quarantined),
            "warnings": self.warnings,
        }


def _provenance(pack: StagePack | None, task: str, pipeline_version: str) -> ModelProvenance:
    fields = pack.provenance_fields() if pack else {"model_name": f"missing:{task}", "model_revision": None}
    return ModelProvenance(
        model_name=f"{fields['model_name']}:{task}",
        model_revision=fields.get("model_revision"),
        pipeline_version=pipeline_version,
        device=str((pack.model_info.get("device") if pack else None) or "unknown"),
    )


class PackEmbeddingReader:
    """Đọc vector frame từ embedding pack thay vì gọi lại model.

    Chấp nhận vector inline (`vector`) hoặc file JSON/`.npy` trỏ bởi
    `vector_path` (tương đối so với gốc pack). Frame không có vector -> lỗi
    tường minh, vì pooling thiếu frame sẽ tạo clip vector sai lệch âm thầm.
    """

    def __init__(self, pack: StagePack, vectors: dict[tuple[str, int], dict[str, Any]]) -> None:
        self._pack = pack
        self._vectors = vectors
        self._cache: dict[str, list[float]] = {}

    async def read(self, keyframe: Keyframe) -> list[float]:
        cached = self._cache.get(keyframe.keyframe_id)
        if cached is not None:
            return cached
        row = self._vectors.get((keyframe.video_id, keyframe.frame_idx))
        if row is None:
            raise StagePackError(
                f"embedding pack thiếu vector cho {keyframe.video_id} frame {keyframe.frame_idx}"
            )
        vector = _read_vector(row, self._pack.root)
        self._cache[keyframe.keyframe_id] = vector
        return vector


def _read_vector(row: dict[str, Any], pack_root: Path) -> list[float]:
    if row.get("vector"):
        return [float(value) for value in row["vector"]]
    path_value = row.get("vector_path")
    if not path_value:
        raise StagePackError(f"embedding row thiếu cả `vector` lẫn `vector_path`: {row}")
    path = pack_root / str(path_value)
    if path.suffix == ".npy":
        import numpy  # local: chỉ cần khi pack dùng .npy

        return [float(value) for value in numpy.load(path)]
    return [float(value) for value in json.loads(path.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------
# Nạp từng stage
# --------------------------------------------------------------------------


def _load_video_rows(pack: StagePack) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for row in pack.rows():
        video_id = str(row["video_id"])
        if video_id in videos:
            raise StagePackError(f"video pack có {video_id} trùng lặp")
        videos[video_id] = row
    return videos


def _load_scene_rows(pack: StagePack, report: AssembleReport) -> dict[str, list[dict[str, Any]]]:
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pack.rows():
        try:
            video_id = str(row["video_id"])
            int(row["start_frame"])
            int(row["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            report.quarantine("scene", f"thiếu/sai trường bắt buộc: {exc}", row)
            continue
        scenes[video_id].append(row)
    for rows in scenes.values():
        rows.sort(key=lambda item: (int(item["start_frame"]), int(item.get("scene_index", 0))))
    return scenes


def _keyed_by_frame(
    pack: StagePack, stage: str, report: AssembleReport
) -> dict[tuple[str, int], dict[str, Any]]:
    """Gom record mức keyframe theo `(video_id, frame_idx)` — khóa join chuẩn."""

    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in pack.rows():
        try:
            key = (str(row["video_id"]), int(row["frame_idx"]))
        except (KeyError, TypeError, ValueError) as exc:
            report.quarantine(stage, f"thiếu video_id/frame_idx: {exc}", row)
            continue
        if key in keyed:
            report.quarantine(stage, f"trùng khóa (video_id, frame_idx)={key}", row)
            continue
        keyed[key] = row
    return keyed


# --------------------------------------------------------------------------
# Dựng entity canonical
# --------------------------------------------------------------------------


def _scene_bounds(row: dict[str, Any], fps: float, frame_count: int) -> tuple[int, int, float, float]:
    """Chuyển `end_frame` inclusive của notebook sang `end_frame_exclusive`.

    Thời gian được TÍNH LẠI từ frame chứ không lấy `start_sec`/`end_sec` của
    notebook: notebook tính `end_sec = end_frame / fps` với `end_frame`
    inclusive, lệch một frame so với khoảng nửa mở canonical và làm keyframe
    ở frame cuối rơi ra ngoài `[start_sec, end_sec)`.
    """

    start_frame = max(0, int(row["start_frame"]))
    end_frame_exclusive = min(frame_count, int(row["end_frame"]) + 1)
    if end_frame_exclusive <= start_frame:
        raise ValueError(
            f"scene rỗng sau khi chuẩn hóa: start_frame={start_frame} "
            f"end_frame={row['end_frame']} frame_count={frame_count}"
        )
    return start_frame, end_frame_exclusive, start_frame / fps, end_frame_exclusive / fps


def _keyframe_from_rows(
    *,
    video_id: str,
    scene_id: str,
    frame_idx: int,
    base: dict[str, Any],
    fps: float,
    enrichment: dict[str, dict[tuple[str, int], dict[str, Any]]],
    packs: dict[str, StagePack],
    pipeline_version: str,
) -> Keyframe:
    key = (video_id, frame_idx)
    caption_row = enrichment.get("caption", {}).get(key, {})
    ocr_row = enrichment.get("ocr", {}).get(key, {})
    object_row = enrichment.get("object", {}).get(key, {})
    color_row = enrichment.get("color", {}).get(key, {})
    embedding_row = enrichment.get("embedding", {}).get(key, {})

    captions = [
        CaptionRecord(
            caption_type=item.get("caption_type", "detailed"),
            language=item.get("language", "vi"),
            text=item["text"],
            confidence=item.get("confidence"),
            provenance=_provenance(packs.get("caption"), "caption", pipeline_version),
        )
        for item in caption_row.get("captions", [])
        if item.get("text")
    ]
    ocr_instances = [
        OCRInstance(
            text=item["text"],
            normalized_text=item.get("normalized_text"),
            language=item.get("language"),
            confidence=item.get("confidence", 0.0),
            bbox=BoundingBox.model_validate(item["bbox"]),
            provenance=_provenance(packs.get("ocr"), "ocr", pipeline_version),
        )
        for item in ocr_row.get("instances", [])
        if item.get("text")
    ]
    objects = [
        ObjectInstance(
            label=item["label"],
            confidence=item.get("confidence", 0.0),
            bbox=BoundingBox.model_validate(item["bbox"]),
            attributes=item.get("attributes", {}),
            provenance=_provenance(packs.get("object"), "object", pipeline_version),
        )
        for item in object_row.get("objects", [])
        if item.get("label")
    ]
    color = None
    if color_row.get("dominant_colors") or color_row.get("hsv_histogram"):
        color = ColorFeature(
            dominant_hex=color_row.get("dominant_hex", []),
            dominant_colors=[NamedColorRatio(**item) for item in color_row.get("dominant_colors", [])],
            mean_hsv=tuple(color_row["mean_hsv"]) if color_row.get("mean_hsv") else None,
            hsv_histogram=color_row.get("hsv_histogram", []),
            regions=color_row.get("regions", {}),
            provenance=_provenance(packs.get("color"), "color", pipeline_version),
        )
    caption_text = " ".join(item.text for item in captions)
    return Keyframe(
        keyframe_id=f"{scene_id}_F{frame_idx:06d}",
        video_id=video_id,
        scene_id=scene_id,
        frame_idx=frame_idx,
        # Luôn tính lại từ frame_idx/fps, không lấy timestamp_sec thô của pack —
        # cùng nguyên tắc đã áp cho start_sec/end_sec ở _scene_bounds(). Nếu tin
        # timestamp_sec thô (vd pts_time thật từ ffmpeg trong mapping CSV), nó có
        # thể lệch frame_idx/fps tới 1 frame do sai số làm tròn khác nhau giữa hai
        # nguồn — với keyframe nằm sát biên scene, giá trị thô có thể CHẠM đúng
        # end_sec đã tính lại và làm strict "<" trong Scene validator false-fail
        # (đã xảy ra thật với L21_V001_S0055_F007811: pts_time=260.4 == end_sec).
        # Dùng cùng công thức fps ở cả hai phía loại trừ khả năng đó hoàn toàn.
        timestamp_sec=frame_idx / fps,
        image_path=base["image_path"],
        width=int(base["width"]),
        height=int(base["height"]),
        roles=base.get("roles") or ["representative"],
        selection_score=base.get("selection_score"),
        quality=base.get("quality") or {},
        captions=captions,
        ocr_instances=ocr_instances,
        objects=objects,
        color=color,
        action_tags=extract_action_tags(caption_text),
        embedding_refs=embedding_row.get("embedding_refs", []),
        source_checksum=base.get("source_checksum"),
    )


def _project_asr(
    scene: Scene, segments: list[dict[str, Any]], pipeline_version: str, pack: StagePack | None
) -> list[ASRSegment]:
    """Cắt các segment ASR theo biên scene (giao khoảng thời gian)."""

    projected: list[ASRSegment] = []
    for source_idx, item in enumerate(segments):
        start_sec = max(scene.start_sec, float(item["start_sec"]))
        end_sec = min(scene.end_sec, float(item["end_sec"]))
        if end_sec <= start_sec or not str(item.get("text", "")).strip():
            continue
        projected.append(
            ASRSegment(
                segment_id=f"{scene.scene_id}_A{len(projected):04d}",
                # Notebook đặt asr_segment_id dạng `{vid}_A{n:06d}`, canonical
                # ASRSourceId là `{vid}_ASR{n:06d}` — dựng lại từ chỉ số nguồn.
                source_segment_id=f"{scene.video_id}_ASR{source_idx:06d}",
                start_sec=start_sec,
                end_sec=end_sec,
                text=str(item["text"]).strip(),
                normalized_text=item.get("normalized_text"),
                language=item.get("language"),
                confidence=item.get("confidence"),
                speaker_id=item.get("speaker_id"),
                provenance=_provenance(pack, "asr", pipeline_version),
            )
        )
    return projected


def _scene_keywords(keyframes: list[Keyframe], pipeline_version: str) -> list[SceneKeyword]:
    labels = sorted({item.label.casefold() for frame in keyframes for item in frame.objects})
    provenance = ModelProvenance(
        model_name="assemble:keyword", pipeline_version=pipeline_version
    )
    return [
        SceneKeyword(text=label, normalized_text=label, sources=["object"], provenance=provenance)
        for label in labels
    ]


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------


async def assemble(
    packs: dict[str, StagePack],
    *,
    settings: OfflineSettings,
    dataset_id: str = "aic-video-v1",
) -> tuple[DatasetManifest, AssembleReport]:
    """Dựng canonical dataset từ các stage pack và ghi ra `settings.export_dir`."""

    known_stages = set(REQUIRED_STAGES) | {
        "asr", "caption", "ocr", "object", "color", "embedding",
    }
    report = AssembleReport(
        stages_present=sorted(packs),
        stages_missing=sorted(known_stages - set(packs)),
    )
    missing_required = [stage for stage in REQUIRED_STAGES if stage not in packs]
    if missing_required:
        raise StagePackError(
            "thiếu stage bắt buộc: " + ", ".join(missing_required)
            + " — không dựng được Scene canonical hợp lệ"
        )
    for stage in report.stages_missing:
        report.warnings.append(
            f"stage {stage!r} vắng mặt: dataset sẽ không có dữ liệu {stage} "
            "(nhánh retrieval tương ứng sẽ im lặng, không phải lỗi runtime)"
        )

    video_rows = _load_video_rows(packs["video"])
    scene_rows_by_video = _load_scene_rows(packs["scene"], report)
    keyframe_rows = _keyed_by_frame(packs["keyframe"], "keyframe", report)
    enrichment = {
        stage: _keyed_by_frame(packs[stage], stage, report)
        for stage in ("caption", "ocr", "object", "color", "embedding")
        if stage in packs
    }
    asr_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if "asr" in packs:
        for row in packs["asr"].rows():
            asr_by_video[str(row["video_id"])].append(row)
        for rows in asr_by_video.values():
            rows.sort(key=lambda item: float(item["start_sec"]))

    frames_by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (video_id, frame_idx), row in keyframe_rows.items():
        frames_by_video[video_id].append((frame_idx, row))
    for rows in frames_by_video.values():
        rows.sort(key=lambda item: item[0])

    videos: list[Video] = []
    for video_id in sorted(video_rows):
        video_row = video_rows[video_id]
        fps = float(video_row["fps"])
        frame_count = int(video_row["frame_count"])
        rows = scene_rows_by_video.get(video_id, [])
        if not rows:
            report.quarantine("scene", f"video {video_id} không có scene nào", video_row)
            continue
        if len(rows) > MAX_SCENES_PER_VIDEO:
            raise StagePackError(
                f"{video_id} có {len(rows)} scene, vượt giới hạn canonical "
                f"{MAX_SCENES_PER_VIDEO} (scene_idx tối đa 4 chữ số)"
            )

        unassigned = dict(frames_by_video.get(video_id, []))
        scenes: list[Scene] = []
        # scene_idx được đánh lại 0..N-1 theo thứ tự thời gian: notebook đánh số
        # theo định dạng riêng (`_S%05d`) và có thể có khoảng trống sau khi rerun.
        for scene_idx, row in enumerate(rows):
            try:
                start_frame, end_frame_exclusive, start_sec, end_sec = _scene_bounds(
                    row, fps, frame_count
                )
            except ValueError as exc:
                report.quarantine("scene", str(exc), row)
                continue
            scene_id = f"{video_id}_S{scene_idx:04d}"
            frame_indices = sorted(
                idx for idx in unassigned if start_frame <= idx < end_frame_exclusive
            )
            if not frame_indices:
                report.quarantine(
                    "keyframe",
                    f"scene {scene_id} [{start_frame}, {end_frame_exclusive}) không có keyframe nào "
                    "— canonical Scene bắt buộc >= 1 keyframe",
                    {"video_id": video_id, "scene_index": row.get("scene_index")},
                )
                continue
            keyframes = [
                _keyframe_from_rows(
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_idx=frame_idx,
                    base=unassigned.pop(frame_idx),
                    fps=fps,
                    enrichment=enrichment,
                    packs=packs,
                    pipeline_version=settings.pipeline_version,
                )
                for frame_idx in frame_indices
            ]
            caption_text = " ".join(
                record.text for frame in keyframes for record in frame.captions
            )
            scene_captions = (
                [
                    SceneCaptionRecord(
                        caption_type="visual",
                        language="vi",
                        text=caption_text,
                        evidence_keyframe_ids=[frame.keyframe_id for frame in keyframes],
                        provenance=_provenance(
                            packs.get("caption"), "scene-caption", settings.pipeline_version
                        ),
                    )
                ]
                if caption_text
                else []
            )
            scene = Scene(
                scene_id=scene_id,
                video_id=video_id,
                scene_idx=scene_idx,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                start_sec=start_sec,
                end_sec=end_sec,
                segmentation_provenance=_provenance(
                    packs["scene"], "scene-detection", settings.pipeline_version
                ),
                keyframes=keyframes,
                captions=scene_captions,
                keywords=_scene_keywords(keyframes, settings.pipeline_version),
                action_tags=sorted({tag for frame in keyframes for tag in frame.action_tags}),
            )
            projected = _project_asr(
                scene, asr_by_video.get(video_id, []), settings.pipeline_version, packs.get("asr")
            )
            if projected:
                scene = scene.model_copy(update={"asr_segments": projected})
                report.asr_segment_count += len(projected)
            scenes.append(scene)

        for frame_idx in sorted(unassigned):
            report.quarantine(
                "keyframe",
                f"frame {frame_idx} của {video_id} không rơi vào scene nào",
                {"video_id": video_id, "frame_idx": frame_idx},
            )
        if not scenes:
            report.quarantine("scene", f"video {video_id} không còn scene hợp lệ", video_row)
            continue

        clips = await _build_clips(scenes, fps, packs, enrichment, settings, report)
        groups = group_scenes_into_events(
            scenes,
            settings.event_max_gap_sec,
            settings.event_max_duration_sec,
            settings.event_min_text_overlap,
        )
        events = link_event_neighbors([
            build_event(
                video_id,
                event_idx,
                group,
                settings.event_config_id,
                _provenance(packs["scene"], "event-grouping", settings.pipeline_version),
            )
            for event_idx, group in enumerate(groups)
        ])
        # Event trỏ tới scene, nhưng online cần chiều ngược lại để dedup theo
        # event mà không phải nạp toàn bộ events.jsonl cho mỗi request.
        event_by_scene = {
            scene_id: event.event_id for event in events for scene_id in event.scene_ids
        }
        scenes = [
            scene.model_copy(
                update={
                    "extensions": {
                        **scene.extensions,
                        "event_id": event_by_scene[scene.scene_id],
                    }
                }
            )
            if scene.scene_id in event_by_scene
            else scene
            for scene in scenes
        ]
        videos.append(
            Video(
                video_id=video_id,
                source_path=str(video_row["source_path"]),
                source_checksum=video_row.get("source_checksum"),
                fps=fps,
                frame_count=frame_count,
                duration_sec=float(video_row.get("duration_sec", frame_count / fps)),
                width=int(video_row["width"]),
                height=int(video_row["height"]),
                codec=str(video_row.get("codec", "unknown")),
                audio_present=bool(video_row.get("audio_present", False)),
                probe_provenance=_provenance(
                    packs["video"], "ffprobe", settings.pipeline_version
                ),
                scenes=scenes,
                clips=clips,
                events=events,
            )
        )

    if not videos:
        raise StagePackError("không dựng được video nào — xem quarantine.jsonl")

    report.video_count = len(videos)
    report.scene_count = sum(len(item.scenes) for item in videos)
    report.keyframe_count = sum(len(scene.keyframes) for item in videos for scene in item.scenes)
    report.clip_count = sum(len(item.clips) for item in videos)
    report.event_count = sum(len(item.events) for item in videos)

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        build_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        pipeline_version=settings.pipeline_version,
        video_count=report.video_count,
        scene_count=report.scene_count,
        keyframe_count=report.keyframe_count,
        clip_count=report.clip_count,
        event_count=report.event_count,
        # `video` không nằm trong ModelArtifact.task: ffprobe là bước đo đạc,
        # không phải model sinh nội dung.
        models=[
            ModelArtifact(
                task=stage,
                model_name=str(pack.model_info.get("model") or f"unknown:{stage}"),
                revision=str(pack.model_info.get("pack_version") or ""),
            )
            for stage, pack in sorted(packs.items())
            if stage in MODEL_ARTIFACT_TASKS
        ],
    )
    result = export_dataset(videos, settings.export_dir, manifest)
    atomic_jsonl(settings.export_dir / "quarantine.jsonl", report.quarantined)
    (settings.export_dir / "assemble_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, report


async def _build_clips(
    scenes: list[Scene],
    fps: float,
    packs: dict[str, StagePack],
    enrichment: dict[str, dict[tuple[str, int], dict[str, Any]]],
    settings: OfflineSettings,
    report: AssembleReport,
) -> list[ClipSegment]:
    """Clip pooling chỉ chạy khi có embedding pack.

    Không có embedding thì clip vector sẽ không tồn tại — dựng ClipSegment
    rỗng chỉ tạo ra entity giả, nên bỏ hẳn và ghi warning.
    """

    pack = packs.get("embedding")
    if pack is None:
        return []
    reader = PackEmbeddingReader(pack, enrichment.get("embedding", {}))
    clips: list[ClipSegment] = []
    for scene in scenes:
        windows = build_clip_windows(
            scene,
            fps,
            settings.clip_duration_sec,
            settings.clip_stride_sec,
            settings.clip_min_tail_sec,
        )
        for window in windows:
            selected = select_clip_frames(
                scene,
                window,
                settings.clip_max_sampled_frames,
                settings.clip_fallback_max_distance_sec,
                settings.clip_empty_window_policy,
            )
            if selected is None:
                continue
            try:
                clips.append(
                    await build_clip_segment(
                        scene,
                        window,
                        selected,
                        reader,
                        settings.data_root,
                        settings.clip_config_id,
                        _provenance(pack, "embedding", settings.pipeline_version),
                        str(pack.model_info.get("model") or "unknown"),
                        str(pack.model_info.get("pack_version") or ""),
                    )
                )
            except StagePackError as exc:
                report.quarantine(
                    "embedding", str(exc), {"scene_id": scene.scene_id, "window": str(window)}
                )
    return clips


__all__ = ["AssembleReport", "PackEmbeddingReader", "assemble"]
