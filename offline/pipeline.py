"""End-to-end ingest → enrich → canonical export pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import warnings

from datasection.exporter import export_dataset
from datasection.schemas import (
    ASRSegment, BoundingBox, CaptionRecord, ClipSegment, ColorFeature, DatasetManifest, Event, Keyframe,
    ModelArtifact, ModelProvenance, NamedColorRatio, ObjectInstance, OCRInstance, Scene,
    SceneCaptionRecord, SceneKeyword, Video,
)
from offline.action_tags import extract_action_tags
from offline.clip_pooling import build_clip_segment, build_clip_windows, select_clip_frames
from offline.config import OfflineSettings
from offline.embedding_reader import MemoizedEmbeddingReader, ProviderEmbeddingReader
from offline.event_grouping import build_event, group_scenes_into_events, link_event_neighbors
from offline.media import FFmpegMedia
from offline.providers import MockInferenceProvider, RemoteInferenceProvider
from offline.state import JobLedger


VIDEO_RE = re.compile(r"^L\d{2}_V\d{3}$")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class OfflinePipeline:
    def __init__(self, settings: OfflineSettings, media: FFmpegMedia | None = None, provider=None) -> None:
        self.settings = settings
        self.media = media or FFmpegMedia()
        self.provider = provider or (
            RemoteInferenceProvider(settings.gpu_url or "", settings.gpu_api_key, settings.timeout_sec, settings.retries)
            if settings.provider == "remote" else MockInferenceProvider()
        )
        self.ledger = JobLedger(settings.state_dir)

    def provenance(self, task: str) -> ModelProvenance:
        return ModelProvenance(
            model_name=f"{self.provider.model_name}:{task}", model_revision=self.provider.revision,
            pipeline_version=self.settings.pipeline_version, device="remote-gpu" if self.settings.provider == "remote" else "cpu-mock",
        )

    async def _keyframe(self, video_id: str, scene_id: str, source: Path, info, frame_idx: int) -> Keyframe:
        relative = Path("processed/keyframes") / video_id / f"frame_{frame_idx:06d}.jpg"
        target = self.settings.data_root / relative
        if not target.exists():
            await asyncio.to_thread(self.media.extract_frame, source, frame_idx, target)
        caption = await self.provider.image("caption", target)
        caption_text = " ".join(item.get("text", "") for item in caption.get("captions", []))
        candidate_labels = sorted({token.casefold() for token in re.findall(r"[\wÀ-ỹ]+", caption_text) if len(token) >= 4})[:32]
        ocr, objects, color = await asyncio.gather(
            self.provider.image("ocr", target),
            self.provider.image("object", target, caption=caption_text or None, candidate_labels=candidate_labels),
            self.provider.image("color", target),
        )
        captions = [CaptionRecord(
            caption_type="detailed", text=item["text"], language=item.get("language", "vi"),
            confidence=item.get("confidence"), provenance=self.provenance("caption")
        ) for item in caption.get("captions", [])]
        ocr_instances = [OCRInstance(
            text=item["text"], normalized_text=item.get("normalized_text"), language=item.get("language"),
            confidence=item.get("confidence", 0.0), bbox=BoundingBox.model_validate(item["bbox"]), provenance=self.provenance("ocr")
        ) for item in ocr.get("instances", [])]
        object_instances = [ObjectInstance(
            label=item["label"], confidence=item.get("confidence", 0.0),
            bbox=BoundingBox.model_validate(item["bbox"]), attributes=item.get("attributes", {}), provenance=self.provenance("object")
        ) for item in objects.get("objects", [])]
        # Mock provider trả rỗng (không có model thật) -> giữ color=None thay vì một
        # ColorFeature rỗng không mang tín hiệu gì (khớp cách ocr/object xử lý "chưa có
        # model thật": list rỗng cho list-field, None cho optional single-object field).
        color_feature = None
        if color.get("dominant_colors") or color.get("hsv_histogram"):
            color_feature = ColorFeature(
                dominant_colors=[NamedColorRatio(**item) for item in color.get("dominant_colors", [])],
                mean_hsv=tuple(color["mean_hsv"]) if color.get("mean_hsv") else None,
                hsv_histogram=color.get("hsv_histogram", []),
                regions=color.get("regions", {}),
                provenance=self.provenance("color"),
            )
        return Keyframe(
            keyframe_id=f"{scene_id}_F{frame_idx:06d}", video_id=video_id, scene_id=scene_id,
            frame_idx=frame_idx, timestamp_sec=frame_idx / info.fps, image_path=relative.as_posix(),
            width=info.width, height=info.height, roles=["representative"], captions=captions,
            ocr_instances=ocr_instances, objects=object_instances, color=color_feature,
            action_tags=extract_action_tags(caption_text), source_checksum=_checksum(target),
        )

    async def process_video(self, source: Path) -> Video:
        video_id = source.stem
        if not VIDEO_RE.fullmatch(video_id):
            raise ValueError(f"video filename must be canonical, got {source.name}")
        self.ledger.write(video_id, "probe", "running")
        info = await asyncio.to_thread(self.media.probe, source)
        scene_frames = max(1, round(self.settings.scene_seconds * info.fps))
        scenes: list[Scene] = []
        for scene_idx, start in enumerate(range(0, info.frame_count, scene_frames)):
            end = min(info.frame_count, start + scene_frames)
            scene_id = f"{video_id}_S{scene_idx:04d}"
            positions = sorted({min(end - 1, start + round((i + 1) * (end - start) / (self.settings.keyframes_per_scene + 1))) for i in range(self.settings.keyframes_per_scene)})
            frames = [await self._keyframe(video_id, scene_id, source, info, frame_idx) for frame_idx in positions]
            captions = []
            text = " ".join(record.text for frame in frames for record in frame.captions)
            if text:
                captions.append(SceneCaptionRecord(
                    caption_type="visual", language="vi", text=text,
                    evidence_keyframe_ids=[frame.keyframe_id for frame in frames], provenance=self.provenance("scene-caption")
                ))
            keywords: list[SceneKeyword] = []
            labels = sorted({item.label.casefold() for frame in frames for item in frame.objects})
            for label in labels:
                keywords.append(SceneKeyword(text=label, normalized_text=label, sources=["object"], provenance=self.provenance("keyword")))
            action_tags = sorted({tag for frame in frames for tag in frame.action_tags})
            scenes.append(Scene(
                scene_id=scene_id, video_id=video_id, scene_idx=scene_idx,
                start_frame=start, end_frame_exclusive=end, start_sec=start / info.fps, end_sec=end / info.fps,
                segmentation_provenance=self.provenance("uniform-scene"), keyframes=frames, captions=captions,
                keywords=keywords, action_tags=action_tags,
            ))
            self.ledger.write(video_id, "enrich", "running", completed_scenes=scene_idx + 1)
        relative_source = source.resolve().relative_to(self.settings.data_root).as_posix()
        if info.audio_present:
            asr = await self.provider.video("asr", relative_source)
            enriched_scenes: list[Scene] = []
            for scene in scenes:
                projected: list[ASRSegment] = []
                for source_idx, item in enumerate(asr.get("segments", [])):
                    start_sec = max(scene.start_sec, float(item["start_sec"]))
                    end_sec = min(scene.end_sec, float(item["end_sec"]))
                    if end_sec <= start_sec:
                        continue
                    projected.append(ASRSegment(
                        segment_id=f"{scene.scene_id}_A{len(projected):04d}",
                        source_segment_id=f"{video_id}_ASR{source_idx:06d}",
                        start_sec=start_sec, end_sec=end_sec, text=item["text"],
                        normalized_text=item.get("normalized_text"), language=item.get("language"),
                        confidence=item.get("confidence"), speaker_id=item.get("speaker_id"), provenance=self.provenance("asr"),
                    ))
                enriched_scenes.append(scene.model_copy(update={"asr_segments": projected}))
            scenes = enriched_scenes
        embedding_reader = MemoizedEmbeddingReader(ProviderEmbeddingReader(self.provider, self.settings.data_root))
        clips: list[ClipSegment] = []
        for scene in scenes:
            windows = build_clip_windows(
                scene, info.fps, self.settings.clip_duration_sec, self.settings.clip_stride_sec, self.settings.clip_min_tail_sec,
            )
            for window in windows:
                selected = select_clip_frames(
                    scene, window, self.settings.clip_max_sampled_frames,
                    self.settings.clip_fallback_max_distance_sec, self.settings.clip_empty_window_policy,
                )
                if selected is None:
                    warnings.warn(
                        f"skipping clip window {scene.scene_id} [{window.start_sec:.2f}s-{window.end_sec:.2f}s]: "
                        "no keyframe within fallback distance", stacklevel=2,
                    )
                    continue
                clips.append(await build_clip_segment(
                    scene, window, selected, embedding_reader, self.settings.data_root,
                    self.settings.clip_config_id, self.provenance("embedding"),
                    self.provider.model_name, self.provider.revision,
                ))
        groups = group_scenes_into_events(
            scenes, self.settings.event_max_gap_sec, self.settings.event_max_duration_sec, self.settings.event_min_text_overlap,
        )
        events = [
            build_event(video_id, event_idx, group, self.settings.event_config_id, self.provenance("event-grouping"))
            for event_idx, group in enumerate(groups)
        ]
        events = link_event_neighbors(events)
        video = Video(
            video_id=video_id, source_path=relative_source, source_checksum=_checksum(source), fps=info.fps,
            frame_count=info.frame_count, duration_sec=info.frame_count / info.fps, width=info.width, height=info.height,
            codec=info.codec, audio_present=info.audio_present, probe_provenance=self.provenance("ffprobe"), scenes=scenes,
            clips=clips, events=events,
        )
        self.ledger.write(video_id, "complete", "succeeded", scene_count=len(scenes))
        return video

    async def run(self) -> DatasetManifest:
        sources = sorted(path for path in self.settings.input_dir.glob("*") if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi"})
        if not sources:
            raise FileNotFoundError(f"no video found in {self.settings.input_dir}")
        videos = []
        for source in sources:
            try:
                videos.append(await self.process_video(source))
            except Exception as exc:
                self.ledger.write(source.stem, "failed", "failed", error=str(exc))
                raise
        manifest = DatasetManifest(
            dataset_id="aic-video-v1", build_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            pipeline_version=self.settings.pipeline_version, video_count=len(videos),
            scene_count=sum(len(item.scenes) for item in videos),
            keyframe_count=sum(len(scene.keyframes) for item in videos for scene in item.scenes),
            clip_count=sum(len(item.clips) for item in videos),
            event_count=sum(len(item.events) for item in videos),
            models=[ModelArtifact(task=task, model_name=self.provider.model_name, revision=self.provider.revision) for task in ("caption", "ocr", "object", "asr", "embedding", "color")],
        )
        return await asyncio.to_thread(export_dataset, videos, self.settings.export_dir, manifest)
