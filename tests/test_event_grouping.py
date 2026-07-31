"""W1 (offline feature #3/#4 — event grouping + aggregation)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from datasection.schemas import Keyframe, ModelProvenance, Scene, SceneCaptionRecord, SceneKeyword, Video
from offline.config import OfflineSettings
from offline.event_grouping import build_event, group_scenes_into_events, link_event_neighbors
from offline.media import MediaInfo
from offline.pipeline import OfflinePipeline
from offline.providers import MockInferenceProvider


def run(coro):
    return asyncio.run(coro)


def _provenance(name: str) -> ModelProvenance:
    return ModelProvenance(model_name=name, model_revision="test", pipeline_version="test-v1")


def _keyframe(scene_id: str, video_id: str, frame_idx: int, fps: float) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{scene_id}_F{frame_idx:06d}", video_id=video_id, scene_id=scene_id,
        frame_idx=frame_idx, timestamp_sec=frame_idx / fps, image_path=f"processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg",
        width=64, height=64, roles=["representative"],
    )


def _scene(
    video_id: str, scene_idx: int, start_frame: int, end_frame: int, fps: float,
    keywords: list[str] | None = None, action_tags: list[str] | None = None, caption_text: str | None = None,
) -> Scene:
    scene_id = f"{video_id}_S{scene_idx:04d}"
    keyframe = _keyframe(scene_id, video_id, start_frame, fps)
    scene_keywords = [
        SceneKeyword(text=word, normalized_text=word, sources=["object"], provenance=_provenance("keyword"))
        for word in (keywords or [])
    ]
    captions = []
    if caption_text:
        captions.append(SceneCaptionRecord(
            caption_type="visual", language="vi", text=caption_text,
            evidence_keyframe_ids=[keyframe.keyframe_id], provenance=_provenance("scene-caption"),
        ))
    return Scene(
        scene_id=scene_id, video_id=video_id, scene_idx=scene_idx,
        start_frame=start_frame, end_frame_exclusive=end_frame,
        start_sec=start_frame / fps, end_sec=end_frame / fps,
        segmentation_provenance=_provenance("seg"), keyframes=[keyframe],
        keywords=scene_keywords, action_tags=action_tags or [], captions=captions,
    )


class GroupScenesIntoEventsTests(unittest.TestCase):
    def test_empty_input_yields_no_groups(self) -> None:
        self.assertEqual(group_scenes_into_events([], max_gap_sec=2.0, max_event_duration_sec=60.0), [])

    def test_contiguous_scenes_merge_into_one_event_by_default(self) -> None:
        fps = 30.0
        scenes = [_scene("L01_V001", i, i * 30, (i + 1) * 30, fps) for i in range(4)]
        groups = group_scenes_into_events(scenes, max_gap_sec=2.0, max_event_duration_sec=60.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 4)

    def test_large_gap_splits_into_separate_events(self) -> None:
        fps = 30.0
        first = _scene("L01_V001", 0, 0, 30, fps)  # ends at t=1.0s
        second = _scene("L01_V001", 1, 300, 330, fps)  # starts at t=10.0s -> gap=9s
        groups = group_scenes_into_events([first, second], max_gap_sec=2.0, max_event_duration_sec=60.0)
        self.assertEqual(len(groups), 2)
        self.assertEqual([s.scene_id for s in groups[0]], [first.scene_id])
        self.assertEqual([s.scene_id for s in groups[1]], [second.scene_id])

    def test_max_duration_cap_splits_even_with_zero_gap(self) -> None:
        fps = 30.0
        # Each scene is 40s long with zero gap; a 2nd merge would exceed a 60s cap.
        scenes = [_scene("L01_V001", i, i * 1200, (i + 1) * 1200, fps) for i in range(3)]
        groups = group_scenes_into_events(scenes, max_gap_sec=2.0, max_event_duration_sec=60.0)
        self.assertEqual([len(g) for g in groups], [1, 1, 1])

    def test_text_overlap_gate_is_off_by_default(self) -> None:
        fps = 30.0
        first = _scene("L01_V001", 0, 0, 30, fps, keywords=["salt"])
        second = _scene("L01_V001", 1, 30, 60, fps, keywords=["boat"])
        groups = group_scenes_into_events([first, second], max_gap_sec=2.0, max_event_duration_sec=60.0)
        self.assertEqual(len(groups), 1)

    def test_text_overlap_gate_splits_when_enabled_and_disjoint(self) -> None:
        fps = 30.0
        first = _scene("L01_V001", 0, 0, 30, fps, keywords=["salt"])
        second = _scene("L01_V001", 1, 30, 60, fps, keywords=["boat"])
        groups = group_scenes_into_events([first, second], max_gap_sec=2.0, max_event_duration_sec=60.0, min_text_overlap=0.5)
        self.assertEqual(len(groups), 2)

    def test_text_overlap_gate_merges_when_enabled_and_shared(self) -> None:
        fps = 30.0
        first = _scene("L01_V001", 0, 0, 30, fps, keywords=["salt", "field"])
        second = _scene("L01_V001", 1, 30, 60, fps, keywords=["salt", "worker"])
        groups = group_scenes_into_events([first, second], max_gap_sec=2.0, max_event_duration_sec=60.0, min_text_overlap=0.2)
        self.assertEqual(len(groups), 1)


class BuildEventTests(unittest.TestCase):
    def test_aggregates_captions_keywords_and_action_tags(self) -> None:
        fps = 30.0
        first = _scene("L01_V001", 0, 0, 30, fps, keywords=["salt"], action_tags=["raking"], caption_text="Người cào muối")
        second = _scene("L01_V001", 1, 30, 60, fps, keywords=["boat"], action_tags=["waving"], caption_text="Người vẫy tay")
        event = build_event("L01_V001", 0, [first, second], "event-v1", _provenance("event-grouping"))
        self.assertEqual(event.event_id, "L01_V001_E0000")
        self.assertEqual(event.scene_ids, [first.scene_id, second.scene_id])
        self.assertEqual(event.start_frame, 0)
        self.assertEqual(event.end_frame_exclusive, 60)
        self.assertEqual(event.keywords, ["boat", "salt"])
        self.assertEqual(event.action_tags, ["raking", "waving"])
        self.assertEqual(event.event_caption, "Người cào muối Người vẫy tay")
        self.assertEqual(event.representative_frame_ids, [first.keyframes[0].keyframe_id, second.keyframes[0].keyframe_id])

    def test_single_scene_group_produces_valid_event(self) -> None:
        fps = 30.0
        scene = _scene("L01_V001", 0, 0, 30, fps)
        event = build_event("L01_V001", 0, [scene], "event-v1", _provenance("event-grouping"))
        self.assertEqual(event.scene_ids, [scene.scene_id])
        self.assertIsNone(event.event_caption)


class LinkEventNeighborsTests(unittest.TestCase):
    def test_first_and_last_have_none_neighbor(self) -> None:
        fps = 30.0
        scenes = [_scene("L01_V001", i, i * 30, (i + 1) * 30, fps) for i in range(3)]
        events = [build_event("L01_V001", i, [scene], "event-v1", _provenance("event-grouping")) for i, scene in enumerate(scenes)]
        linked = link_event_neighbors(events)
        self.assertIsNone(linked[0].previous_event_id)
        self.assertEqual(linked[0].next_event_id, linked[1].event_id)
        self.assertEqual(linked[1].previous_event_id, linked[0].event_id)
        self.assertEqual(linked[1].next_event_id, linked[2].event_id)
        self.assertIsNone(linked[2].next_event_id)


class VideoEventValidationTests(unittest.TestCase):
    def _video_kwargs(self, scenes: list[Scene]) -> dict:
        return dict(
            video_id="L01_V001", source_path="raw/videos/L01_V001.mp4", fps=30.0, frame_count=60,
            duration_sec=2.0, width=64, height=64, probe_provenance=_provenance("ffprobe"), scenes=scenes,
        )

    def test_scene_assigned_to_two_events_is_rejected(self) -> None:
        fps = 30.0
        scenes = [_scene("L01_V001", i, i * 30, (i + 1) * 30, fps) for i in range(2)]
        event_a = build_event("L01_V001", 0, [scenes[0]], "event-v1", _provenance("event-grouping"))
        event_b = build_event("L01_V001", 1, [scenes[0], scenes[1]], "event-v1", _provenance("event-grouping"))
        with self.assertRaisesRegex(ValueError, "more than one event"):
            Video(**self._video_kwargs(scenes), events=[event_a, event_b])

    def test_unknown_scene_reference_is_rejected(self) -> None:
        fps = 30.0
        scenes = [_scene("L01_V001", 0, 0, 30, fps)]
        foreign_scene = _scene("L01_V001", 1, 30, 60, fps)
        event = build_event("L01_V001", 0, [foreign_scene], "event-v1", _provenance("event-grouping"))
        with self.assertRaisesRegex(ValueError, "references unknown scenes"):
            Video(**self._video_kwargs(scenes), events=[event])

    def test_valid_full_coverage_events_pass(self) -> None:
        fps = 30.0
        scenes = [_scene("L01_V001", i, i * 30, (i + 1) * 30, fps) for i in range(2)]
        event = build_event("L01_V001", 0, scenes, "event-v1", _provenance("event-grouping"))
        video = Video(**self._video_kwargs(scenes), events=[event])
        self.assertEqual(video.event_count, 1)


class FakeMedia:
    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(fps=10, frame_count=20, duration_sec=2, width=320, height=180, codec="fake", audio_present=False)

    def extract_frame(self, source: Path, frame_idx: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"frame-{frame_idx}".encode())


class EventGroupingPipelineIntegrationTests(unittest.TestCase):
    def test_every_scene_belongs_to_exactly_one_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw/videos/L01_V001.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake-video")
            settings = OfflineSettings(
                data_root=root, input_dir=source.parent, export_dir=root / "exports", state_dir=root / "state",
                gpu_url=None, gpu_api_key=None, timeout_sec=2, retries=2, scene_seconds=1,
                keyframes_per_scene=1, pipeline_version="test-v1", provider="mock",
            )
            run(OfflinePipeline(settings, media=FakeMedia(), provider=MockInferenceProvider()).run())
            scenes = [json.loads(x) for x in (root / "exports/scenes.jsonl").read_text(encoding="utf-8").splitlines()]
            events = [json.loads(x) for x in (root / "exports/events.jsonl").read_text(encoding="utf-8").splitlines()]
            all_scene_ids = {scene["scene_id"] for scene in scenes}
            covered_scene_ids = [scene_id for event in events for scene_id in event["scene_ids"]]
            self.assertEqual(sorted(covered_scene_ids), sorted(all_scene_ids))
            self.assertEqual(len(covered_scene_ids), len(set(covered_scene_ids)))


if __name__ == "__main__":
    unittest.main()
