"""W1 (offline feature #1 — clip pooling): sliding-window embedding pooling.

Uses synthetic Scene/Keyframe fixtures and a fake in-memory EmbeddingReader —
no ffmpeg/model dependency, matching the pattern in test_color_features.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from datasection.schemas import Keyframe, ModelProvenance, Scene
from offline.clip_pooling import (
    build_clip_segment,
    build_clip_windows,
    choose_representative_frame,
    pool_embeddings,
    select_clip_frames,
)
from offline.embedding_reader import MemoizedEmbeddingReader


def run(coro):
    return asyncio.run(coro)


def _provenance(name: str) -> ModelProvenance:
    return ModelProvenance(model_name=name, model_revision="test", pipeline_version="test-v1")


def _keyframe(scene_id: str, video_id: str, frame_idx: int, fps: float, selection_score: float | None = None) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{scene_id}_F{frame_idx:06d}", video_id=video_id, scene_id=scene_id,
        frame_idx=frame_idx, timestamp_sec=frame_idx / fps, image_path=f"processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg",
        width=64, height=64, roles=["representative"], selection_score=selection_score,
    )


def _scene(video_id: str, scene_idx: int, start_frame: int, end_frame: int, fps: float, keyframes: list[Keyframe]) -> Scene:
    return Scene(
        scene_id=f"{video_id}_S{scene_idx:04d}", video_id=video_id, scene_idx=scene_idx,
        start_frame=start_frame, end_frame_exclusive=end_frame,
        start_sec=start_frame / fps, end_sec=end_frame / fps,
        segmentation_provenance=_provenance("seg"), keyframes=keyframes,
    )


class FakeEmbeddingReader:
    """Deterministic in-memory reader that records every keyframe it was asked to read."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    async def read(self, keyframe: Keyframe) -> list[float]:
        self.calls.append(keyframe.keyframe_id)
        return self.vectors[keyframe.keyframe_id]


class BuildClipWindowsTests(unittest.TestCase):
    def test_short_scene_yields_exactly_one_clip_spanning_the_whole_scene(self) -> None:
        fps = 30.0
        scene = _scene("L01_V001", 0, 0, 129, fps, [_keyframe("L01_V001_S0000", "L01_V001", 0, fps)])
        windows = build_clip_windows(scene, fps, duration_sec=4.0, stride_sec=4.0, min_tail_sec=1.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_frame, scene.start_frame)
        self.assertEqual(windows[0].end_frame, scene.end_frame_exclusive)

    def test_long_scene_yields_correct_stride_multiple_clips(self) -> None:
        fps = 30.0
        scene = _scene("L01_V001", 0, 0, 240, fps, [_keyframe("L01_V001_S0000", "L01_V001", 0, fps)])
        windows = build_clip_windows(scene, fps, duration_sec=4.0, stride_sec=2.0, min_tail_sec=1.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].start_frame, 0)
        self.assertEqual(windows[-1].end_frame, scene.end_frame_exclusive)
        for window in windows:
            self.assertGreater(window.end_frame, window.start_frame)
            self.assertLessEqual(window.end_frame, scene.end_frame_exclusive)
            self.assertGreaterEqual(window.start_frame, scene.start_frame)

    def test_short_trailing_window_is_merged_into_previous(self) -> None:
        fps = 30.0
        # duration=4, stride=4 (non-overlapping): second window would be (4, 4.3)s, a 0.3s
        # sliver below min_tail_sec=1 -> must be merged into the first window instead.
        scene = _scene("L01_V001", 0, 0, round(4.3 * fps), fps, [_keyframe("L01_V001_S0000", "L01_V001", 0, fps)])
        windows = build_clip_windows(scene, fps, duration_sec=4.0, stride_sec=4.0, min_tail_sec=1.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_frame, scene.start_frame)
        self.assertEqual(windows[0].end_frame, scene.end_frame_exclusive)

    def test_clip_windows_never_exceed_scene_boundary(self) -> None:
        fps = 25.0
        scene = _scene("L01_V001", 2, 500, 700, fps, [_keyframe("L01_V001_S0002", "L01_V001", 500, fps)])
        windows = build_clip_windows(scene, fps, duration_sec=3.0, stride_sec=1.5, min_tail_sec=0.5)
        for window in windows:
            self.assertGreaterEqual(window.start_frame, scene.start_frame)
            self.assertLessEqual(window.end_frame, scene.end_frame_exclusive)

    def test_rerun_produces_identical_deterministic_windows(self) -> None:
        fps = 30.0
        scene = _scene("L01_V001", 0, 0, 240, fps, [_keyframe("L01_V001_S0000", "L01_V001", 0, fps)])
        first = build_clip_windows(scene, fps, 4.0, 2.0, 1.0)
        second = build_clip_windows(scene, fps, 4.0, 2.0, 1.0)
        self.assertEqual(first, second)


class SelectClipFramesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fps = 30.0
        self.video_id = "L01_V001"
        self.scene_id = f"{self.video_id}_S0000"

    def test_in_window_keyframe_is_selected_without_degradation(self) -> None:
        keyframe = _keyframe(self.scene_id, self.video_id, 90, self.fps)  # t=3.0s
        scene = _scene(self.video_id, 0, 0, 240, self.fps, [keyframe])
        window = build_clip_windows(scene, self.fps, 4.0, 2.0, 1.0)[0]
        selected = select_clip_frames(scene, window, max_sampled_frames=4, fallback_max_distance_sec=6.0, empty_window_policy="nearest_scene_keyframe")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.sampling_method, "in_window")
        self.assertFalse(selected.sampling_degraded)
        self.assertIsNone(selected.fallback_distance_sec)
        self.assertEqual(selected.keyframes, [keyframe])

    def test_empty_window_falls_back_to_nearest_scene_keyframe(self) -> None:
        # Single keyframe at t=0.5s; scene spans 0-8s split into 4s windows, so the
        # second window (4s-8s) contains no keyframe and must fall back.
        keyframe = _keyframe(self.scene_id, self.video_id, 15, self.fps)  # t=0.5s
        scene = _scene(self.video_id, 0, 0, 240, self.fps, [keyframe])
        windows = build_clip_windows(scene, self.fps, 4.0, 4.0, 1.0)
        second_window = windows[1]
        selected = select_clip_frames(scene, second_window, max_sampled_frames=4, fallback_max_distance_sec=6.0, empty_window_policy="nearest_scene_keyframe")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.sampling_method, "nearest_scene_keyframe")
        self.assertTrue(selected.sampling_degraded)
        self.assertEqual(selected.keyframes, [keyframe])
        window_center = (second_window.start_sec + second_window.end_sec) / 2
        self.assertAlmostEqual(selected.fallback_distance_sec, abs(keyframe.timestamp_sec - window_center))

    def test_fallback_beyond_max_distance_returns_none(self) -> None:
        keyframe = _keyframe(self.scene_id, self.video_id, 15, self.fps)  # t=0.5s
        scene = _scene(self.video_id, 0, 0, 240, self.fps, [keyframe])
        second_window = build_clip_windows(scene, self.fps, 4.0, 4.0, 1.0)[1]
        selected = select_clip_frames(scene, second_window, max_sampled_frames=4, fallback_max_distance_sec=0.1, empty_window_policy="nearest_scene_keyframe")
        self.assertIsNone(selected)

    def test_skip_policy_never_falls_back(self) -> None:
        keyframe = _keyframe(self.scene_id, self.video_id, 15, self.fps)
        scene = _scene(self.video_id, 0, 0, 240, self.fps, [keyframe])
        second_window = build_clip_windows(scene, self.fps, 4.0, 4.0, 1.0)[1]
        selected = select_clip_frames(scene, second_window, max_sampled_frames=4, fallback_max_distance_sec=100.0, empty_window_policy="skip")
        self.assertIsNone(selected)


class PoolEmbeddingsTests(unittest.TestCase):
    def test_single_frame_pooled_vector_equals_normalized_frame_vector(self) -> None:
        pooled = pool_embeddings([[3.0, 4.0]])
        self.assertAlmostEqual(pooled[0], 0.6)
        self.assertAlmostEqual(pooled[1], 0.8)
        self.assertAlmostEqual(sum(x * x for x in pooled), 1.0)

    def test_pooled_vector_norm_is_one(self) -> None:
        pooled = pool_embeddings([[1.0, 2.0, 2.0], [3.0, -2.0, 6.0]])
        self.assertAlmostEqual(sum(x * x for x in pooled), 1.0, places=6)

    def test_zero_norm_vector_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero norm"):
            pool_embeddings([[1.0, 1.0], [-1.0, -1.0]])

    def test_empty_vector_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pool_embeddings([])


class ChooseRepresentativeFrameTests(unittest.TestCase):
    def test_closest_to_pooled_vector_wins(self) -> None:
        fps = 30.0
        near = _keyframe("L01_V001_S0000", "L01_V001", 60, fps)  # t=2.0s
        far = _keyframe("L01_V001_S0000", "L01_V001", 90, fps)  # t=3.0s
        pooled = [1.0, 0.0]
        vectors = [[0.99, 0.01], [0.1, 0.99]]
        chosen = choose_representative_frame([near, far], vectors, pooled, window_center_sec=2.5)
        self.assertEqual(chosen.keyframe_id, near.keyframe_id)


class BuildClipSegmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_vector_file_exists_before_reference_is_created(self) -> None:
        fps = 30.0
        video_id, scene_id = "L01_V001", "L01_V001_S0000"
        keyframe = _keyframe(scene_id, video_id, 0, fps)
        scene = _scene(video_id, 0, 0, 120, fps, [keyframe])
        window = build_clip_windows(scene, fps, 4.0, 4.0, 1.0)[0]
        reader = MemoizedEmbeddingReader(FakeEmbeddingReader({keyframe.keyframe_id: [3.0, 4.0]}))
        selected = select_clip_frames(scene, window, 4, 6.0, "nearest_scene_keyframe")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            clip = await build_clip_segment(
                scene, window, selected, reader, data_root, "clip-v1", _provenance("embedding"), "mock/aic-v1", "deterministic",
            )
            location = clip.embedding_refs[0].storage_locations[0]
            vector_path = data_root / location.vector_uri
            self.assertTrue(vector_path.exists())
            self.assertEqual(json.loads(vector_path.read_text()), [0.6, 0.8])
            self.assertEqual(clip.representative_frame_id, keyframe.keyframe_id)
            self.assertEqual(clip.sampled_frame_ids, [keyframe.keyframe_id])

    async def test_clip_id_is_boundary_derived_and_reruns_are_identical(self) -> None:
        fps = 30.0
        video_id, scene_id = "L01_V001", "L01_V001_S0000"
        keyframe = _keyframe(scene_id, video_id, 0, fps)
        scene = _scene(video_id, 0, 0, 120, fps, [keyframe])
        window = build_clip_windows(scene, fps, 4.0, 4.0, 1.0)[0]
        selected = select_clip_frames(scene, window, 4, 6.0, "nearest_scene_keyframe")
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            reader_a = MemoizedEmbeddingReader(FakeEmbeddingReader({keyframe.keyframe_id: [3.0, 4.0]}))
            reader_b = MemoizedEmbeddingReader(FakeEmbeddingReader({keyframe.keyframe_id: [3.0, 4.0]}))
            clip_a = await build_clip_segment(scene, window, selected, reader_a, data_root, "clip-v1", _provenance("embedding"), "mock/aic-v1", "deterministic")
            clip_b = await build_clip_segment(scene, window, selected, reader_b, data_root, "clip-v1", _provenance("embedding"), "mock/aic-v1", "deterministic")
            self.assertEqual(clip_a.clip_id, clip_b.clip_id)
            self.assertEqual(clip_a.clip_id, f"{scene_id}_C{window.start_frame:08d}_{window.end_frame:08d}")


class MemoizedEmbeddingReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_overlapping_windows_read_the_same_keyframe_only_once(self) -> None:
        fps = 30.0
        video_id, scene_id = "L01_V001", "L01_V001_S0000"
        keyframe = _keyframe(scene_id, video_id, 60, fps)  # t=2.0s, inside every overlapping window below
        scene = _scene(video_id, 0, 0, 240, fps, [keyframe])
        windows = build_clip_windows(scene, fps, duration_sec=4.0, stride_sec=2.0, min_tail_sec=1.0)
        fake = FakeEmbeddingReader({keyframe.keyframe_id: [1.0, 0.0]})
        reader = MemoizedEmbeddingReader(fake)
        for window in windows:
            selected = select_clip_frames(scene, window, 4, 6.0, "nearest_scene_keyframe")
            if selected is not None and selected.keyframes == [keyframe]:
                await reader.read(keyframe)
        self.assertEqual(fake.calls, [keyframe.keyframe_id])


class ClipSegmentDomainValidationTests(unittest.TestCase):
    """The pieces of the "empty window / boundary" contract enforced by the schema
    itself (not clip_pooling.py) — kept here to document the guarantee explicitly."""

    def test_video_rejects_clip_outside_its_scenes_frame_range(self) -> None:
        from datasection.schemas import ClipSegment, Video

        fps = 30.0
        video_id, scene_id = "L01_V001", "L01_V001_S0000"
        keyframe = _keyframe(scene_id, video_id, 0, fps)
        scene = _scene(video_id, 0, 0, 120, fps, [keyframe])
        bad_clip = ClipSegment(
            clip_id=f"{scene_id}_C00000000_00000200", video_id=video_id, scene_id=scene_id,
            start_sec=0.0, end_sec=6.0, duration_sec=6.0, start_frame=0, end_frame=200,
            sampled_frame_ids=[keyframe.keyframe_id], representative_frame_id=keyframe.keyframe_id,
            sampling_method="in_window", clip_config_id="clip-v1", provenance=_provenance("embedding"),
        )
        with self.assertRaisesRegex(ValueError, "exceeds its scene's frame range"):
            Video(
                video_id=video_id, source_path="raw/videos/L01_V001.mp4", fps=fps, frame_count=120,
                duration_sec=120 / fps, width=64, height=64, probe_provenance=_provenance("ffprobe"),
                scenes=[scene], clips=[bad_clip],
            )


if __name__ == "__main__":
    unittest.main()
