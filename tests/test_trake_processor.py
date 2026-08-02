"""PR-07: TRAKE video-first, đúng luật "sai video = 0".

Trước PR-07 toàn bộ TRAKE là 41 dòng nối scene bằng scene_idx, không hề khóa
video trước và không tinh chỉnh frame. Các test này khóa lại ba giai đoạn.
"""

from __future__ import annotations

import unittest

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument, SearchHit
from online.services.trake import TrakeProcessor
from online.services.trake.frame_refinement import RefinementConfig, refine_step
from online.services.trake.sequence_search import (
    SequenceConfig,
    local_variants,
    search_sequences,
)
from online.services.trake.video_retriever import VideoRetrieverConfig, rank_videos


def hit(
    video: str, scene_idx: int, frame_idx: int, score: float, *, scene_id: str | None = None
) -> SearchHit:
    sid = scene_id or f"{video}_S{scene_idx:04d}"
    return SearchHit(
        candidate_id=sid, scene_id=sid, video_id=video, scene_idx=scene_idx,
        start_frame=frame_idx - 10, end_frame_exclusive=frame_idx + 10,
        start_sec=frame_idx / 30 - 0.3, end_sec=frame_idx / 30 + 0.3,
        best_frame_idx=frame_idx, best_timestamp_sec=frame_idx / 30, score=score,
    )


class VideoRetrieverTests(unittest.TestCase):
    def test_video_with_full_step_coverage_outranks_partial(self) -> None:
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9), hit("L21_V002", 0, 50, 0.8)],
            [hit("L21_V001", 1, 200, 0.9)],  # chỉ V001 có bằng chứng cho step 2
        ]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.4))
        self.assertEqual(videos[0].video_id, "L21_V001")
        self.assertEqual(videos[0].step_coverage, 1.0)

    def test_video_below_min_coverage_is_dropped(self) -> None:
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [], [], []]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        self.assertEqual(videos, [])

    def test_ordered_pairs_score_higher_than_reversed_ones(self) -> None:
        # Cả hai step đều có bằng chứng ở cả hai video; chỉ khác nhau thứ tự
        # frame: V001 tăng dần (100 -> 200), V002 giảm dần (200 -> 100).
        step_hits = [
            [hit("L21_V001", 0, 100, 0.8), hit("L21_V002", 0, 200, 0.8)],
            [hit("L21_V001", 1, 200, 0.8), hit("L21_V002", 1, 100, 0.8)],
        ]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        by_id = {item.video_id: item for item in videos}
        self.assertEqual(by_id["L21_V001"].ordered_pair_coverage, 1.0)
        self.assertEqual(by_id["L21_V002"].ordered_pair_coverage, 0.0)

    def test_all_steps_in_one_scene_are_penalized_as_duplicate(self) -> None:
        step_hits = [
            [hit("L21_V001", 0, 100, 0.8, scene_id="L21_V001_S0000")],
            [hit("L21_V001", 0, 105, 0.8, scene_id="L21_V001_S0000")],
        ]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        self.assertGreater(videos[0].duplicate_penalty, 0.0)


class SequenceSearchTests(unittest.TestCase):
    def test_only_hits_from_the_locked_video_are_considered(self) -> None:
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9), hit("L21_V002", 0, 90, 0.95)],
            [hit("L21_V001", 1, 200, 0.9)],
        ]
        hypotheses = search_sequences("L21_V001", step_hits)
        self.assertTrue(all(
            h.video_id == "L21_V001" for hypothesis in hypotheses for h in hypothesis.hits if h
        ))

    def test_frames_must_strictly_increase(self) -> None:
        step_hits = [[hit("L21_V001", 0, 200, 0.9)], [hit("L21_V001", 1, 100, 0.9)]]
        hypotheses = search_sequences(
            "L21_V001", step_hits, SequenceConfig(allow_missing_steps=False)
        )
        self.assertEqual(hypotheses, [])

    def test_gap_beyond_max_is_rejected(self) -> None:
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [hit("L21_V001", 1, 100 + 30 * 1000, 0.9)]]
        hypotheses = search_sequences(
            "L21_V001", step_hits,
            SequenceConfig(max_gap_sec=60.0, allow_missing_steps=False),
        )
        self.assertEqual(hypotheses, [])

    def test_missing_step_is_allowed_when_configured(self) -> None:
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [], [hit("L21_V001", 2, 300, 0.9)]]
        hypotheses = search_sequences(
            "L21_V001", step_hits, SequenceConfig(allow_missing_steps=True)
        )
        self.assertTrue(hypotheses)
        self.assertEqual(hypotheses[0].covered, 2)

    def test_local_variants_shift_by_one_frame_and_stay_increasing(self) -> None:
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [hit("L21_V001", 1, 200, 0.9)]]
        hypotheses = search_sequences("L21_V001", step_hits)
        variants = local_variants(hypotheses[0])
        for variant in variants:
            self.assertTrue(all(b > a for a, b in zip(variant, variant[1:])))


class FrameRefinementTests(unittest.TestCase):
    def _scene(self, frames: list[tuple[int, str]]) -> SceneDocument:
        keyframes = [
            FrameEvidence(
                keyframe_id=f"L21_V001_S0000_F{idx:06d}", video_id="L21_V001",
                scene_id="L21_V001_S0000", frame_idx=idx, timestamp_sec=idx / 30,
                image_path="f.jpg", captions=[caption] if caption else [],
            )
            for idx, caption in frames
        ]
        return SceneDocument(
            scene_id="L21_V001_S0000", video_id="L21_V001", scene_idx=0,
            start_frame=0, end_frame_exclusive=300, start_sec=0.0, end_sec=10.0,
            keyframes=keyframes,
        )

    def test_refinement_picks_the_best_matching_frame_in_the_window(self) -> None:
        scene = self._scene([(100, "cột nước phun cao"), (110, "người đàn ông tiến sát")])
        step = refine_step(2, "người đàn ông tiến sát", scene, anchor_frame_idx=100)
        self.assertEqual(step.frame_idx, 110)

    def test_anchor_outside_scene_keeps_the_anchor(self) -> None:
        scene = self._scene([(100, "cột nước")])
        step = refine_step(1, "cột nước", scene, anchor_frame_idx=5000)
        self.assertEqual(step.frame_idx, 5000)

    def test_extra_frames_mark_dense_window_refinement(self) -> None:
        scene = self._scene([(100, "cột nước")])
        extra = FrameEvidence(
            keyframe_id="extra", video_id="L21_V001", scene_id="L21_V001_S0000",
            frame_idx=105, timestamp_sec=105 / 30, image_path="extra.jpg",
            captions=["người đàn ông tiến sát"],
        )
        step = refine_step(
            2, "người đàn ông tiến sát", scene, anchor_frame_idx=100,
            extra_frames=[extra], config=RefinementConfig(window_frames=20),
        )
        self.assertEqual(step.refinement, "dense_window")
        self.assertEqual(step.frame_idx, 105)


class TrakeProcessorTests(unittest.TestCase):
    def _document(self, video: str, scene_idx: int, frame_idx: int) -> SceneDocument:
        sid = f"{video}_S{scene_idx:04d}"
        return SceneDocument(
            scene_id=sid, video_id=video, scene_idx=scene_idx,
            start_frame=frame_idx - 10, end_frame_exclusive=frame_idx + 10,
            start_sec=(frame_idx - 10) / 30, end_sec=(frame_idx + 10) / 30,
            keyframes=[FrameEvidence(
                keyframe_id=f"{sid}_F{frame_idx:06d}", video_id=video, scene_id=sid,
                frame_idx=frame_idx, timestamp_sec=frame_idx / 30, image_path="f.jpg",
            )],
        )

    def test_end_to_end_produces_strictly_increasing_frame_ids(self) -> None:
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9)],
            [hit("L21_V001", 1, 200, 0.85)],
            [hit("L21_V001", 2, 300, 0.8)],
        ]
        documents = {
            f"L21_V001_S{i:04d}": self._document("L21_V001", i, frame)
            for i, frame in enumerate((100, 200, 300))
        }
        results = TrakeProcessor().run(["a", "b", "c"], step_hits, documents)
        self.assertTrue(results)
        self.assertEqual(results[0].video_id, "L21_V001")
        frames = results[0].frame_ids
        self.assertTrue(all(b > a for a, b in zip(frames, frames[1:])))

    def test_video_with_low_step_coverage_is_excluded_even_with_high_scores(self) -> None:
        # L21_V002 chỉ phủ 1/3 step (dù điểm rất cao) -> dưới ngưỡng
        # min_step_coverage mặc định (0.5) -> Stage A loại nó, dù điểm fusion
        # từng bước một cao hơn hẳn L21_V001.
        step_hits = [
            [hit("L21_V001", 0, 100, 0.5), hit("L21_V002", 0, 999, 0.99)],
            [hit("L21_V001", 1, 200, 0.5)],
            [hit("L21_V001", 2, 300, 0.5)],
        ]
        documents = {
            f"L21_V001_S{i:04d}": self._document("L21_V001", i, frame)
            for i, frame in enumerate((100, 200, 300))
        }
        results = TrakeProcessor().run(["a", "b", "c"], step_hits, documents)
        self.assertTrue(all(item.video_id == "L21_V001" for item in results))

    def test_no_coverage_anywhere_returns_empty(self) -> None:
        results = TrakeProcessor().run(["a", "b"], [[], []], {})
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
