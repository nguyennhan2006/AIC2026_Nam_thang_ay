"""PR-07: TRAKE video-first, đúng luật "sai video = 0".

Trước PR-07 toàn bộ TRAKE là 41 dòng nối scene bằng scene_idx, không hề khóa
video trước và không tinh chỉnh frame. Các test này khóa lại ba giai đoạn.
"""

from __future__ import annotations

import unittest

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument, SearchHit
from online.domain.search_config import TemporalOptions
from online.services.trake import TrakeProcessor, trake_processor_for_request
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

    def test_video_below_min_coverage_is_kept_as_degraded_fallback(self) -> None:
        # PR-14A: khi KHÔNG video nào đạt ngưỡng nhưng có bằng chứng thật (ở
        # đây chỉ 1/4 step), giữ lại ứng viên tốt nhất thay vì trả rỗng hoàn
        # toàn — một sequence suy yếu vẫn hơn "TRAKE luôn 0" (xem docstring
        # rank_videos()). Đánh dấu below_min_coverage=True để tầng trên biết.
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [], [], []]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0].video_id, "L21_V001")
        self.assertTrue(videos[0].below_min_coverage)

    def test_video_meeting_coverage_is_not_marked_degraded(self) -> None:
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [hit("L21_V001", 1, 200, 0.9)]]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        self.assertFalse(videos[0].below_min_coverage)

    def test_better_alternative_still_excludes_the_low_coverage_one(self) -> None:
        # Có video khác đạt ngưỡng đầy đủ -> KHÔNG cần fallback, video thiếu
        # coverage vẫn bị loại như cũ (chỉ fallback khi không còn lựa chọn nào).
        step_hits = [
            [hit("L21_V001", 0, 100, 0.5), hit("L21_V002", 0, 999, 0.99)],
            [hit("L21_V001", 1, 200, 0.5)],
            [hit("L21_V001", 2, 300, 0.5)],
        ]
        videos = rank_videos(step_hits, VideoRetrieverConfig(min_step_coverage=0.5))
        self.assertEqual([item.video_id for item in videos], ["L21_V001"])

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

    def test_reversed_evidence_cannot_form_a_path_only_the_forward_one_does(self) -> None:
        # E1 chỉ có bằng chứng ở frame 300, E2 chỉ có ở frame 100 (trước E1) —
        # một chain đúng thứ tự không thể tồn tại; hệ thống phải rơi về
        # "bỏ qua step" (khi cho phép) thay vì tạo path đảo thứ tự.
        step_hits = [[hit("L21_V001", 0, 300, 0.9)], [hit("L21_V001", 1, 100, 0.9)]]
        hypotheses = search_sequences(
            "L21_V001", step_hits, SequenceConfig(allow_missing_steps=True)
        )
        for hypothesis in hypotheses:
            frames = hypothesis.frame_ids
            self.assertTrue(all(b > a for a, b in zip(frames, frames[1:])))

    def test_complete_chain_beats_chain_with_a_missing_step(self) -> None:
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9)],
            [hit("L21_V001", 1, 200, 0.9)],
            [hit("L21_V001", 2, 300, 0.9)],
        ]
        hypotheses = search_sequences(
            "L21_V001", step_hits, SequenceConfig(allow_missing_steps=True)
        )
        best = hypotheses[0]
        self.assertEqual(best.covered, 3)
        self.assertTrue(all(h.score <= best.score for h in hypotheses))

    def test_two_steps_can_land_in_the_same_scene_if_frames_differ(self) -> None:
        # Scene dài có thể chứa hai khoảnh khắc khác nhau — sequence_search so
        # theo frame_idx chứ không theo scene_id (đúng docstring module).
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9, scene_id="L21_V001_S0000")],
            [hit("L21_V001", 0, 105, 0.9, scene_id="L21_V001_S0000")],
        ]
        hypotheses = search_sequences(
            "L21_V001", step_hits, SequenceConfig(allow_missing_steps=False)
        )
        self.assertTrue(hypotheses)
        self.assertEqual(hypotheses[0].frame_ids, [100, 105])

    def test_larger_gap_scores_lower_than_smaller_gap_within_allowed_range(self) -> None:
        config = SequenceConfig(allow_missing_steps=False, max_gap_sec=600.0)
        close = search_sequences(
            "L21_V001",
            [[hit("L21_V001", 0, 100, 0.9)], [hit("L21_V001", 1, 130, 0.9)]],
            config,
        )[0]
        far = search_sequences(
            "L21_V001",
            [[hit("L21_V001", 0, 100, 0.9)], [hit("L21_V001", 1, 100 + 30 * 200, 0.9)]],
            config,
        )[0]
        self.assertGreater(close.score, far.score)

    def test_deterministic_tie_break_prefers_smaller_frame_ids(self) -> None:
        # Hai candidate cùng score cho step 2 -> tie-break phải ổn định (chọn
        # frame nhỏ hơn), không phụ thuộc thứ tự ngẫu nhiên của input.
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9)],
            [hit("L21_V001", 1, 250, 0.5), hit("L21_V001", 1, 200, 0.5)],
        ]
        first = search_sequences("L21_V001", step_hits, SequenceConfig(allow_missing_steps=False))
        step_hits_reordered = [
            [hit("L21_V001", 0, 100, 0.9)],
            [hit("L21_V001", 1, 200, 0.5), hit("L21_V001", 1, 250, 0.5)],
        ]
        second = search_sequences(
            "L21_V001", step_hits_reordered, SequenceConfig(allow_missing_steps=False)
        )
        self.assertEqual(first[0].frame_ids, second[0].frame_ids)
        self.assertEqual(first[0].frame_ids, [100, 200])


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
        # UI competition studio: step card cần thumbnail/timestamp thật, không
        # đoán path hay suy ra từ fps (không có endpoint expose fps).
        self.assertEqual(step.image_path, "f.jpg")
        self.assertAlmostEqual(step.timestamp_sec, 110 / 30)

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

    def test_missing_step_is_reported_not_silently_dropped(self) -> None:
        # step 2 (giữa) không có bằng chứng -> chain vẫn phải khác rỗng
        # (partial chain), và output phải ghi rõ step nào thiếu (PR-14A Gate C).
        step_hits = [
            [hit("L21_V001", 0, 100, 0.9)],
            [],
            [hit("L21_V001", 2, 300, 0.9)],
        ]
        documents = {
            f"L21_V001_S{i:04d}": self._document("L21_V001", i, frame)
            for i, frame in zip((0, 2), (100, 300))
        }
        results = TrakeProcessor().run(["a", "b", "c"], step_hits, documents)
        self.assertTrue(results)
        self.assertEqual(results[0].missing_steps, [2])

    def test_only_one_video_below_threshold_is_marked_degraded_not_dropped(self) -> None:
        # Chỉ 1/4 step có bằng chứng, đúng ngưỡng min_step_coverage mặc định
        # (0.5) sẽ loại video này -> rank_videos fallback giữ lại (below_min_
        # coverage=True) thay vì trả rỗng hoàn toàn; TrakeProcessor phải
        # truyền cờ degraded đó ra tới kết quả cuối.
        step_hits = [[hit("L21_V001", 0, 100, 0.9)], [], [], []]
        documents = {"L21_V001_S0000": self._document("L21_V001", 0, 100)}
        results = TrakeProcessor().run(["a", "b", "c", "d"], step_hits, documents)
        self.assertTrue(results)
        self.assertTrue(results[0].degraded)


class TrakeProcessorForRequestTests(unittest.TestCase):
    """UI competition studio: TRAKE Alignment sliders phải override đúng tham
    số đã có sẵn trong VideoRetrieverConfig/SequenceConfig, không đổi gì khác,
    và không đổi hành vi khi request không đặt search_options (PR-15)."""

    def test_no_explicit_fields_returns_the_same_base_instance(self) -> None:
        base = TrakeProcessor()
        result = trake_processor_for_request(base, TemporalOptions())
        self.assertIs(result, base)

    def test_order_weight_overrides_video_config_only(self) -> None:
        base = TrakeProcessor()
        options = TemporalOptions(order_weight=0.9)
        result = trake_processor_for_request(base, options)
        self.assertEqual(result.video_config.ordering_weight, 0.9)
        self.assertEqual(result.sequence_config, base.sequence_config)
        self.assertIsNot(result, base)

    def test_gap_penalty_and_missing_step_penalty_override_sequence_config(self) -> None:
        base = TrakeProcessor()
        options = TemporalOptions(gap_penalty_per_sec=0.01, missing_step_penalty=0.9)
        result = trake_processor_for_request(base, options)
        self.assertEqual(result.sequence_config.gap_penalty_per_sec, 0.01)
        self.assertEqual(result.sequence_config.missing_step_penalty, 0.9)
        self.assertEqual(result.video_config, base.video_config)

    def test_maximum_gap_and_allow_missing_step_are_overridable_even_at_default_value(self) -> None:
        # Đây là field bool/threshold (không phải Optional) — chỉ ghi đè khi
        # caller đặt tường minh (model_fields_set), kể cả khi giá trị trùng
        # default, để phân biệt "không gửi" với "gửi đúng giá trị mặc định".
        base = TrakeProcessor()
        options = TemporalOptions(maximum_gap_sec=45.0, allow_missing_optional_step=True)
        result = trake_processor_for_request(base, options)
        self.assertEqual(result.sequence_config.max_gap_sec, 45.0)
        self.assertTrue(result.sequence_config.allow_missing_steps)

    def test_refinement_config_is_never_touched(self) -> None:
        base = TrakeProcessor(refinement_config=RefinementConfig(window_frames=99))
        result = trake_processor_for_request(base, TemporalOptions(order_weight=0.1))
        self.assertIs(result.refinement_config, base.refinement_config)


if __name__ == "__main__":
    unittest.main()
