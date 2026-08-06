"""PR-07: KisSignature + safe-frame selection."""

from __future__ import annotations

import unittest

from online.domain.candidate import FrameEvidence, FrameQuality
from online.domain.models import SceneDocument
from online.services.kis import build_signature
from online.services.safe_frame import SafeFrameConfig, score_frames, select_safe_frame


def make_scene(
    scene_id: str,
    frames: list[tuple[int, str, float | None]],
    *,
    start: int = 0,
    end: int = 300,
) -> SceneDocument:
    keyframes = [
        FrameEvidence(
            keyframe_id=f"{scene_id}_F{idx:06d}",
            video_id="L01_V001",
            scene_id=scene_id,
            frame_idx=idx,
            timestamp_sec=idx / 30,
            image_path=f"processed/keyframes/L01_V001/frame_{idx:06d}.jpg",
            captions=[caption] if caption else [],
            quality=FrameQuality(sharpness=sharpness) if sharpness is not None else FrameQuality(),
        )
        for idx, caption, sharpness in frames
    ]
    return SceneDocument(
        scene_id=scene_id, video_id="L01_V001", scene_idx=0,
        start_frame=start, end_frame_exclusive=end, start_sec=0.0, end_sec=10.0,
        keyframes=keyframes,
    )


class SignatureTests(unittest.TestCase):
    def test_quoted_phrase_becomes_must_match_and_rare_cue(self) -> None:
        sig = build_signature('người đàn ông cầm biển ghi "cảnh báo sạt lở"')
        self.assertIn("cảnh báo sạt lở", sig.must_match)
        self.assertIn("cảnh báo sạt lở", sig.rare_cues)

    def test_numbers_and_proper_nouns_are_rare_cues(self) -> None:
        sig = build_signature("Buổi lễ tại UNESCO có 14 đại biểu tham dự")
        self.assertIn("14", sig.rare_cues)
        self.assertIn("UNESCO", sig.rare_cues)

    def test_lowercase_accented_words_are_not_proper_nouns(self) -> None:
        """Từ THƯỜNG mở đầu bằng nguyên âm có dấu không phải danh từ riêng.

        Dải Unicode `À-Ỹ` xen kẽ hoa và thường, nên lớp ký tự `[A-ZĐÀ-Ỹ]` khớp
        cả `đ à á ê ô`. Hậu quả đo được trên 36 truy vấn KIS: `rare_cues` chứa
        `đó, được, đường, đêm, đất, áo đen, đàn ông` — tức những từ phổ biến
        nhất, đúng ngược với ý nghĩa "khớp được thì gần như chắc đúng".
        """

        sig = build_signature(
            "Tìm cảnh người đàn ông mặc áo đen đứng trên đường ướt trong đêm"
        )
        self.assertEqual(sig.rare_cues, ())

    def test_proper_nouns_still_detected_next_to_lowercase_accents(self) -> None:
        sig = build_signature("Tìm cảnh ông Nguyễn Văn Nam phát biểu ở Hà Nội")
        self.assertIn("Nguyễn Văn Nam", sig.rare_cues)
        self.assertIn("Hà Nội", sig.rare_cues)
        # "ông" là chức danh viết thường, không được nuốt vào tên riêng.
        self.assertNotIn("ông Nguyễn Văn Nam", sig.rare_cues)

    def test_negative_constraint_is_extracted_and_excluded_from_must_match(self) -> None:
        sig = build_signature("cảnh có xe máy, không có ô tô")
        self.assertTrue(sig.negative)
        must_normalized = " ".join(sig.must_match)
        self.assertNotIn("to", must_normalized.split())

    def test_coverage_scores_full_when_all_must_match_terms_present(self) -> None:
        sig = build_signature('biển ghi "cảnh báo sạt lở"')
        must, _nice, contradicted = sig.coverage("Biển ghi cảnh báo sạt lở nguy hiểm")
        self.assertEqual(must, 1.0)
        self.assertFalse(contradicted)

    def test_coverage_is_partial_when_a_must_match_term_is_missing(self) -> None:
        sig = build_signature('biển ghi "cảnh báo sạt lở"')
        must, _nice, _contradicted = sig.coverage("một tấm biển bình thường")
        self.assertLess(must, 1.0)

    def test_negative_constraint_present_flags_contradiction(self) -> None:
        sig = build_signature("cảnh có xe máy, không có ô tô")
        _must, _nice, contradicted = sig.coverage("đường phố có ô tô và xe máy")
        self.assertTrue(contradicted)


class SafeFrameTests(unittest.TestCase):
    def test_semantic_match_outranks_a_frame_with_no_text(self) -> None:
        scene = make_scene("L01_V001_S0000", [
            (50, "người cào muối trên cánh đồng", None),
            (100, "", None),
        ])
        scored = score_frames(scene, "cào muối")
        self.assertEqual(scored[0].frame.frame_idx, 50)

    def test_blurry_frame_is_penalized_below_a_sharper_one(self) -> None:
        scene = make_scene("L01_V001_S0000", [
            (50, "người cào muối", 10.0),   # rất mờ
            (100, "người cào muối", 200.0),  # nét
        ])
        scored = score_frames(scene, "cào muối")
        self.assertEqual(scored[0].frame.frame_idx, 100)
        self.assertGreater(scored[0].total, scored[1].total)

    def test_frame_near_scene_boundary_is_penalized(self) -> None:
        scene = make_scene(
            "L01_V001_S0000",
            [(1, "cào muối", None), (150, "cào muối", None)],
            start=0, end=300,
        )
        config = SafeFrameConfig(boundary_margin_frames=20)
        scored = score_frames(scene, "cào muối", config)
        # Cả hai khớp semantic như nhau; frame giữa scene phải thắng vì ít bị
        # phạt biên + centrality cao hơn.
        self.assertEqual(scored[0].frame.frame_idx, 150)

    def test_missing_quality_signal_is_neutral_not_zero(self) -> None:
        scene = make_scene("L01_V001_S0000", [(150, "cào muối", None)])
        scored = score_frames(scene, "cào muối")
        self.assertGreater(scored[0].quality, 0.0)

    def test_prefer_frame_idx_is_honored_when_not_penalized(self) -> None:
        scene = make_scene("L01_V001_S0000", [
            (50, "cào muối", 200.0), (150, "cào muối", 200.0),
        ])
        best = select_safe_frame(scene, "cào muối", prefer_frame_idx=50)
        self.assertEqual(best.frame.frame_idx, 50)

    def test_prefer_frame_idx_is_ignored_when_blurry(self) -> None:
        scene = make_scene("L01_V001_S0000", [
            (50, "cào muối", 5.0),   # rất mờ
            (150, "cào muối", 200.0),
        ])
        best = select_safe_frame(scene, "cào muối", prefer_frame_idx=50)
        self.assertEqual(best.frame.frame_idx, 150)

    def test_empty_scene_returns_none(self) -> None:
        scene = make_scene("L01_V001_S0000", [])
        self.assertIsNone(select_safe_frame(scene, "cào muối"))

    def test_tie_break_is_deterministic_by_frame_idx(self) -> None:
        scene = make_scene("L01_V001_S0000", [(200, "", None), (100, "", None)])
        first = score_frames(scene, "")
        second = score_frames(scene, "")
        self.assertEqual([item.frame.frame_idx for item in first],
                         [item.frame.frame_idx for item in second])


if __name__ == "__main__":
    unittest.main()
