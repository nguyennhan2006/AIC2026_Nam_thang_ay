"""Cửa sổ phát video cho kết quả tìm kiếm.

Trước đây UI không phát được đoạn của kết quả vì không result nào mang ĐỦ
thông tin: KIS/QA chỉ có giây khi `evidence` đi kèm, TRAKE/AVS không có giây
nào, và không cái nào mang đường dẫn video.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from online.domain.models import SceneDocument
from online.services.playback import DEFAULT_PAD_SEC, build_window


def scene(**overrides) -> SceneDocument:
    payload = {
        "scene_id": "L21_V001_S0015", "video_id": "L21_V001",
        "video_path": "raw/videos/L21_V001.mp4", "scene_idx": 15,
        "start_frame": 2036, "end_frame_exclusive": 2131,
        "start_sec": 67.867, "end_sec": 71.033,
    }
    payload.update(overrides)
    return SceneDocument(**payload)


class BuildWindowTests(unittest.TestCase):
    def test_window_is_padded_on_both_sides(self) -> None:
        """Scene p50 chỉ 4.1s — xem đúng 4 giây không đủ hiểu bối cảnh."""

        window = build_window(scene(), pad_sec=5.0)
        self.assertIsNotNone(window)
        self.assertAlmostEqual(window.start_sec, 62.867, places=2)
        self.assertAlmostEqual(window.end_sec, 76.033, places=2)
        self.assertGreater(window.end_sec - window.start_sec, 13.0)

    def test_focus_marks_the_submitted_frame_not_the_padded_start(self) -> None:
        """UI phải nhảy tới frame được nộp; phần nới chỉ là bối cảnh."""

        window = build_window(scene(), focus_frame=2130, pad_sec=5.0)
        self.assertAlmostEqual(window.focus_sec, 71.0, places=1)
        self.assertLess(window.start_sec, window.focus_sec)

    def test_start_never_goes_below_zero(self) -> None:
        window = build_window(scene(start_sec=1.0, end_sec=3.0), pad_sec=5.0)
        self.assertEqual(window.start_sec, 0.0)

    def test_explicit_frame_range_wins_over_scene_bounds(self) -> None:
        """AVS và TRAKE có khoảng riêng, có thể rộng hơn một scene."""

        window = build_window(
            scene(), start_frame=2036, end_frame=2100, pad_sec=0.0
        )
        self.assertAlmostEqual(window.start_sec, 67.867, places=2)
        self.assertLess(window.end_sec, 71.033)

    def test_missing_video_path_yields_no_window(self) -> None:
        self.assertIsNone(build_window(scene(video_path=None)))

    def test_missing_media_file_yields_no_window(self) -> None:
        """`videos.jsonl` khai `source_path` cho cả ba video nhưng chỉ V001 có
        file mp4 thật. Trả URL 404 làm người dùng tưởng player hỏng — phải nói
        thẳng là chưa có video."""

        window = build_window(
            scene(video_path="raw/videos/KHONG_TON_TAI.mp4"),
            media_root=Path("storage"),
        )
        self.assertIsNone(window)

    def test_existing_media_file_yields_a_window(self) -> None:
        if not Path("storage/raw/videos/L21_V001.mp4").is_file():
            self.skipTest("thiếu storage/raw/videos/L21_V001.mp4")
        window = build_window(scene(), media_root=Path("storage"))
        self.assertIsNotNone(window)
        # Đường dẫn TƯƠNG ĐỐI, không kèm `/v1/media/`. Bake prefix vào đây
        # làm client ghép thành `/v1/media/%2Fv1%2Fmedia%2F...` -> HTTP 400.
        self.assertEqual(window.media_path, "raw/videos/L21_V001.mp4")

    def test_window_is_clamped_to_video_duration(self) -> None:
        window = build_window(scene(), pad_sec=5.0, video_duration_sec=72.0)
        self.assertLessEqual(window.end_sec, 72.0)

    def test_default_pad_is_five_seconds(self) -> None:
        self.assertEqual(DEFAULT_PAD_SEC, 5.0)


if __name__ == "__main__":
    unittest.main()
