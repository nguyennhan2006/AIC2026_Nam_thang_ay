"""Dung sai TRAKE phải tính bằng GIÂY và quy đổi bằng FPS thật.

File gold ghi cửa sổ ±4 frame quanh mốc ngữ nghĩa, còn luật chấm chấp nhận
lệch 3–6 giây tuỳ độ dài scene. Chấm bằng ±4 frame cho r_score = 0.000 giả
tạo — đã từng dẫn tới một chẩn đoán sai (xem docs/20_EXPERIMENT_LOG.md).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_tasks import Interval, load_fps


class ToleranceTests(unittest.TestCase):
    def test_zero_tolerance_keeps_the_narrow_gold_window(self) -> None:
        interval = Interval(start_frame=5696, end_frame=5704)
        self.assertTrue(interval.contains(5700))
        self.assertFalse(interval.contains(5750))

    def test_tolerance_widens_both_sides(self) -> None:
        interval = Interval(start_frame=5696, end_frame=5704)
        # ±3s @30fps = ±90 frame
        self.assertTrue(interval.contains(5790, tolerance_frames=90))
        self.assertTrue(interval.contains(5610, tolerance_frames=90))
        self.assertFalse(interval.contains(5800, tolerance_frames=90))


class FpsTests(unittest.TestCase):
    def test_fps_is_read_from_videos_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "videos.jsonl").write_text(
                json.dumps({"video_id": "L21_V001", "fps": 25.0}) + "\n", encoding="utf-8"
            )
            self.assertEqual(load_fps(root / "scenes.jsonl"), 25.0)

    def test_missing_file_falls_back_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_fps(Path(tmp) / "scenes.jsonl", default=29.97), 29.97)


if __name__ == "__main__":
    unittest.main()
