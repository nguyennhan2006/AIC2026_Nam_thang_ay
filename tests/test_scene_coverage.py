"""SCENE-COVERAGE-01: scene phải lát kín video, quy ước interval nửa mở.

Bối cảnh đo được (docs/20_EXPERIMENT_LOG.md § SCENE-COVERAGE-01): export
L21_V001 chỉ phủ **78.6%** số frame — 84 gap, mất 8083 frame. Nguyên nhân:
119/336 scene bị quarantine vì "không có keyframe nào", do keyframe được
trích theo stride cố định ~123 frame nên scene ngắn hơn stride rơi hết qua
lưới (scene bị loại dài trung vị 61 frame).

Hệ quả trực tiếp: 5/35 bước TRAKE có frame gold không thuộc scene nào, nên
candidate tương ứng KHÔNG TỒN TẠI — không model nào ở tầng trên cứu được.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from scripts.check_scene_coverage import analyse


@dataclass
class FakeScene:
    scene_id: str
    start_frame: int
    end_frame_exclusive: int


class CoverageAnalysisTests(unittest.TestCase):
    def test_contiguous_scenes_have_no_gap(self) -> None:
        scenes = [FakeScene("s0", 0, 100), FakeScene("s1", 100, 250), FakeScene("s2", 250, 300)]
        report = analyse("V", scenes, frame_count=300)
        self.assertEqual(report["gaps"], [])
        self.assertEqual(report["overlaps"], [])
        self.assertIsNone(report["missing_head"])
        self.assertIsNone(report["missing_tail"])
        self.assertEqual(report["coverage_ratio"], 1.0)

    def test_gap_between_scenes_is_reported(self) -> None:
        scenes = [FakeScene("s0", 0, 100), FakeScene("s2", 150, 300)]
        report = analyse("V", scenes, frame_count=300)
        self.assertEqual(len(report["gaps"]), 1)
        self.assertEqual(report["gaps"][0]["start_frame"], 100)
        self.assertEqual(report["gaps"][0]["end_frame_exclusive"], 150)
        self.assertEqual(report["gaps"][0]["length"], 50)

    def test_end_frame_is_exclusive_so_touching_bounds_are_not_a_gap(self) -> None:
        """`[0,100)` rồi `[100,200)` là liền mạch — KHÔNG phải thiếu frame 100.

        Nhầm quy ước ở đây sinh ra gap giả ở mọi ranh giới scene.
        """

        report = analyse("V", [FakeScene("s0", 0, 100), FakeScene("s1", 100, 200)], 200)
        self.assertEqual(report["gaps"], [])

    def test_overlap_is_reported_separately_from_gap(self) -> None:
        report = analyse("V", [FakeScene("s0", 0, 120), FakeScene("s1", 100, 200)], 200)
        self.assertEqual(report["gaps"], [])
        self.assertEqual(len(report["overlaps"]), 1)
        self.assertEqual(report["overlaps"][0]["length"], 20)

    def test_missing_head_and_tail_are_reported(self) -> None:
        report = analyse("V", [FakeScene("s1", 50, 150)], 300)
        self.assertEqual(report["missing_head"]["length"], 50)
        self.assertEqual(report["missing_tail"]["length"], 150)

    def test_out_of_range_scene_is_flagged(self) -> None:
        report = analyse("V", [FakeScene("s0", 0, 400)], 300)
        self.assertEqual(len(report["out_of_range"]), 1)

    def test_empty_video_does_not_crash(self) -> None:
        report = analyse("V", [], 300)
        self.assertEqual(report["scene_count"], 0)


class RegressionGuardTests(unittest.TestCase):
    """Khoá lại bất biến mà export HIỆN TẠI đang vi phạm.

    Test này CỐ Ý mô tả trạng thái mong muốn chứ không mô tả trạng thái hiện
    tại — nó chạy trên fixture, không trên `storage/exports_l21`. Khi
    SCENE-COVERAGE-01 được sửa xong, thêm một test tương tự chạy trên export
    thật để không tái phát.
    """

    def test_scene_shorter_than_keyframe_stride_must_still_be_kept(self) -> None:
        # Scene 45 frame nằm giữa hai scene dài: nếu keyframe trích theo stride
        # 123 frame thì nó không có keyframe nào và bị loại -> sinh gap.
        scenes = [FakeScene("s0", 0, 9), FakeScene("s1", 9, 54), FakeScene("s2", 54, 200)]
        report = analyse("V", scenes, frame_count=200)
        self.assertEqual(report["gaps"], [], "scene ngắn bị loại sẽ để lại gap ở đây")
        self.assertEqual(report["coverage_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
