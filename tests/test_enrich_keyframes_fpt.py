"""scripts/enrich_keyframes_fpt.py — parsing thuần (không gọi FPT thật).

VLM 7B không luôn tuân schema (đã thấy thật: OCR trả về chuỗi trần thay vì
{"text":..., "bbox_2d":...}) — `_extract_json`/`_normalize_bbox` phải xử lý
được JSON kèm markdown fence/prose, và loại bbox suy biến mà KHÔNG bịa lại
giá trị hợp lý (đúng nguyên tắc "không suy đoán" của repo).
"""

from __future__ import annotations

import unittest

from scripts.enrich_keyframes_fpt import _extract_json, _normalize_bbox


class ExtractJsonTests(unittest.TestCase):
    def test_plain_json_object(self) -> None:
        self.assertEqual(_extract_json('{"caption": "a"}'), {"caption": "a"})

    def test_markdown_fenced_json(self) -> None:
        text = '```json\n{"caption": "a"}\n```'
        self.assertEqual(_extract_json(text), {"caption": "a"})

    def test_json_embedded_in_prose(self) -> None:
        text = 'Đây là kết quả: {"caption": "a", "objects": []} Xong.'
        self.assertEqual(_extract_json(text), {"caption": "a", "objects": []})

    def test_malformed_json_returns_none(self) -> None:
        self.assertIsNone(_extract_json("không phải JSON gì cả"))

    def test_json_array_top_level_returns_none(self) -> None:
        # Contract yêu cầu object ở top-level (caption/ocr_instances/objects
        # là field của MỘT object) — mảng trần không đúng shape mong đợi.
        self.assertIsNone(_extract_json("[1, 2, 3]"))


class NormalizeBboxTests(unittest.TestCase):
    def test_valid_pixel_bbox_normalizes_to_unit_range(self) -> None:
        result = _normalize_bbox([100, 50, 300, 150], width=1000, height=500)
        self.assertEqual(result, {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3})

    def test_zero_width_or_height_returns_none(self) -> None:
        self.assertIsNone(_normalize_bbox([100, 50, 100, 150], width=1000, height=500))
        self.assertIsNone(_normalize_bbox([100, 50, 300, 50], width=1000, height=500))

    def test_swapped_corners_are_recovered_by_sorting_not_rejected(self) -> None:
        # Model đôi khi liệt kê góc theo thứ tự ngược (x2,y2 trước x1,y1) —
        # đây vẫn là 4 số THẬT, chỉ cần sắp lại min/max, không phải bịa dữ
        # liệu mới nên không cần loại bỏ như box thật sự suy biến.
        result = _normalize_bbox([300, 150, 100, 50], width=1000, height=500)
        self.assertEqual(result, {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3})

    def test_out_of_frame_values_are_clamped_not_rejected(self) -> None:
        # Model đôi khi ước lượng hơi lố khỏi khung hình — clamp về [0,1] thay
        # vì loại bỏ toàn bộ box, miễn sau khi clamp vẫn còn diện tích dương.
        result = _normalize_bbox([-50, -20, 1200, 600], width=1000, height=500)
        self.assertEqual(result, {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})

    def test_wrong_length_returns_none(self) -> None:
        self.assertIsNone(_normalize_bbox([1, 2, 3], width=1000, height=500))

    def test_non_list_returns_none(self) -> None:
        self.assertIsNone(_normalize_bbox("not-a-list", width=1000, height=500))

    def test_zero_dimension_returns_none(self) -> None:
        self.assertIsNone(_normalize_bbox([1, 2, 3, 4], width=0, height=500))


if __name__ == "__main__":
    unittest.main()
