"""W1 (offline feature #1 — color): baseline HSV histogram + named dominant colors.

Không cần torch/transformers — _color_sync chỉ dùng PIL+numpy (thuần CPU), nên test
được trực tiếp mà không cần mock model nặng.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from offline.gpu_engine import TransformersGpuEngine
from offline.providers import MockInferenceProvider
from offline.worker import ImageRequest


def run(coro):
    return asyncio.run(coro)


def _image_request(image: Image.Image) -> ImageRequest:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ImageRequest(image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"))


class MockColorTaskTests(unittest.TestCase):
    def test_mock_provider_returns_empty_color_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            path.write_bytes(b"not-a-real-image")  # mock không đọc nội dung file
            result = run(MockInferenceProvider().image("color", path))
        self.assertEqual(result, {"dominant_colors": [], "hsv_histogram": [], "mean_hsv": None, "regions": {}})


class ColorSyncBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = TransformersGpuEngine()

    def test_solid_red_image_is_dominant_red_everywhere(self) -> None:
        image = Image.new("RGB", (60, 60), color=(220, 20, 20))
        result = self.engine._color_sync(_image_request(image))

        self.assertGreater(len(result["dominant_colors"]), 0)
        top = result["dominant_colors"][0]
        self.assertEqual(top["name"], "red")
        self.assertGreater(top["ratio"], 0.95)

        self.assertAlmostEqual(sum(result["hsv_histogram"]), 1.0, places=3)

        for region in ("upper", "center", "lower"):
            self.assertEqual(result["regions"][region], ["red"])

    def test_two_band_image_has_different_dominant_region_colors(self) -> None:
        image = Image.new("RGB", (60, 60), color=(0, 0, 0))
        top_half = Image.new("RGB", (60, 30), color=(220, 20, 20))  # red
        bottom_half = Image.new("RGB", (60, 30), color=(20, 20, 220))  # blue
        image.paste(top_half, (0, 0))
        image.paste(bottom_half, (0, 30))

        result = self.engine._color_sync(_image_request(image))
        self.assertEqual(result["regions"]["upper"], ["red"])
        self.assertEqual(result["regions"]["lower"], ["blue"])

    def test_grayscale_image_is_named_by_value_not_hue(self) -> None:
        white = Image.new("RGB", (40, 40), color=(250, 250, 250))
        result = self.engine._color_sync(_image_request(white))
        self.assertEqual(result["dominant_colors"][0]["name"], "white")

        black = Image.new("RGB", (40, 40), color=(5, 5, 5))
        result = self.engine._color_sync(_image_request(black))
        self.assertEqual(result["dominant_colors"][0]["name"], "black")

    def test_hist_bin_count_respects_env_override(self) -> None:
        import os

        image = Image.new("RGB", (20, 20), color=(220, 20, 20))
        old = os.environ.get("AIC_COLOR_HIST_BINS")
        os.environ["AIC_COLOR_HIST_BINS"] = "8"
        try:
            result = self.engine._color_sync(_image_request(image))
            self.assertEqual(len(result["hsv_histogram"]), 8)
        finally:
            if old is None:
                os.environ.pop("AIC_COLOR_HIST_BINS", None)
            else:
                os.environ["AIC_COLOR_HIST_BINS"] = old


class LoadQwenModelFamilyGuardTests(unittest.TestCase):
    """_load_qwen only ever instantiates Qwen2_5_VLForConditionalGeneration — a wrong
    AIC_CAPTION_MODEL family must fail fast, before the heavy torch/transformers
    import/download, with a message naming the actual problem."""

    def setUp(self) -> None:
        self._old = os.environ.get("AIC_CAPTION_MODEL")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("AIC_CAPTION_MODEL", None)
        else:
            os.environ["AIC_CAPTION_MODEL"] = self._old

    def test_non_qwen25vl_model_name_raises_clear_error(self) -> None:
        os.environ["AIC_CAPTION_MODEL"] = "Salesforce/blip-image-captioning-base"
        engine = TransformersGpuEngine()
        with self.assertRaisesRegex(ValueError, "Qwen2.5-VL"):
            engine._load_qwen()

    def test_qwen3_vl_name_also_rejected_here(self) -> None:
        # Qwen3-VL captioning is scripts/caption_qwen3vl.py's separate HTTP path, not
        # this in-process transformers loader.
        os.environ["AIC_CAPTION_MODEL"] = "Qwen/Qwen3-VL-32B-Instruct"
        engine = TransformersGpuEngine()
        with self.assertRaisesRegex(ValueError, "Qwen2.5-VL"):
            engine._load_qwen()


if __name__ == "__main__":
    unittest.main()
