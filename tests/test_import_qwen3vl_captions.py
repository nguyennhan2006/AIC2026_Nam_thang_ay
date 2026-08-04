from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

from datasection.schemas import Video
from scripts.import_qwen3vl_captions import merge_captions
from scripts.seed_demo import main as seed


def run(coro):
    return asyncio.run(coro)


CAPTION_ROW = {
    "schema_version": "aic-multikeyframe-v2.0",
    "scene_key": "L01_V001_S0000",
    "parse_ok": True,
    "keyframes": [
        {
            "keyframe_id": "1",
            "frame_idx": 150,
            "parsed": {
                "short_caption_en": "A worker rakes salt in a field.",
                "short_caption_vi": "Một người đang cào muối trên cánh đồng.",
                "detailed_caption_en": "A worker rakes white salt into piles under bright sunlight.",
                "detailed_caption_vi": "Một người cào muối trắng thành đống dưới nắng.",
                "keywords_en": ["salt field", "raking"],
                "keywords_vi": ["cánh đồng muối", "cào muối"],
                "entities": [{"entity_id": "E001", "name_en": "worker"}],
                "relations": [],
                "ocr_regions": [{"region_id": "OCR001", "text_raw": "should not leak into OCRInstance"}],
            },
        }
    ],
    "scene_context": {
        "short_caption_en": "Salt harvesting scene.",
        "short_caption_vi": "Cảnh thu hoạch muối.",
        "detailed_caption_en": "Workers rake salt into piles in a coastal salt field.",
        "detailed_caption_vi": "Người dân cào muối thành đống trên cánh đồng ven biển.",
        "scene_entities": [{"name_en": "salt field", "name_vi": "cánh đồng muối"}],
        "scene_actions": [{"scene_action_id": "SA001", "label_en": "raking"}],
        "visual_evidence": [],
    },
}


class ImportQwen3VLCaptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed(False)
        cls.path = Path(__file__).resolve().parents[1] / "storage/exports/videos.jsonl"

    def load_videos(self) -> list[Video]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Video.model_validate_json(line) for line in lines if line.strip()]

    def test_merge_adds_captions_keywords_and_debug_extensions(self) -> None:
        videos = self.load_videos()
        merged, warnings = merge_captions(videos, [CAPTION_ROW], model_revision="test-rev")
        self.assertEqual(warnings, [])

        scene = next(s for v in merged for s in v.scenes if s.scene_id == "L01_V001_S0000")
        self.assertTrue(any(c.text == "Salt harvesting scene." for c in scene.captions))
        self.assertTrue(any(c.text == "Workers rake salt into piles in a coastal salt field." for c in scene.captions))
        keyword_texts = {kw.text for kw in scene.keywords}
        self.assertIn("cánh đồng muối", keyword_texts)
        self.assertIn("raking", keyword_texts)
        self.assertEqual(scene.extensions["qwen3vl_scene_actions"], [{"scene_action_id": "SA001", "label_en": "raking"}])

        keyframe = next(k for k in scene.keyframes if k.frame_idx == 150)
        self.assertTrue(any(c.text == "A worker rakes salt in a field." for c in keyframe.captions))
        self.assertTrue(any(c.text == "Một người cào muối trắng thành đống dưới nắng." for c in keyframe.captions))
        # ocr_regions phải chỉ nằm trong extensions debug, KHÔNG lọt vào OCRInstance chính thức.
        self.assertEqual(keyframe.ocr_instances, [])
        self.assertEqual(
            keyframe.extensions["qwen3vl_ocr_regions_debug"],
            [{"region_id": "OCR001", "text_raw": "should not leak into OCRInstance"}],
        )

    def test_unknown_scene_key_is_skipped_with_warning(self) -> None:
        videos = self.load_videos()
        bad_row = dict(CAPTION_ROW, scene_key="L99_V999_S9999")
        merged, warnings = merge_captions(videos, [bad_row], model_revision=None)
        self.assertEqual(len(warnings), 1)
        self.assertIn("L99_V999_S9999", warnings[0])
        # Không đổi gì so với input khi không khớp scene nào.
        original_scene = next(s for v in videos for s in v.scenes if s.scene_id == "L01_V001_S0000")
        merged_scene = next(s for v in merged for s in v.scenes if s.scene_id == "L01_V001_S0000")
        self.assertEqual(len(original_scene.captions), len(merged_scene.captions))

    def test_merge_is_idempotent_free_of_duplicate_keywords(self) -> None:
        videos = self.load_videos()
        merged_once, _ = merge_captions(videos, [CAPTION_ROW], model_revision=None)
        merged_twice, _ = merge_captions(merged_once, [CAPTION_ROW], model_revision=None)
        scene = next(s for v in merged_twice for s in v.scenes if s.scene_id == "L01_V001_S0000")
        keys = [(kw.normalized_text, kw.language) for kw in scene.keywords]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
