"""W1 (offline feature #2 — action tags): rule-based lexicon match on caption text."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from offline.action_tags import extract_action_tags
from offline.config import OfflineSettings
from offline.media import MediaInfo
from offline.pipeline import OfflinePipeline
from offline.providers import MockInferenceProvider


def run(coro):
    return asyncio.run(coro)


class FakeMedia:
    def probe(self, path: Path) -> MediaInfo:
        return MediaInfo(fps=10, frame_count=20, duration_sec=2, width=320, height=180, codec="fake", audio_present=False)

    def extract_frame(self, source: Path, frame_idx: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"frame-{frame_idx}".encode())


class ExtractActionTagsTests(unittest.TestCase):
    def test_matches_vietnamese_single_word_verb(self) -> None:
        self.assertEqual(extract_action_tags("Người đang cào muối trên cánh đồng"), ["raking"])

    def test_matches_vietnamese_multiword_phrase(self) -> None:
        self.assertEqual(extract_action_tags("Đoàn người vẫy tay phía sau bảng chữ"), ["waving"])

    def test_matches_english_caption(self) -> None:
        self.assertEqual(extract_action_tags("A group of workers are cooking rice in the field"), ["cooking"])

    def test_multiple_tags_are_sorted_and_deduplicated(self) -> None:
        tags = extract_action_tags("Người đứng lên và chạy đi, sau đó lại đứng lại")
        self.assertEqual(tags, sorted(set(tags)))
        self.assertIn("standing", tags)
        self.assertIn("running", tags)

    def test_short_verb_does_not_false_positive_inside_unrelated_word(self) -> None:
        # "đợi" (wait) contains "đi" as a substring but must not match as a whole token.
        self.assertEqual(extract_action_tags("Cô ấy đợi ở sân ga"), [])

    def test_empty_caption_yields_no_tags(self) -> None:
        self.assertEqual(extract_action_tags(""), [])

    def test_caption_without_lexicon_words_yields_no_tags(self) -> None:
        self.assertEqual(extract_action_tags("Khung hình frame_000150"), [])


class ActionCaptionProvider(MockInferenceProvider):
    async def image(self, task: str, path: Path, **context) -> dict:
        if task == "caption":
            return {"captions": [{"text": "Người đang cào muối trên cánh đồng", "language": "vi", "confidence": 0.9}]}
        return await super().image(task, path, **context)


class ActionTagsPipelineIntegrationTests(unittest.TestCase):
    def test_keyframe_scene_and_clip_all_carry_the_same_action_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw/videos/L01_V001.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake-video")
            settings = OfflineSettings(
                data_root=root, input_dir=source.parent, export_dir=root / "exports", state_dir=root / "state",
                gpu_url=None, gpu_api_key=None, timeout_sec=2, retries=2, scene_seconds=1,
                keyframes_per_scene=1, pipeline_version="test-v1", provider="mock",
            )
            run(OfflinePipeline(settings, media=FakeMedia(), provider=ActionCaptionProvider()).run())
            scenes = [json.loads(x) for x in (root / "exports/scenes.jsonl").read_text(encoding="utf-8").splitlines()]
            clips = [json.loads(x) for x in (root / "exports/clips.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(scene["action_tags"] == ["raking"] for scene in scenes))
            self.assertTrue(scenes[0]["keyframes"][0]["action_tags"] == ["raking"])
            self.assertTrue(all(clip["action_tags"] == ["raking"] for clip in clips))


if __name__ == "__main__":
    unittest.main()
