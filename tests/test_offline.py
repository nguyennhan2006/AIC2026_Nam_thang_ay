from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline.config import OfflineSettings
from offline.indexing import QdrantIndexer, build_local_index, hashing_vector, scene_rows
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


class ASRProvider(MockInferenceProvider):
    async def video(self, task: str, uri: str) -> dict:
        return {"segments": [{"start_sec": 0.5, "end_sec": 1.5, "text": "xin chào", "language": "vi", "confidence": .9}]}


class OfflineTests(unittest.TestCase):
    def test_hashing_vector_is_normalized_and_stable(self) -> None:
        first = hashing_vector("xin chào thế giới", 64)
        self.assertEqual(first, hashing_vector("xin chào thế giới", 64))
        self.assertAlmostEqual(sum(x * x for x in first), 1.0)

    def test_mock_pipeline_exports_canonical_video(self) -> None:
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
            result = run(OfflinePipeline(settings, media=FakeMedia(), provider=MockInferenceProvider()).run())
            self.assertEqual((result.video_count, result.scene_count, result.keyframe_count), (1, 2, 2))
            state = json.loads((root / "state/L01_V001.json").read_text())
            self.assertEqual(state["status"], "succeeded")

    def test_asr_is_clipped_and_projected_to_each_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw/videos/L01_V001.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fake-video")
            class AudioMedia(FakeMedia):
                def probe(self, path):
                    base = super().probe(path)
                    return MediaInfo(base.fps, base.frame_count, base.duration_sec, base.width, base.height, base.codec, True)
            settings = OfflineSettings(root, source.parent, root/"exports", root/"state", None, None, 2, 2, 1, 1, "test-v1", "mock")
            run(OfflinePipeline(settings, media=AudioMedia(), provider=ASRProvider()).run())
            scenes = [json.loads(x) for x in (root/"exports/scenes.jsonl").read_text().splitlines()]
            self.assertEqual((scenes[0]["asr_segments"][0]["start_sec"], scenes[0]["asr_segments"][0]["end_sec"]), (0.5, 1.0))
            self.assertEqual((scenes[1]["asr_segments"][0]["start_sec"], scenes[1]["asr_segments"][0]["end_sec"]), (1.0, 1.5))

    def test_qdrant_indexer_uses_uuid_and_named_vector(self) -> None:
        captured = []
        client = QdrantIndexer("http://qdrant:6333", "scenes")
        async def fake(method, path, body):
            captured.append((method, path, body))
        with patch.object(client, "_request", side_effect=lambda m,p,b: captured.append((m,p,b)) or {}):
            run(client.provision(32))
            run(client.upsert([{"id":"L01_V001_S0001", "vector":[0.0]*32, "payload":{"scene_id":"L01_V001_S0001"}}]))
        point = captured[-1][2]["points"][0]
        self.assertIn("visual", point["vector"])
        self.assertNotEqual(point["id"], "L01_V001_S0001")


if __name__ == "__main__":
    unittest.main()
