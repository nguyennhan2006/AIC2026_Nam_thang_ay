"""online/adapters/frame_vector_store.py — dense_visual local (PR-13).

`build_frame_vector_rows` phải: (a) trả `has_real_embeddings=False` và rows
rỗng khi export chưa qua enrichment embedding (giữ nguyên hành vi
lexical_hash_fallback hiện có), (b) đọc đúng vector từ `storage_locations`
khi export có embedding thật, không suy đoán đường dẫn.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from online.adapters.frame_vector_store import build_frame_vector_rows
from online.adapters.json_metadata import JsonlSceneRepository


def run(coro):
    return asyncio.run(coro)


_BASE_SCENE: dict = {
    "action_tags": [], "asr_segments": [], "boundary_confidence_in": None,
    "boundary_confidence_out": None, "captions": [], "created_at": "2026-08-03T00:00:00Z",
    "embedding_refs": [], "end_frame_exclusive": 300, "end_sec": 10.0, "extensions": {},
    "keywords": [], "scene_clip_checksum": None, "scene_clip_path": None,
    "scene_id": "L01_V001_S0000", "scene_idx": 0, "schema_version": "1.0.0",
    "segmentation_provenance": {
        "created_at": "2026-08-03T00:00:00Z", "device": None, "model_name": "demo-segmentation",
        "model_revision": "demo", "parameters": {}, "pipeline_version": "aic-v1.0.0", "prompt_version": None,
    },
    "start_frame": 0, "start_sec": 0.0, "transition_in": "unknown", "transition_out": "unknown",
    "video_id": "L01_V001",
    "keyframes": [{
        "action_tags": [], "captions": [], "color": None, "created_at": "2026-08-03T00:00:00Z",
        "embedding_refs": [], "extensions": {}, "frame_idx": 150, "height": 540,
        "image_path": "processed/keyframes/L01_V001/frame_000150.jpg",
        "keyframe_id": "L01_V001_S0000_F000150", "objects": [], "ocr_instances": [],
        "quality": {
            "black_frame_ratio": None, "brightness": None, "contrast": None,
            "duplicate_score": None, "sharpness": None,
        },
        "roles": ["representative"], "scene_id": "L01_V001_S0000", "schema_version": "1.0.0",
        "selection_score": None, "source_checksum": None, "timestamp_sec": 5.0,
        "video_id": "L01_V001", "width": 960,
    }],
}


def _embedding_ref(vector_uri: str) -> dict:
    return {
        "embedding_name": "clip_vit_l14_v1",
        "modality": "image",
        "model_name": "openai/clip-vit-large-patch14",
        "model_revision": None,
        "dimension": 3,
        "normalized": True,
        "storage_locations": [{
            "backend": "file",
            "vector_id": "L01_V001_S0000_F000150",
            "index_name": "clip_vit_l14_v1",
            "vector_uri": vector_uri,
        }],
    }


class FrameVectorStoreTests(unittest.TestCase):
    def _write_export(self, tmp: Path, *, with_embedding: bool) -> Path:
        scene = copy.deepcopy(_BASE_SCENE)
        keyframe = scene["keyframes"][0]
        if with_embedding:
            vector_uri = "processed/embeddings/L01_V001/frame_000150.json"
            keyframe["embedding_refs"] = [_embedding_ref(vector_uri)]
            vector_path = tmp / vector_uri
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            vector_path.write_text(json.dumps([0.1, 0.2, 0.3]), encoding="utf-8")

        scenes_path = tmp / "scenes.jsonl"
        scenes_path.write_text(json.dumps(scene), encoding="utf-8")
        # keyframes.jsonl là export phẳng riêng (datasection/exporter.py) — cùng
        # nội dung keyframe nhưng KHÔNG lồng trong scene.
        (tmp / "keyframes.jsonl").write_text(json.dumps(keyframe), encoding="utf-8")
        return scenes_path

    def test_no_real_embeddings_returns_empty_and_false(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scenes_path = self._write_export(tmp, with_embedding=False)
            repository = run(JsonlSceneRepository.load(scenes_path))
            rows, has_real = run(build_frame_vector_rows(repository, tmp))
            self.assertEqual(rows, [])
            self.assertFalse(has_real)

    def test_real_embedding_ref_resolves_vector_from_storage_location(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scenes_path = self._write_export(tmp, with_embedding=True)
            repository = run(JsonlSceneRepository.load(scenes_path))
            rows, has_real = run(build_frame_vector_rows(repository, tmp))
            self.assertTrue(has_real)
            self.assertEqual(len(rows), 1)
            keyframe_id, video_id, vector, payload = rows[0]
            self.assertEqual(keyframe_id, "L01_V001_S0000_F000150")
            self.assertEqual(video_id, "L01_V001")
            self.assertEqual(vector, [0.1, 0.2, 0.3])
            self.assertEqual(payload["frame_idx"], 150)
            self.assertEqual(payload["scene_id"], "L01_V001_S0000")
            self.assertEqual(payload["start_frame"], 0)
            self.assertEqual(payload["end_frame"], 299)

    def test_embedding_name_filter_skips_non_matching_reference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scenes_path = self._write_export(tmp, with_embedding=True)
            repository = run(JsonlSceneRepository.load(scenes_path))
            rows, has_real = run(
                build_frame_vector_rows(repository, tmp, embedding_name="siglip2_frame")
            )
            # Có embedding thật (has_real dò trên embedding_names bất kỳ) nhưng
            # không khớp embedding_name yêu cầu -> row bị bỏ qua, không đoán đại.
            self.assertTrue(has_real)
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
