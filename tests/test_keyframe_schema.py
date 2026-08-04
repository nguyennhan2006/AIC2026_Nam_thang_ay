"""Contract tests for keyframe identity, portability, and storage references."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from datasection.schemas.common import EmbeddingReference, VectorLocation
from datasection.schemas.keyframe import Keyframe, KeyframeRole


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_keyframe_data() -> dict:
    return {
        "keyframe_id": "L01_V001_S0003_F001234",
        "video_id": "L01_V001",
        "scene_id": "L01_V001_S0003",
        "frame_idx": 1234,
        "timestamp_sec": 41.133,
        "image_path": "processed/keyframes/L01_V001/frame_001234.jpg",
        "width": 1920,
        "height": 1080,
        "roles": ["representative", "ocr_rich"],
        "source_checksum": "sha256:" + "a" * 64,
    }


class KeyframeContractTests(unittest.TestCase):
    def test_valid_keyframe_keeps_enum_and_serializes_string(self) -> None:
        keyframe = Keyframe(**valid_keyframe_data())
        self.assertIs(keyframe.roles[1], KeyframeRole.OCR_RICH)
        self.assertEqual(keyframe.model_dump(mode="json")["roles"][1], "ocr_rich")

    def test_rejects_noncanonical_ids(self) -> None:
        cases = [
            ("video_id", "video-1"),
            ("scene_id", "L01_V001_scene_3"),
            ("keyframe_id", "L01_V001_S0003_K01"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                data = valid_keyframe_data()
                data[field] = value
                with self.assertRaises(ValidationError):
                    Keyframe(**data)

    def test_rejects_scene_from_another_video(self) -> None:
        data = valid_keyframe_data()
        data["video_id"] = "L02_V001"
        with self.assertRaisesRegex(ValidationError, "scene_id must belong"):
            Keyframe(**data)

    def test_rejects_keyframe_id_that_disagrees_with_frame_idx(self) -> None:
        data = valid_keyframe_data()
        data["frame_idx"] = 1235
        with self.assertRaisesRegex(ValidationError, "keyframe_id must equal"):
            Keyframe(**data)

    def test_image_path_must_be_relative_to_data_root(self) -> None:
        for path in [
            "/absolute/frame.jpg",
            "../outside/frame.jpg",
            "s3://bucket/frame.jpg",
        ]:
            with self.subTest(path=path):
                data = valid_keyframe_data()
                data["image_path"] = path
                with self.assertRaises(ValidationError):
                    Keyframe(**data)

    def test_rejects_noncanonical_checksum(self) -> None:
        data = valid_keyframe_data()
        data["source_checksum"] = "md5:abc"
        with self.assertRaises(ValidationError):
            Keyframe(**data)

    def test_one_embedding_can_exist_in_faiss_and_qdrant(self) -> None:
        embedding = EmbeddingReference(
            embedding_name="clip_visual_v1",
            modality="image",
            model_name="openai/clip-vit-large-patch14",
            dimension=768,
            storage_locations=[
                VectorLocation(
                    backend="faiss", vector_id="42", index_name="keyframes_v1"
                ),
                VectorLocation(
                    # Qdrant point id phải là UUID (đúng cách offline/indexing.py::
                    # QdrantIndexer sinh id thật qua uuid5(NAMESPACE_URL, ...)) — không
                    # dùng thẳng keyframe_id dạng chuỗi.
                    backend="qdrant",
                    vector_id="7c8c7c0a-3f2a-5b7a-9b0a-1f1a2b3c4d5e",
                    index_name="aic_keyframes_v1",
                    vector_uri="qdrant://localhost/aic_keyframes_v1",
                ),
            ],
        )
        data = valid_keyframe_data()
        data["embedding_refs"] = [embedding]
        keyframe = Keyframe(**data)
        self.assertEqual(
            {
                item.backend
                for item in keyframe.embedding_refs[0].storage_locations
            },
            {"faiss", "qdrant"},
        )

    def test_committed_json_schema_matches_model(self) -> None:
        contract_path = PROJECT_ROOT / "contracts" / "keyframe.schema.json"
        committed = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(committed, Keyframe.model_json_schema())


if __name__ == "__main__":
    unittest.main()
