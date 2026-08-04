"""Contract tests for nested keyframes and scene-level temporal evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from datasection.schemas.common import BoundingBox, ModelProvenance
from datasection.schemas.keyframe import Keyframe, KeyframeRole, OCRInstance
from datasection.schemas.scene import ASRSegment, Scene, SceneCaptionRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def provenance(model_name: str = "test/model") -> ModelProvenance:
    return ModelProvenance(
        model_name=model_name,
        model_revision="test-revision",
        pipeline_version="test-pipeline-v1",
    )


def valid_keyframe() -> Keyframe:
    return Keyframe(
        keyframe_id="L01_V001_S0003_F001234",
        video_id="L01_V001",
        scene_id="L01_V001_S0003",
        frame_idx=1234,
        timestamp_sec=41.133,
        image_path="processed/keyframes/L01_V001/frame_001234.jpg",
        width=1920,
        height=1080,
        roles=[KeyframeRole.REPRESENTATIVE],
        ocr_instances=[
            OCRInstance(
                text="Gừng cay muối mặn",
                normalized_text="gừng cay muối mặn",
                language="vi",
                confidence=0.97,
                bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.9, y2=0.4),
                provenance=provenance("ocr/test"),
            )
        ],
    )


def valid_scene_data() -> dict:
    return {
        "scene_id": "L01_V001_S0003",
        "video_id": "L01_V001",
        "scene_idx": 3,
        "start_frame": 1200,
        "end_frame_exclusive": 1300,
        "start_sec": 40.0,
        "end_sec": 43.334,
        "segmentation_provenance": provenance("TransNetV2"),
        "keyframes": [valid_keyframe()],
        "captions": [
            SceneCaptionRecord(
                caption_type="visual",
                text="A group stands near a sign with Vietnamese text.",
                evidence_keyframe_ids=["L01_V001_S0003_F001234"],
                provenance=provenance("caption/test"),
            )
        ],
        "asr_segments": [
            ASRSegment(
                segment_id="L01_V001_S0003_A0001",
                source_segment_id="L01_V001_ASR000123",
                start_sec=40.2,
                end_sec=43.334,
                text="Xin đừng quên nhau",
                language="vi",
                confidence=0.95,
                provenance=provenance("asr/test"),
            )
        ],
    }


class SceneContractTests(unittest.TestCase):
    def test_valid_scene_embeds_keyframes_and_derives_text(self) -> None:
        scene = Scene(**valid_scene_data())
        self.assertEqual(scene.keyframes[0].scene_id, scene.scene_id)
        self.assertEqual(scene.duration_frames, 100)
        self.assertAlmostEqual(scene.duration_sec, 3.334)
        self.assertEqual(scene.ocr_text, "Gừng cay muối mặn")
        self.assertEqual(scene.asr_text, "Xin đừng quên nhau")

    def test_scene_id_must_match_video_and_scene_idx(self) -> None:
        data = valid_scene_data()
        data["scene_idx"] = 4
        with self.assertRaisesRegex(ValidationError, "scene_id must equal"):
            Scene(**data)

    def test_rejects_empty_or_reversed_intervals(self) -> None:
        cases = [
            ("end_frame_exclusive", 1200),
            ("end_sec", 40.0),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                data = valid_scene_data()
                data[field] = value
                with self.assertRaises(ValidationError):
                    Scene(**data)

    def test_half_open_frame_interval_rejects_end_boundary(self) -> None:
        data = valid_scene_data()
        keyframe = data["keyframes"][0].model_copy(
            update={
                "keyframe_id": "L01_V001_S0003_F001300",
                "frame_idx": 1300,
                "timestamp_sec": 43.0,
            }
        )
        data["keyframes"] = [keyframe]
        with self.assertRaisesRegex(ValidationError, "outside scene frame interval"):
            Scene(**data)

    def test_rejects_keyframe_from_another_scene(self) -> None:
        data = valid_scene_data()
        child = valid_keyframe().model_copy(
            update={
                "keyframe_id": "L01_V001_S0004_F001234",
                "scene_id": "L01_V001_S0004",
            }
        )
        data["keyframes"] = [child]
        with self.assertRaisesRegex(ValidationError, "belongs to another scene"):
            Scene(**data)

    def test_rejects_duplicate_keyframes(self) -> None:
        data = valid_scene_data()
        data["keyframes"] = [valid_keyframe(), valid_keyframe()]
        with self.assertRaisesRegex(ValidationError, "keyframe_id values must be unique"):
            Scene(**data)

    def test_keyframes_must_be_chronological(self) -> None:
        data = valid_scene_data()
        later = valid_keyframe().model_copy(
            update={
                "keyframe_id": "L01_V001_S0003_F001250",
                "frame_idx": 1250,
                "timestamp_sec": 41.667,
            }
        )
        data["keyframes"] = [later, valid_keyframe()]
        with self.assertRaisesRegex(ValidationError, "ordered by frame_idx"):
            Scene(**data)

    def test_caption_evidence_must_reference_a_child(self) -> None:
        data = valid_scene_data()
        data["captions"] = [
            SceneCaptionRecord(
                caption_type="visual",
                text="Invalid evidence reference",
                evidence_keyframe_ids=["L01_V001_S0003_F001250"],
                provenance=provenance("caption/test"),
            )
        ]
        with self.assertRaisesRegex(ValidationError, "unknown keyframes"):
            Scene(**data)

    def test_asr_projection_must_belong_to_scene_and_fit_interval(self) -> None:
        cases = [
            {"segment_id": "L01_V001_S0004_A0001"},
            {"start_sec": 39.9},
            {"end_sec": 43.5},
            {"source_segment_id": "L02_V001_ASR000123"},
        ]
        for update in cases:
            with self.subTest(update=update):
                data = valid_scene_data()
                data["asr_segments"] = [
                    data["asr_segments"][0].model_copy(update=update)
                ]
                with self.assertRaises(ValidationError):
                    Scene(**data)

    def test_qdrant_payload_contains_filterable_scene_identity(self) -> None:
        payload = Scene(**valid_scene_data()).qdrant_payload()
        self.assertEqual(payload["entity_type"], "scene")
        self.assertEqual(payload["keyframe_count"], 1)
        self.assertTrue(payload["has_ocr"])
        self.assertTrue(payload["has_asr"])

    def test_committed_json_schema_matches_model(self) -> None:
        contract_path = PROJECT_ROOT / "contracts" / "scene.schema.json"
        committed = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(committed, Scene.model_json_schema())


if __name__ == "__main__":
    unittest.main()
