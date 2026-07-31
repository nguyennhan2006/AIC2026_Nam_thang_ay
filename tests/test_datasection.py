from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from datasection.exporter import export_dataset, verify_export
from datasection.schemas import DatasetManifest, VectorLocation, Video
from datasection.vector_ids import qdrant_point_id
from scripts.seed_demo import main as seed
from scripts.export_schemas import main as export_schemas


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed(False)
        cls.root = Path(__file__).resolve().parents[1]

    def test_demo_export_verifies(self) -> None:
        manifest = verify_export(self.root / "storage/exports")
        self.assertEqual((manifest.video_count, manifest.scene_count, manifest.keyframe_count), (1, 3, 3))

    def test_every_scene_line_is_canonical(self) -> None:
        from datasection.schemas import Scene
        lines = (self.root / "storage/exports/scenes.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            Scene.model_validate_json(line)

    def test_committed_json_schemas_match_models(self) -> None:
        from datasection.schemas import ClipSegment, DatasetManifest, Event, Keyframe, Scene, Video
        export_schemas()
        for name, model in (
            ("keyframe", Keyframe), ("scene", Scene), ("clip", ClipSegment), ("event", Event),
            ("video", Video), ("dataset_manifest", DatasetManifest),
        ):
            committed = json.loads((self.root / f"contracts/{name}.schema.json").read_text(encoding="utf-8"))
            self.assertEqual(committed, model.model_json_schema())

    def test_qdrant_mapping_is_uuid_and_business_id_is_rejected(self) -> None:
        value = qdrant_point_id("L01_V001_S0002")
        self.assertEqual(value, qdrant_point_id("L01_V001_S0002"))
        VectorLocation(backend="qdrant", vector_id=value, index_name="scenes")
        with self.assertRaises(ValidationError):
            VectorLocation(backend="qdrant", vector_id="L01_V001_S0002", index_name="scenes")

    def test_fps_cross_check_rejects_wrong_timestamp(self) -> None:
        raw = json.loads((self.root / "storage/exports/videos.jsonl").read_text(encoding="utf-8"))
        raw["scenes"][0]["keyframes"][0]["timestamp_sec"] = 7.0
        with self.assertRaisesRegex(ValidationError, "timestamp disagrees"):
            Video.model_validate(raw)

    def test_checksum_tamper_is_detected(self) -> None:
        source = self.root / "storage/exports"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in ("dataset_manifest.json", "videos.jsonl", "scenes.jsonl", "keyframes.jsonl"):
                (target / name).write_bytes((source / name).read_bytes())
            with (target / "scenes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_export(target)


if __name__ == "__main__":
    unittest.main()
