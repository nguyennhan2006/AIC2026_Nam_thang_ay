"""PR-02: stage pack contract + assemble notebook -> canonical.

Các test này khóa lại đúng những chỗ hai contract lệch nhau, vì đó là nơi
dữ liệu thật sẽ hỏng một cách âm thầm:

* notebook đánh `scene_id` 5 chữ số, canonical dùng 4;
* notebook ghi `end_frame` *inclusive*, canonical dùng `end_frame_exclusive`;
* notebook đặt `asr_segment_id = {vid}_A{n:06d}`, canonical `ASRSourceId`
  là `{vid}_ASR{n:06d}`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from datasection.schemas import Scene
from offline.assemble import assemble
from offline.config import OfflineSettings
from offline.stagepack import StagePackError, discover_packs, open_pack


def run(coro):
    return asyncio.run(coro)


def write_pack(
    root: Path,
    stage: str,
    manifest_name: str,
    rows: list[dict],
    *,
    model: str = "test-model",
    status: str = "success",
) -> Path:
    pack = root / stage
    (pack / "manifests").mkdir(parents=True, exist_ok=True)
    (pack / "_SUCCESS.json").write_text(
        json.dumps({"status": status, "stage": stage, "count": len(rows)}), encoding="utf-8"
    )
    (pack / "model_info.json").write_text(
        json.dumps({"component": stage, "model": model, "pack_version": "1.1.0"}),
        encoding="utf-8",
    )
    with (pack / "manifests" / manifest_name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return pack


def build_packs_dir(root: Path, *, with_asr: bool = True, with_caption: bool = True) -> Path:
    packs = root / "packs"
    packs.mkdir(parents=True, exist_ok=True)
    write_pack(packs, "video", "video_manifest.jsonl", [{
        "video_id": "L21_V001", "source_path": "raw/videos/L21_V001.mp4", "fps": 25.0,
        "frame_count": 500, "duration_sec": 20.0, "width": 1280, "height": 720,
        "codec": "h264", "audio_present": True,
    }])
    # end_frame inclusive, scene_id 5 chữ số — đúng như notebook đang ghi.
    write_pack(packs, "scene", "scene_manifest.jsonl", [
        {"video_id": "L21_V001", "scene_id": "L21_V001_S00000", "scene_index": 0,
         "start_frame": 0, "end_frame": 249, "start_sec": 0.0, "end_sec": 9.96,
         "detector": "transnetv2_pytorch", "confidence": 0.9},
        {"video_id": "L21_V001", "scene_id": "L21_V001_S00001", "scene_index": 1,
         "start_frame": 250, "end_frame": 499, "start_sec": 10.0, "end_sec": 19.96,
         "detector": "transnetv2_pytorch", "confidence": 0.8},
    ])
    write_pack(packs, "keyframe", "keyframe_manifest.jsonl", [
        {"video_id": "L21_V001", "frame_idx": 100, "timestamp_sec": 4.0,
         "image_path": "processed/keyframes/L21_V001/frame_000100.jpg",
         "width": 1280, "height": 720, "roles": ["representative"],
         "quality": {"sharpness": 120.0, "brightness": 0.5}},
        {"video_id": "L21_V001", "frame_idx": 380, "timestamp_sec": 15.2,
         "image_path": "processed/keyframes/L21_V001/frame_000380.jpg",
         "width": 1280, "height": 720, "roles": ["representative"]},
    ])
    if with_caption:
        write_pack(packs, "caption", "caption_manifest.jsonl", [
            {"video_id": "L21_V001", "frame_idx": 100,
             "captions": [{"text": "Một người đang chạy trên đường", "language": "vi"}]},
            {"video_id": "L21_V001", "frame_idx": 380,
             "captions": [{"text": "Đoàn người vẫy tay", "language": "vi"}]},
        ])
    if with_asr:
        write_pack(packs, "asr", "asr_segments.jsonl", [
            {"video_id": "L21_V001", "asr_segment_id": "L21_V001_A000000",
             "start_sec": 1.0, "end_sec": 5.0, "text": "Xin chào quý vị"},
            {"video_id": "L21_V001", "asr_segment_id": "L21_V001_A000001",
             "start_sec": 12.0, "end_sec": 16.0, "text": "Hẹn gặp lại"},
        ])
    return packs


def settings_for(root: Path) -> OfflineSettings:
    return replace(
        OfflineSettings.from_env(),
        data_root=root,
        export_dir=root / "exports",
        state_dir=root / "state",
    )


class StagePackContractTests(unittest.TestCase):
    def test_pack_without_success_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root, "scene", "scene_manifest.jsonl", [], status="running")
            with self.assertRaises(StagePackError) as ctx:
                open_pack(pack, "scene")
            self.assertIn("chưa chạy xong", str(ctx.exception))

    def test_discover_rejects_two_packs_for_the_same_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "packs"
            write_pack(root, "01_scene_detection", "scene_manifest.jsonl", [])
            write_pack(root, "scene_retry", "scene_manifest.jsonl", [])
            with self.assertRaises(StagePackError) as ctx:
                discover_packs(root)
            self.assertIn("nhiều pack cho cùng một stage", str(ctx.exception))

    def test_discover_matches_stage_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packs = build_packs_dir(Path(tmp))
            found = discover_packs(packs)
            self.assertEqual(
                sorted(found), ["asr", "caption", "keyframe", "scene", "video"]
            )

    def test_missing_required_stage_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = build_packs_dir(root)
            (packs / "keyframe" / "_SUCCESS.json").unlink()
            (packs / "keyframe" / "model_info.json").unlink()
            (packs / "keyframe" / "manifests" / "keyframe_manifest.jsonl").unlink()
            with self.assertRaises(StagePackError):
                run(assemble(discover_packs(packs), settings=settings_for(root)))


class AssembleTests(unittest.TestCase):
    def _assemble(self, root: Path, **kwargs):
        packs = build_packs_dir(root, **kwargs)
        return run(assemble(discover_packs(packs), settings=settings_for(root)))

    def test_scene_id_is_renumbered_to_canonical_four_digits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest, report = self._assemble(root)
            scenes = [
                json.loads(line)
                for line in (root / "exports" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["scene_id"] for item in scenes],
                ["L21_V001_S0000", "L21_V001_S0001"],
            )
            self.assertEqual(report.scene_count, 2)
            for raw in scenes:
                Scene.model_validate(raw)

    def test_inclusive_end_frame_becomes_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._assemble(root)
            scenes = [
                json.loads(line)
                for line in (root / "exports" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            # notebook: [0, 249] và [250, 499] -> canonical: [0, 250) và [250, 500)
            self.assertEqual(scenes[0]["start_frame"], 0)
            self.assertEqual(scenes[0]["end_frame_exclusive"], 250)
            self.assertEqual(scenes[1]["end_frame_exclusive"], 500)
            # Không có khoảng hở và không chồng lấn giữa hai scene liền kề.
            self.assertEqual(scenes[0]["end_frame_exclusive"], scenes[1]["start_frame"])

    def test_time_bounds_are_recomputed_so_keyframes_stay_inside(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._assemble(root)
            scenes = [
                json.loads(line)
                for line in (root / "exports" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            for scene in scenes:
                for frame in scene["keyframes"]:
                    self.assertTrue(
                        scene["start_sec"] <= frame["timestamp_sec"] < scene["end_sec"]
                    )

    def test_keyframe_id_is_built_by_assemble_from_video_and_frame_idx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._assemble(root)
            keyframes = [
                json.loads(line)
                for line in (root / "exports" / "keyframes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sorted(item["keyframe_id"] for item in keyframes),
                ["L21_V001_S0000_F000100", "L21_V001_S0001_F000380"],
            )
            # Enrichment join theo (video_id, frame_idx) chứ không theo keyframe_id.
            captions = {
                item["keyframe_id"]: [c["text"] for c in item["captions"]] for item in keyframes
            }
            self.assertEqual(
                captions["L21_V001_S0000_F000100"], ["Một người đang chạy trên đường"]
            )

    def test_asr_is_clipped_to_scene_and_gets_canonical_source_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest, report = self._assemble(root)
            scenes = [
                json.loads(line)
                for line in (root / "exports" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            first = scenes[0]["asr_segments"][0]
            self.assertEqual(first["source_segment_id"], "L21_V001_ASR000000")
            self.assertEqual(first["segment_id"], "L21_V001_S0000_A0000")
            self.assertEqual(first["start_sec"], 1.0)
            self.assertEqual(report.asr_segment_count, 2)

    def test_missing_optional_stage_is_reported_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest, report = self._assemble(root, with_asr=False, with_caption=False)
            self.assertIn("asr", report.stages_missing)
            self.assertIn("ocr", report.stages_missing)
            self.assertTrue(any("asr" in item for item in report.warnings))
            self.assertEqual(report.asr_segment_count, 0)

    def test_frame_outside_every_scene_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = build_packs_dir(root)
            path = packs / "keyframe" / "manifests" / "keyframe_manifest.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "video_id": "L21_V001", "frame_idx": 900, "timestamp_sec": 36.0,
                    "image_path": "processed/keyframes/L21_V001/frame_000900.jpg",
                    "width": 1280, "height": 720, "roles": ["representative"],
                }) + "\n")
            _manifest, report = run(assemble(discover_packs(packs), settings=settings_for(root)))
            reasons = [item["reason"] for item in report.quarantined]
            self.assertTrue(any("không rơi vào scene nào" in item for item in reasons))
            quarantine = (root / "exports" / "quarantine.jsonl").read_text(encoding="utf-8")
            self.assertIn("900", quarantine)

    def test_export_is_validated_by_datasection_verify(self) -> None:
        from datasection.exporter import verify_export

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._assemble(root)
            manifest = verify_export(root / "exports")
            self.assertEqual(manifest.scene_count, 2)
            self.assertEqual(manifest.keyframe_count, 2)


if __name__ == "__main__":
    unittest.main()
