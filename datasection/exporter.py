"""Validated, deterministic and atomic dataset exports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from datasection.schemas import DatasetManifest, Keyframe, Scene, Video


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def atomic_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def export_dataset(videos: list[Video], output_dir: Path, manifest: DatasetManifest) -> DatasetManifest:
    validated = [Video.model_validate(item) for item in videos]
    scenes: list[Scene] = [scene for video in validated for scene in video.scenes]
    keyframes = [frame for scene in scenes for frame in scene.keyframes]
    for label, values in (
        ("video_id", [item.video_id for item in validated]),
        ("scene_id", [item.scene_id for item in scenes]),
        ("keyframe_id", [item.keyframe_id for item in keyframes]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} in export")
    if (manifest.video_count, manifest.scene_count, manifest.keyframe_count) != (
        len(validated), len(scenes), len(keyframes)
    ):
        raise ValueError("manifest counts do not match exported entities")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "videos.jsonl": [item.model_dump(mode="json") for item in validated],
        "scenes.jsonl": [item.model_dump(mode="json") for item in scenes],
        "keyframes.jsonl": [item.model_dump(mode="json") for item in keyframes],
    }
    checksums: dict[str, str] = {}
    for name, rows in files.items():
        target = output_dir / name
        atomic_jsonl(target, rows)
        checksums[name] = sha256_file(target)
    final_manifest = manifest.model_copy(update={"export_checksums": checksums})
    target = output_dir / "dataset_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(final_manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)
    return final_manifest


def verify_export(output_dir: Path) -> DatasetManifest:
    manifest = DatasetManifest.model_validate_json((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest.export_checksums.items():
        if sha256_file(output_dir / name) != checksum:
            raise ValueError(f"checksum mismatch: {name}")
    counts = {}
    models = {"videos.jsonl": Video, "scenes.jsonl": Scene, "keyframes.jsonl": Keyframe}
    for name, model in models.items():
        count = 0
        with (output_dir / name).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    model.model_validate_json(line)
                    count += 1
        counts[name] = count
    if (
        counts["videos.jsonl"] != manifest.video_count
        or counts["scenes.jsonl"] != manifest.scene_count
        or counts["keyframes.jsonl"] != manifest.keyframe_count
    ):
        raise ValueError("export row counts do not match manifest")
    return manifest
