from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .records import KeyframeDocument, LoadedComponents, SceneDocument
from .utils import normalize_space, read_jsonl, safe_extract_zip, scene_parts, unique_paths


RELEVANT_NAMES = {
    "scene_metadata.jsonl",
    "scene_embeddings.npy",
    "scene_docs.jsonl",
    "scene_visual_embeddings.npy",
    "frame_docs.jsonl",
    "frame_visual_embeddings.npy",
    "metadata_manifest.json",
    "validation_report.json",
    "scene_semantics_qwen3vl.jsonl",
    "keyframe_index.csv",
    "keyframe_visual_embeddings.npy",
    "keyframes.json",
    "ocr_keyframes.jsonl",
    "ocr_scenes.jsonl",
    "asr_scenes.jsonl",
    "scene_clip_manifest.json",
    "component_validation_report.json",
}

SCENE_ID_IN_TEXT = re.compile(r"(?P<scene_id>[A-Za-z0-9_.-]+_S\d+)")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _discover(roots: list[Path], filename: str) -> list[Path]:
    return unique_paths(path for root in roots for path in root.rglob(filename) if path.is_file())


def _prepare_roots(input_root: Path, staging_dir: Path) -> list[Path]:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    roots = [input_root]
    staging_dir.mkdir(parents=True, exist_ok=True)
    for number, archive in enumerate(sorted(input_root.rglob("*.zip"))):
        try:
            with zipfile.ZipFile(archive) as zipped:
                names = {Path(name).name for name in zipped.namelist()}
        except zipfile.BadZipFile:
            continue
        if not names.intersection(RELEVANT_NAMES) and not any(
            "valid" in name.casefold() or "quality_report" in name.casefold()
            for name in names
        ):
            continue
        destination = staging_dir / f"{number:04d}_{archive.stem}"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        safe_extract_zip(archive, destination)
        roots.append(destination)
    return roots


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _put_record(target: dict[str, dict], key: str, value: dict, source: Path) -> bool:
    if key not in target:
        target[key] = value
        return True
    if _canonical(target[key]) == _canonical(value):
        return False
    raise ValueError(f"Duplicate id {key!r} with different content in {source}")


def _model_name(info: dict | None) -> str:
    if not info:
        return "open_clip:ViT-B-32:openai"
    if info.get("model"):
        return str(info["model"])
    sources = info.get("source_embedding_models") or []
    for source in sources:
        if isinstance(source, dict) and source.get("model"):
            return str(source["model"])
    return "open_clip:ViT-B-32:openai"


def _metadata_manifest_model(info: dict | None) -> str:
    if not info:
        return "open_clip:ViT-B-32:openai"
    models = ((info.get("embedding") or {}).get("component_models") or [])
    return _model_name(models[0]) if models else "open_clip:ViT-B-32:openai"


def _load_keyframes(roots: list[Path]) -> tuple[list[KeyframeDocument], np.ndarray | None, str]:
    documents: list[KeyframeDocument] = []
    vectors: list[np.ndarray] = []
    seen: dict[str, tuple[dict, np.ndarray]] = {}
    model_names: set[str] = set()
    dimension: int | None = None

    for index_path in _discover(roots, "keyframe_index.csv"):
        matrix_path = index_path.parent / "keyframe_visual_embeddings.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing keyframe vectors next to {index_path}")
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Keyframe embeddings must be 2D: {matrix_path}")
        dimension = dimension or int(matrix.shape[1])
        if matrix.shape[1] != dimension:
            raise ValueError(f"Mixed keyframe embedding dimensions: {matrix_path}")
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(matrix):
            raise ValueError(f"CSV/vector row mismatch in {index_path}")
        quality_path = index_path.parent / "keyframes.json"
        quality_items = _read_json(quality_path) if quality_path.exists() else []
        quality_by_id = {str(item["keyframe_id"]): item for item in quality_items}
        model_path = index_path.parent / "model_info.json"
        if model_path.exists():
            model_names.add(_model_name(_read_json(model_path)))

        for csv_position, row in enumerate(rows):
            keyframe_id = str(row["keyframe_id"])
            local_row = int(row.get("embedding_row", csv_position))
            if not 0 <= local_row < len(matrix):
                raise IndexError(f"{keyframe_id}: invalid embedding_row={local_row}")
            vector = np.asarray(matrix[local_row], dtype=np.float32)
            public = {
                "keyframe_id": keyframe_id,
                "scene_id": str(row["scene_id"]),
                "frame_idx": int(row["frame_idx"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "image_path": str(row.get("image_path", "")),
            }
            if keyframe_id in seen:
                old_public, old_vector = seen[keyframe_id]
                if old_public != public or not np.allclose(old_vector, vector, atol=1e-6):
                    raise ValueError(f"Duplicate keyframe differs: {keyframe_id}")
                continue
            seen[keyframe_id] = (public, vector)
            item = quality_by_id.get(keyframe_id, {})
            quality = item.get("quality", {}) if isinstance(item, dict) else {}
            documents.append(
                KeyframeDocument(
                    **public,
                    vector_row=len(vectors),
                    quality_score=float(quality.get("score", 0.0) or 0.0),
                    metadata=item,
                )
            )
            vectors.append(vector)

    if not vectors:
        # Rich metadata schema (scene_docs/frame_docs) already carries the
        # frame-to-vector mapping, so outputs 01-04 need not be supplied again.
        for docs_path in _discover(roots, "frame_docs.jsonl"):
            matrix_path = docs_path.parent / "frame_visual_embeddings.npy"
            if not matrix_path.exists():
                raise FileNotFoundError(f"Missing frame vectors next to {docs_path}")
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
            items = list(read_jsonl(docs_path))
            if matrix.ndim != 2 or len(matrix) != len(items):
                raise ValueError(f"Frame docs/vector mismatch: {docs_path}")
            dimension = dimension or int(matrix.shape[1])
            if matrix.shape[1] != dimension:
                raise ValueError(f"Mixed frame embedding dimensions: {matrix_path}")
            manifest_path = docs_path.parent / "metadata_manifest.json"
            if manifest_path.exists():
                model_names.add(_metadata_manifest_model(_read_json(manifest_path)))
            for position, item in enumerate(items):
                keyframe_id = str(item["keyframe_id"])
                local_row = int(item.get("visual_embedding_row", item.get("embedding_row", position)))
                if not 0 <= local_row < len(matrix):
                    raise IndexError(f"{keyframe_id}: invalid visual_embedding_row={local_row}")
                vector = np.asarray(matrix[local_row], dtype=np.float32)
                public = {
                    "keyframe_id": keyframe_id,
                    "scene_id": str(item["scene_id"]),
                    "frame_idx": int(item["frame_idx"]),
                    "timestamp_sec": float(item["timestamp_sec"]),
                    "image_path": str(item.get("image_path") or item.get("image_asset_uri", "")),
                }
                if keyframe_id in seen:
                    old_public, old_vector = seen[keyframe_id]
                    if old_public != public or not np.allclose(old_vector, vector, atol=1e-6):
                        raise ValueError(f"Duplicate rich keyframe differs: {keyframe_id}")
                    continue
                seen[keyframe_id] = (public, vector)
                quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
                documents.append(
                    KeyframeDocument(
                        **public,
                        vector_row=len(vectors),
                        quality_score=float(quality.get("score", 0.0) or 0.0),
                        ocr_text=normalize_space(item.get("ocr_text", "")),
                        metadata={
                            "quality": quality,
                            "selection": item.get("selection", {}),
                            "ocr_status": item.get("ocr_status", ""),
                            "provenance": item.get("provenance", {}),
                        },
                    )
                )
                vectors.append(vector)

    if not vectors:
        return [], None, "open_clip:ViT-B-32:openai"
    if len(model_names) > 1:
        raise ValueError(f"Mixed keyframe embedding models: {sorted(model_names)}")
    matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("Invalid keyframe vectors")
    matrix /= norms
    return documents, matrix, next(iter(model_names), "open_clip:ViT-B-32:openai")


def _load_scene_text(
    roots: list[Path], filename: str, key_field: str = "scene_id"
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in _discover(roots, filename):
        for item in read_jsonl(path):
            _put_record(result, str(item[key_field]), item, path)
    return result


def _load_scene_manifests(roots: list[Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in _discover(roots, "scene_clip_manifest.json"):
        raw = _read_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected manifest list: {path}")
        for item in raw:
            _put_record(result, str(item["scene_id"]), dict(item), path)
    return result


def _load_metadata(
    roots: list[Path],
) -> tuple[dict[str, dict], dict[str, np.ndarray], str]:
    records: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}
    models: set[str] = set()
    dimension: int | None = None
    for path in _discover(roots, "scene_metadata.jsonl"):
        matrix_path = path.parent / "scene_embeddings.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing scene_embeddings.npy next to {path}")
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
        items = list(read_jsonl(path))
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError(f"Metadata/vector mismatch: {path}")
        dimension = dimension or int(matrix.shape[1])
        if matrix.shape[1] != dimension:
            raise ValueError(f"Mixed scene embedding dimensions: {matrix_path}")
        model_path = path.parent / "model_info.json"
        if model_path.exists():
            models.add(_model_name(_read_json(model_path)))
        for item in items:
            scene_id = str(item["scene_id"])
            row = int(item["embedding_row"])
            if not 0 <= row < len(matrix):
                raise IndexError(f"{scene_id}: invalid scene embedding_row={row}")
            vector = np.asarray(matrix[row], dtype=np.float32)
            inserted = _put_record(records, scene_id, item, path)
            if inserted:
                vectors[scene_id] = vector
            elif not np.allclose(vectors[scene_id], vector, atol=1e-6):
                raise ValueError(f"Duplicate scene vector differs: {scene_id}")
    if not records:
        for path in _discover(roots, "scene_docs.jsonl"):
            matrix_path = path.parent / "scene_visual_embeddings.npy"
            if not matrix_path.exists():
                raise FileNotFoundError(f"Missing scene vectors next to {path}")
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
            items = list(read_jsonl(path))
            if matrix.ndim != 2 or len(matrix) != len(items):
                raise ValueError(f"Scene docs/vector mismatch: {path}")
            dimension = dimension or int(matrix.shape[1])
            if matrix.shape[1] != dimension:
                raise ValueError(f"Mixed scene embedding dimensions: {matrix_path}")
            manifest_path = path.parent / "metadata_manifest.json"
            if manifest_path.exists():
                models.add(_metadata_manifest_model(_read_json(manifest_path)))
            for position, raw in enumerate(items):
                item = dict(raw)
                scene_id = str(item["scene_id"])
                row = int(item.get("visual_embedding_row", item.get("embedding_row", position)))
                if not 0 <= row < len(matrix):
                    raise IndexError(f"{scene_id}: invalid visual_embedding_row={row}")
                item["embedding_row"] = row
                item.setdefault("keyframes", [])
                inserted = _put_record(records, scene_id, item, path)
                vector = np.asarray(matrix[row], dtype=np.float32)
                if inserted:
                    vectors[scene_id] = vector
                elif not np.allclose(vectors[scene_id], vector, atol=1e-6):
                    raise ValueError(f"Duplicate rich scene vector differs: {scene_id}")

    if len(models) > 1:
        raise ValueError(f"Mixed scene embedding models: {sorted(models)}")
    return records, vectors, next(iter(models), "open_clip:ViT-B-32:openai")


def _load_semantics(roots: list[Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in _discover(roots, "scene_semantics_qwen3vl.jsonl"):
        for item in read_jsonl(path):
            _put_record(result, str(item["scene_id"]), item, path)
    return result


def _iter_scene_validation(value: Any) -> Iterable[dict]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_scene_validation(item)
    elif isinstance(value, dict):
        if value.get("scene_id"):
            yield value
        for key, child in value.items():
            if key not in {"semantic", "metadata", "input", "generation"}:
                yield from _iter_scene_validation(child)


def _load_validations(roots: list[Path]) -> tuple[dict[str, list[dict]], list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    global_reports: list[dict] = []
    candidates = unique_paths(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".json", ".jsonl"}
        and ("valid" in path.name.casefold() or "quality_report" in path.name.casefold())
        and "raw_outputs" not in path.parts
    )
    for path in candidates:
        try:
            if path.suffix.casefold() == ".jsonl":
                values: Any = list(read_jsonl(path))
            else:
                values = _read_json(path)
        except Exception:
            continue
        if path.name in {"component_validation_report.json", "validation_report.json"} and isinstance(values, dict):
            report = dict(values)
            report["_source_file"] = path.name
            global_reports.append(report)
            for kind, status in (("warnings", "needs_review"), ("errors", "invalid")):
                for message in _list_text(values.get(kind)):
                    match = SCENE_ID_IN_TEXT.search(message)
                    if match:
                        result[match.group("scene_id")].append(
                            {"status": status, kind: [message], "source": path.name}
                        )
        for item in _iter_scene_validation(values):
            result[str(item["scene_id"])].append(item)
    return dict(result), global_reports


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_space(value)] if normalize_space(value) else []
    if not isinstance(value, list):
        return [normalize_space(value)] if normalize_space(value) else []
    output: list[str] = []
    for item in value:
        text = normalize_space(item)
        if text:
            output.append(text)
    return output


def _semantic_fields(record: dict | None) -> dict[str, Any]:
    record = record or {}
    semantic = record.get("semantic") if isinstance(record.get("semantic"), dict) else record
    subjects: list[str] = []
    subject_actions: list[str] = []
    subject_attributes: list[str] = []
    for subject in semantic.get("subjects", []) or []:
        if not isinstance(subject, dict):
            continue
        description = normalize_space(
            " ".join(filter(None, [str(subject.get("type", "")), str(subject.get("description", ""))]))
        )
        if description:
            subjects.append(description)
        subject_actions.extend(_list_text(subject.get("action")))
        subject_attributes.extend(_list_text(subject.get("attributes")))
    visible_text = [
        normalize_space(item.get("text", "") if isinstance(item, dict) else item)
        for item in (semantic.get("visible_text") or [])
    ]
    visible_text = [item for item in visible_text if item]
    events: list[dict[str, Any]] = []
    for index, event in enumerate(semantic.get("temporal_events") or [], 1):
        if not isinstance(event, dict):
            continue
        start = max(0.0, float(event.get("start_sec", 0.0) or 0.0))
        end = max(start, float(event.get("end_sec", start) or start))
        events.append(
            {
                "order": int(event.get("order", index) or index),
                "start_sec": start,
                "end_sec": end,
                "description_vi": normalize_space(event.get("description_vi", "")),
                "description_en": normalize_space(event.get("description_en", "")),
            }
        )
    objects = _list_text(semantic.get("objects"))
    actions = _list_text(semantic.get("actions")) + subject_actions
    attributes = _list_text(semantic.get("attributes")) + subject_attributes
    keywords = _list_text(semantic.get("keywords_vi")) + _list_text(semantic.get("keywords_en"))
    return {
        "caption_vi": normalize_space(semantic.get("caption_vi", "")),
        "caption_en": normalize_space(semantic.get("caption_en", "")),
        "speech_summary": normalize_space(semantic.get("speech_summary", "")),
        "scene_type": normalize_space(semantic.get("scene_type", "other")) or "other",
        "visible_text": normalize_space(" ".join(visible_text)),
        "keywords": normalize_space(" ".join(keywords)),
        "entities": normalize_space(" ".join([*subjects, *objects])),
        "actions": normalize_space(" ".join(actions)),
        "attributes": normalize_space(" ".join(attributes)),
        "relations": normalize_space(" ".join(_list_text(semantic.get("relations")))),
        "event_text": normalize_space(
            " ".join(
                text
                for event in events
                for text in (event["description_vi"], event["description_en"])
                if text
            )
        ),
        "temporal_events": events,
        "semantic_status": str(record.get("status", "generated" if semantic else "missing")),
        "semantic_errors": _list_text(record.get("quality_errors")),
    }


def _quality_state(
    semantic: dict | None,
    validations: list[dict],
    needs_review_penalty: float,
) -> tuple[str, float, list[str]]:
    statuses: list[str] = []
    errors: list[str] = []
    if semantic:
        statuses.append(str(semantic.get("status", "")))
        errors.extend(_list_text(semantic.get("quality_errors")))
    for item in validations:
        for key in ("status", "validation_status", "quality_status", "result"):
            if key in item:
                statuses.append(str(item[key]))
        if item.get("valid") is False or item.get("passed") is False:
            statuses.append("invalid")
        errors.extend(_list_text(item.get("errors")))
        errors.extend(_list_text(item.get("quality_errors")))
        errors.extend(_list_text(item.get("warnings")))
    normalized = " ".join(status.casefold() for status in statuses)
    if any(token in normalized for token in ("invalid", "failed", "error")):
        return "invalid", 0.0, list(dict.fromkeys(errors))
    if "needs_review" in normalized or "warning" in normalized or errors:
        return "needs_review", needs_review_penalty, list(dict.fromkeys(errors))
    return "passed", 1.0, []


def load_components(
    input_root: Path,
    staging_dir: Path,
    needs_review_penalty: float = 0.75,
) -> LoadedComponents:
    roots = _prepare_roots(input_root, staging_dir)
    warnings: list[str] = []
    keyframes, keyframe_embeddings, keyframe_model = _load_keyframes(roots)
    ocr_frames = _load_scene_text(roots, "ocr_keyframes.jsonl", key_field="keyframe_id")
    for frame in keyframes:
        frame.ocr_text = normalize_space(ocr_frames.get(frame.keyframe_id, {}).get("text", ""))

    ocr_scenes = _load_scene_text(roots, "ocr_scenes.jsonl")
    asr_scenes = _load_scene_text(roots, "asr_scenes.jsonl")
    manifests = _load_scene_manifests(roots)
    metadata, metadata_vectors, scene_model = _load_metadata(roots)
    semantics = _load_semantics(roots)
    validations, validation_reports = _load_validations(roots)
    failed_reports = [
        report
        for report in validation_reports
        if report.get("passed") is False
        or str(report.get("status", "")).casefold() in {"failed", "invalid", "error"}
    ]
    if failed_reports:
        errors = [
            message
            for report in failed_reports
            for message in _list_text(report.get("errors"))
        ]
        detail = "; ".join(errors[:8]) or "component compatibility check failed"
        raise ValueError(f"Output 06 validation failed: {detail}")

    frames_by_scene: dict[str, list[KeyframeDocument]] = defaultdict(list)
    for frame in keyframes:
        frames_by_scene[frame.scene_id].append(frame)

    if not metadata:
        warnings.append("scene_metadata.jsonl was not found; scene metadata was rebuilt from outputs 01-04.")
        if not manifests:
            raise FileNotFoundError("Need scene_metadata.jsonl or scene_clip_manifest.json")
        if keyframe_embeddings is None:
            raise FileNotFoundError("Need keyframe embeddings to rebuild scene embeddings")
        for scene_id, manifest in manifests.items():
            video_id, scene_no = scene_parts(scene_id)
            frames = sorted(frames_by_scene.get(scene_id, []), key=lambda item: item.frame_idx)
            if not frames:
                raise ValueError(f"{scene_id}: no keyframes")
            local_vectors = np.stack([keyframe_embeddings[frame.vector_row] for frame in frames])
            vector = local_vectors.mean(axis=0)
            vector /= np.linalg.norm(vector)
            metadata_vectors[scene_id] = vector
            representative = max(frames, key=lambda item: item.quality_score)
            start_sec = float(manifest.get("start_sec_absolute", frames[0].timestamp_sec))
            end_sec = float(manifest.get("end_sec_absolute", frames[-1].timestamp_sec))
            metadata[scene_id] = {
                "scene_id": scene_id,
                "video_id": video_id,
                "scene_index": scene_no,
                "start_frame": frames[0].frame_idx,
                "end_frame": frames[-1].frame_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "clip_path": str(manifest.get("clip_path", "")),
                "representative_keyframe_id": representative.keyframe_id,
                "ocr_text": normalize_space(ocr_scenes.get(scene_id, {}).get("text", "")),
                "transcript_text": normalize_space(asr_scenes.get(scene_id, {}).get("text", "")),
            }
        scene_model = keyframe_model

    # Output 07 keeps lightweight keyframe metadata even when output 01 is not
    # present. Retain those rows for representative-frame evidence; the frame
    # vector branch is simply unavailable without output 01 vectors.
    existing_keyframe_ids = {frame.keyframe_id for frame in keyframes}
    for scene_id, item in metadata.items():
        for raw in item.get("keyframes", []) or []:
            keyframe_id = str(raw.get("keyframe_id", ""))
            if not keyframe_id or keyframe_id in existing_keyframe_ids:
                continue
            frame = KeyframeDocument(
                keyframe_id=keyframe_id,
                scene_id=scene_id,
                frame_idx=int(raw.get("frame_idx", 0)),
                timestamp_sec=float(raw.get("timestamp_sec", item.get("start_sec", 0.0))),
                image_path=str(raw.get("image_path", "")),
                vector_row=len(keyframes),
                metadata={"source": "scene_metadata.jsonl"},
            )
            keyframes.append(frame)
            frames_by_scene[scene_id].append(frame)
            existing_keyframe_ids.add(keyframe_id)

    ordered_ids = sorted(metadata, key=scene_parts)
    scene_vectors: list[np.ndarray] = []
    scenes: list[SceneDocument] = []
    for scene_id in ordered_ids:
        item = metadata[scene_id]
        vector = np.asarray(metadata_vectors[scene_id], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{scene_id}: invalid scene embedding")
        scene_vectors.append(vector / norm)
        video_id, scene_no = scene_parts(scene_id)
        semantic = semantics.get(scene_id)
        fields = _semantic_fields(semantic)
        quality_status, quality_penalty, quality_errors = _quality_state(
            semantic, validations.get(scene_id, []), needs_review_penalty
        )
        manifest = manifests.get(scene_id, {})
        clip_path = str(item.get("clip_path") or manifest.get("clip_path") or "")
        scenes.append(
            SceneDocument(
                scene_id=scene_id,
                video_id=str(item.get("video_id", video_id)),
                scene_no=int(item.get("scene_index", scene_no)),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                clip_path=clip_path,
                start_frame=int(item.get("start_frame", 0)),
                end_frame=int(item.get("end_frame", 0)),
                representative_keyframe_id=str(item.get("representative_keyframe_id", "")),
                vector_row=len(scene_vectors) - 1,
                ocr_text=normalize_space(item.get("ocr_text") or ocr_scenes.get(scene_id, {}).get("text", "")),
                transcript=normalize_space(
                    item.get("transcript_text") or asr_scenes.get(scene_id, {}).get("text", "")
                ),
                **{key: value for key, value in fields.items() if key != "semantic_errors"},
                quality_status=quality_status,
                quality_penalty=quality_penalty,
                quality_errors=quality_errors,
                metadata={
                    "component_status": item.get("component_status", {}),
                    "provenance": item.get("provenance", {}),
                    "semantic_generation": (semantic or {}).get("generation", {}),
                    "semantic_uncertainty": (
                        ((semantic or {}).get("semantic") or {}).get("uncertainty", [])
                        if isinstance((semantic or {}).get("semantic"), dict)
                        else []
                    ),
                    "validation": validations.get(scene_id, []),
                },
            )
        )

    matrix = np.ascontiguousarray(np.stack(scene_vectors), dtype=np.float32)
    if semantics and set(metadata) - set(semantics):
        warnings.append(f"Missing semantics for {len(set(metadata) - set(semantics))} scenes.")
    if not semantics:
        warnings.append("Output 05 semantics was not found; semantic/tag/event branches are sparse.")
    component_reports = [
        report
        for report in validation_reports
        if report.get("_source_file") == "component_validation_report.json"
    ]
    if not component_reports:
        warnings.append("Output 06 component_validation_report.json was not found.")

    stats = {
        "source_roots": [str(path) for path in roots],
        "scene_count": len(scenes),
        "keyframe_count": len(keyframes),
        "video_count": len({scene.video_id for scene in scenes}),
        "semantic_scene_count": len(semantics),
        "validation_scene_count": len(validations),
        "validation_report_count": len(validation_reports),
        "validation_passed": bool(component_reports) and all(
            report.get("passed") is True for report in component_reports
        ),
        "passed_scene_count": sum(scene.quality_status == "passed" for scene in scenes),
        "needs_review_scene_count": sum(scene.quality_status == "needs_review" for scene in scenes),
        "invalid_scene_count": sum(scene.quality_status == "invalid" for scene in scenes),
    }
    return LoadedComponents(
        scenes=scenes,
        keyframes=keyframes,
        scene_embeddings=matrix,
        keyframe_embeddings=keyframe_embeddings,
        scene_embedding_model=scene_model,
        keyframe_embedding_model=keyframe_model,
        embedding_dimension=int(matrix.shape[1]),
        source_root=input_root.resolve(),
        stats=stats,
        warnings=warnings,
    )
