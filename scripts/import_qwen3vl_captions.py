"""Gộp caption Qwen3-VL-32B (`scripts/caption_qwen3vl.py`) vào export canonical.

`caption_qwen3vl.py` sinh `scene_captions_selfhosted.jsonl` (schema
"aic-multikeyframe-v2.0") độc lập với `offline/pipeline.py` — nó không tự tạo
Scene/Keyframe mới (thiếu các field bắt buộc như `start_frame`/`segmentation_provenance`
đến từ pipeline video thật), mà CHỈ bổ sung caption/keyword vào Scene/Keyframe **đã
tồn tại** trong `storage/exports/videos.jsonl` (nguồn sự thật duy nhất, xem
`datasection/exporter.py::export_dataset` — `scenes.jsonl`/`keyframes.jsonl` chỉ là bản
phẳng phái sinh).

Theo quyết định của team (docs/14_TECHNICAL_PREPARATION.md mục "Đã làm"):
    - Caption (short/detailed, EN+VI) -> gộp vào `Keyframe.captions`.
    - `scene_context` (short/detailed EN+VI) -> gộp vào `Scene.captions`.
    - `keywords_en`/`keywords_vi` (scene + keyframe) -> gộp vào `Scene.keywords`.
    - entities/relations/scene_actions/visual_evidence -> giữ nguyên dạng thô trong
      `extensions` (không bỏ phí, nhưng chưa có consumer dùng tới).
    - `ocr_regions` của Qwen3-VL KHÔNG được map vào `OCRInstance` chính thức — nguồn OCR
      chính thức vẫn là Qwen2.5-VL-7B qua `offline/gpu_engine.py::_ocr_sync`. `ocr_regions`
      chỉ giữ lại trong `extensions["qwen3vl_ocr_regions_debug"]` để đối chiếu khi cần.

Không tự tạo Scene/Keyframe mới nếu `scene_key`/`frame_idx` không khớp export hiện có —
im lặng bỏ qua dòng đó (in cảnh báo), tránh sinh dữ liệu không nhất quán với contract.

CÁCH DÙNG:
    python -m scripts.import_qwen3vl_captions \
        --captions storage/exports/qwen3vl_captions/scene_captions_selfhosted.jsonl \
        --export-dir storage/exports \
        --build-id aic-v1.0.0-qwen3vl-caption-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasection.exporter import export_dataset
from datasection.schemas import (
    CaptionRecord,
    DatasetManifest,
    ModelProvenance,
    Scene,
    SceneCaptionRecord,
    SceneKeyword,
    Video,
)

MODEL_NAME = "Qwen3-VL-32B-Instruct"
PROMPT_VERSION = "aic-multikeyframe-v2.0"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _provenance(model_revision: str | None) -> ModelProvenance:
    return ModelProvenance(
        model_name=MODEL_NAME,
        model_revision=model_revision,
        pipeline_version="aic-v1.0.0",
        prompt_version=PROMPT_VERSION,
    )


def _keyframe_captions(parsed: dict, provenance: ModelProvenance) -> list[CaptionRecord]:
    records: list[CaptionRecord] = []
    for lang in ("en", "vi"):
        short = parsed.get(f"short_caption_{lang}")
        detailed = parsed.get(f"detailed_caption_{lang}")
        if short:
            records.append(CaptionRecord(language=lang, caption_type="short", text=short, provenance=provenance))
        if detailed:
            records.append(CaptionRecord(language=lang, caption_type="detailed", text=detailed, provenance=provenance))
    return records


def _scene_captions(scene_context: dict, provenance: ModelProvenance) -> list[SceneCaptionRecord]:
    records: list[SceneCaptionRecord] = []
    for lang in ("en", "vi"):
        short = scene_context.get(f"short_caption_{lang}")
        detailed = scene_context.get(f"detailed_caption_{lang}")
        if short:
            records.append(SceneCaptionRecord(language=lang, caption_type="summary", text=short, provenance=provenance))
        if detailed:
            records.append(SceneCaptionRecord(language=lang, caption_type="visual", text=detailed, provenance=provenance))
    return records


def _new_scene_keywords(
    scene_context: dict, keyframe_rows: list[dict], provenance: ModelProvenance, existing: set[tuple[str, str]]
) -> list[SceneKeyword]:
    seen: dict[tuple[str, str], SceneKeyword] = {}

    def add(text: str | None, lang: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        normalized = text.casefold()
        key = (normalized, lang)
        if key in existing or key in seen:
            return
        seen[key] = SceneKeyword(
            text=text, normalized_text=normalized, language=lang, sources=["caption"], provenance=provenance
        )

    for entity in scene_context.get("scene_entities") or []:
        if isinstance(entity, dict):
            add(entity.get("name_en"), "en")
            add(entity.get("name_vi"), "vi")
    for keyframe_row in keyframe_rows:
        parsed = keyframe_row.get("parsed") or {}
        for kw in parsed.get("keywords_en") or []:
            add(kw, "en")
        for kw in parsed.get("keywords_vi") or []:
            add(kw, "vi")
    return list(seen.values())


def merge_row_into_scene(scene: Scene, row: dict, provenance: ModelProvenance) -> tuple[Scene, list[str]]:
    """Trả về (scene đã gộp, danh sách cảnh báo). Không mutate `scene` gốc."""

    warnings: list[str] = []
    scene_context = row.get("scene_context") or {}
    keyframe_rows = row.get("keyframes") or []

    updated_keyframes = list(scene.keyframes)
    index_by_id = {kf.keyframe_id: idx for idx, kf in enumerate(updated_keyframes)}

    for keyframe_row in keyframe_rows:
        frame_idx = keyframe_row.get("frame_idx")
        parsed = keyframe_row.get("parsed") or {}
        if frame_idx is None or not parsed:
            continue
        expected_id = f"{scene.scene_id}_F{int(frame_idx):06d}"
        idx = index_by_id.get(expected_id)
        if idx is None:
            warnings.append(f"{scene.scene_id}: bỏ qua frame_idx={frame_idx} (không khớp keyframe export hiện có)")
            continue
        target = updated_keyframes[idx]
        new_captions = target.captions + _keyframe_captions(parsed, provenance)
        extensions = dict(target.extensions)
        extensions["qwen3vl_entities"] = parsed.get("entities")
        extensions["qwen3vl_relations"] = parsed.get("relations")
        extensions["qwen3vl_ocr_regions_debug"] = parsed.get("ocr_regions")
        updated_keyframes[idx] = target.model_copy(update={"captions": new_captions, "extensions": extensions})

    existing_keyword_keys = {(kw.normalized_text, kw.language) for kw in scene.keywords}
    new_keywords = scene.keywords + _new_scene_keywords(scene_context, keyframe_rows, provenance, existing_keyword_keys)
    new_scene_captions = scene.captions + _scene_captions(scene_context, provenance)

    extensions = dict(scene.extensions)
    extensions["qwen3vl_scene_actions"] = scene_context.get("scene_actions")
    extensions["qwen3vl_visual_evidence"] = scene_context.get("visual_evidence")

    merged = scene.model_copy(
        update={
            "keyframes": updated_keyframes,
            "captions": new_scene_captions,
            "keywords": new_keywords,
            "extensions": extensions,
        }
    )
    # model_copy(update=...) KHÔNG chạy lại validator (revalidate_instances mặc định
    # "never" trong Pydantic v2) — round-trip qua model_dump/model_validate để bắt lỗi
    # ngay tại đây thay vì để lọt vào export_dataset rồi mới phát hiện (hoặc tệ hơn,
    # không bao giờ phát hiện vì export_dataset cũng gặp cùng vấn đề).
    merged = Scene.model_validate(merged.model_dump(mode="json"))
    return merged, warnings


def merge_captions(videos: list[Video], caption_rows: list[dict], model_revision: str | None) -> tuple[list[Video], list[str]]:
    provenance = _provenance(model_revision)
    scenes_by_id: dict[str, tuple[int, int]] = {
        scene.scene_id: (v_idx, s_idx)
        for v_idx, video in enumerate(videos)
        for s_idx, scene in enumerate(video.scenes)
    }
    updated_scenes: dict[tuple[int, int], Scene] = {}
    all_warnings: list[str] = []

    for row in caption_rows:
        scene_key = row.get("scene_key")
        if not row.get("parse_ok") or not scene_key:
            all_warnings.append(f"bỏ qua row parse_ok=False hoặc thiếu scene_key: {row.get('scene_key')}")
            continue
        location = scenes_by_id.get(scene_key)
        if location is None:
            all_warnings.append(f"scene_key={scene_key} không có trong export hiện có — bỏ qua")
            continue
        current = updated_scenes.get(location) or videos[location[0]].scenes[location[1]]
        merged, warnings = merge_row_into_scene(current, row, provenance)
        updated_scenes[location] = merged
        all_warnings.extend(warnings)

    result_videos = list(videos)
    touched_video_indices = {v_idx for v_idx, _ in updated_scenes}
    for (v_idx, s_idx), merged_scene in updated_scenes.items():
        scenes = list(result_videos[v_idx].scenes)
        scenes[s_idx] = merged_scene
        result_videos[v_idx] = result_videos[v_idx].model_copy(update={"scenes": scenes})
    for v_idx in touched_video_indices:
        # Cùng lý do revalidate ở merge_row_into_scene: model_copy không tự chạy lại
        # validator của Video (vd đối chiếu scene_id/keyframe_id với video_id).
        result_videos[v_idx] = Video.model_validate(result_videos[v_idx].model_dump(mode="json"))

    return result_videos, all_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True, help="scene_captions_selfhosted.jsonl")
    parser.add_argument("--export-dir", type=Path, default=Path("storage/exports"))
    parser.add_argument("--model-revision", default=None, help="Qwen3-VL-32B revision đã pin (nếu có)")
    parser.add_argument("--build-id", default=None, help="build_id mới cho manifest; mặc định thêm hậu tố -qwen3vl")
    args = parser.parse_args()

    videos_path = args.export_dir / "videos.jsonl"
    manifest_path = args.export_dir / "dataset_manifest.json"
    videos = [Video.model_validate_json(line) for line in videos_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    caption_rows = _read_jsonl(args.captions)

    merged_videos, warnings = merge_captions(videos, caption_rows, args.model_revision)
    for warning in warnings:
        print(f"WARNING: {warning}")

    new_manifest = old_manifest.model_copy(
        update={
            "build_id": args.build_id or f"{old_manifest.build_id}-qwen3vl",
            "export_checksums": {},
        }
    )
    result = export_dataset(merged_videos, args.export_dir, new_manifest)
    print(f"Da ghi lai export voi build_id={result.build_id}, {len(merged_videos)} video.")


if __name__ == "__main__":
    main()
