"""Vector rows mức FRAME cho dense_visual thật, xây từ export đã nạp.

Cố tình song song (không import) với `offline/indexing.py::frame_rows` —
`online/` và `offline/` không cross-import nhau ở bất kỳ chỗ nào khác trong
repo, giữ nguyên ranh giới đó ở đây.

`FrameEvidence` (online/domain/candidate.py) CHỦ Ý chỉ giữ `embedding_names`
(tên) chứ không giữ vị trí lưu vector thật — theo đúng comment tại đó "vector
thật nằm trong vector store". Vì vậy hàm này đọc lại MỘT LẦN `keyframes.jsonl`
thô, chỉ để lấy `embedding_refs[].storage_locations`; phần còn lại của payload
(scene bounds, event_id, has_ocr/has_asr...) lấy từ repository đã nạp sẵn,
không parse lại toàn bộ export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from online.adapters.json_metadata import JsonlSceneRepository


def _read_vector_file(path: Path) -> list[float]:
    if path.suffix == ".npy":
        import numpy  # local: chỉ cần khi embedding lưu dạng .npy

        return [float(value) for value in numpy.load(path)]
    return [float(value) for value in json.loads(path.read_text(encoding="utf-8"))]


async def build_frame_vector_rows(
    repository: JsonlSceneRepository,
    data_root: Path,
    *,
    embedding_name: str | None = None,
) -> tuple[list[tuple[str, str, list[float], dict[str, Any]]], bool]:
    """Trả `(rows, has_real_embeddings)`.

    `has_real_embeddings=False` khi KHÔNG có frame nào từng qua enrichment
    embedding (vd fixture demo nhỏ) — caller phải rơi về nhánh lexical fallback
    thay vì dựng một vector store rỗng.
    """

    scenes = await repository.all()
    if not any(frame.embedding_names for scene in scenes for frame in scene.keyframes):
        return [], False

    keyframes_path = repository.path.with_name("keyframes.jsonl")
    raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    with keyframes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_by_key[(str(raw["video_id"]), int(raw["frame_idx"]))] = raw

    rows: list[tuple[str, str, list[float], dict[str, Any]]] = []
    for scene in scenes:
        for frame in scene.keyframes:
            raw = raw_by_key.get((frame.video_id, frame.frame_idx))
            if raw is None:
                continue
            vector: list[float] | None = None
            for reference in raw.get("embedding_refs", []):
                if embedding_name and reference.get("embedding_name") != embedding_name:
                    continue
                for location in reference.get("storage_locations", []):
                    if location.get("backend") == "file" and location.get("vector_uri"):
                        vector = _read_vector_file(data_root / str(location["vector_uri"]))
                        break
                if vector is not None:
                    break
            if vector is None:
                continue
            payload = {
                "entity_type": "keyframe",
                "keyframe_id": frame.keyframe_id,
                "scene_id": scene.scene_id,
                "video_id": scene.video_id,
                "event_id": scene.event_id,
                "frame_idx": frame.frame_idx,
                "timestamp_sec": frame.timestamp_sec,
                "image_path": frame.image_path,
                "start_frame": scene.start_frame,
                "end_frame": scene.end_frame_exclusive - 1,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "has_ocr": bool(frame.ocr_texts),
                "has_asr": bool(scene.asr_texts),
            }
            rows.append((frame.keyframe_id, scene.video_id, vector, payload))
    return rows, True


__all__ = ["build_frame_vector_rows"]
