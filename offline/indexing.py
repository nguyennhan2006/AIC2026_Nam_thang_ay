"""Local exact/FAISS artifact plus Qdrant provisioning and batch upsert."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

from datasection.vector_ids import qdrant_point_id


def hashing_vector(text: str, dimension: int = 256) -> list[float]:
    vector = [0.0] * dimension
    for token in text.casefold().split():
        value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        vector[value % dimension] += 1.0 if value & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def _read_vector_file(path: Path) -> list[float]:
    if path.suffix == ".npy":
        import numpy  # local: chỉ cần khi embedding lưu dạng .npy

        return [float(value) for value in numpy.load(path)]
    return [float(value) for value in json.loads(path.read_text(encoding="utf-8"))]


def _keyframe_text(raw: dict) -> str:
    return " ".join(
        [item.get("text", "") for item in raw.get("captions", [])]
        + [item.get("text", "") for item in raw.get("ocr_instances", [])]
        + [item.get("label", "") for item in raw.get("objects", [])]
    )


def frame_rows(
    exports_dir: Path,
    data_root: Path,
    *,
    embedding_name: str | None = None,
    dimension: int = 256,
) -> tuple[list[dict], bool]:
    """Dựng point mức FRAME cho `aic_frames_v2`.

    Index mức scene (`scene_rows`) không đủ cho KIS/TRAKE: kết quả phải trỏ về
    một `frame_idx` cụ thể, và safe-frame/frame-refinement cần chấm điểm từng
    frame chứ không phải cả scene.

    Vector lấy từ `Keyframe.embedding_refs` (backend `file`). Frame chưa có
    embedding thật -> rơi về hashing text và hàm trả `degraded=True`; caller
    phải nói rõ điều đó thay vì để người dùng tưởng đang chạy dense thật.
    """

    scenes: dict[str, dict] = {}
    with (exports_dir / "scenes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw = json.loads(line)
                scenes[raw["scene_id"]] = raw
    event_by_scene: dict[str, str] = {}
    events_path = exports_dir / "events.jsonl"
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    event = json.loads(line)
                    for scene_id in event.get("scene_ids", []):
                        event_by_scene[scene_id] = event["event_id"]

    rows: list[dict] = []
    degraded = False
    with (exports_dir / "keyframes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            scene = scenes.get(raw["scene_id"], {})
            vector: list[float] | None = None
            used_name = embedding_name
            for reference in raw.get("embedding_refs", []):
                if embedding_name and reference.get("embedding_name") != embedding_name:
                    continue
                for location in reference.get("storage_locations", []):
                    if location.get("backend") == "file" and location.get("vector_uri"):
                        vector = _read_vector_file(data_root / location["vector_uri"])
                        used_name = reference["embedding_name"]
                        break
                if vector is not None:
                    break
            if vector is None:
                vector = hashing_vector(_keyframe_text(raw), dimension)
                used_name = used_name or "hashing_text_fallback"
                degraded = True
            rows.append({
                "id": raw["keyframe_id"],
                "vector": vector,
                "vector_name": used_name,
                "payload": {
                    "entity_type": "keyframe",
                    "keyframe_id": raw["keyframe_id"],
                    "scene_id": raw["scene_id"],
                    "video_id": raw["video_id"],
                    "event_id": event_by_scene.get(raw["scene_id"]),
                    "frame_idx": raw["frame_idx"],
                    "timestamp_sec": raw["timestamp_sec"],
                    "image_path": raw["image_path"],
                    "start_frame": scene.get("start_frame"),
                    "end_frame": (scene.get("end_frame_exclusive") or 1) - 1,
                    "start_sec": scene.get("start_sec"),
                    "end_sec": scene.get("end_sec"),
                    "has_ocr": bool(raw.get("ocr_instances")),
                    "has_asr": bool(scene.get("asr_segments")),
                },
            })
    return rows, degraded


def scene_rows(scenes_jsonl: Path, dimension: int = 256) -> list[dict]:
    rows = []
    with scenes_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            texts = [item["text"] for item in raw.get("captions", [])]
            texts += [item["text"] for frame in raw["keyframes"] for item in frame.get("captions", [])]
            texts += [item["text"] for frame in raw["keyframes"] for item in frame.get("ocr_instances", [])]
            rows.append({"id": raw["scene_id"], "vector": hashing_vector(" ".join(texts), dimension), "payload": {
                "scene_id": raw["scene_id"], "video_id": raw["video_id"], "scene_idx": raw["scene_idx"],
                "start_sec": raw["start_sec"], "end_sec": raw["end_sec"],
                "has_ocr": any(frame.get("ocr_instances") for frame in raw["keyframes"]), "has_asr": bool(raw.get("asr_segments")),
            }})
    return rows


async def scene_rows_remote(scenes_jsonl: Path, data_root: Path, provider) -> list[dict]:
    """Build visual scene vectors by normalized mean pooling keyframe CLIP vectors."""
    rows = []
    with scenes_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            vectors = []
            for frame in raw["keyframes"]:
                response = await provider.image("embedding", data_root / frame["image_path"])
                vectors.append([float(x) for x in response["vector"]])
            if not vectors or len({len(x) for x in vectors}) != 1:
                raise ValueError(f"invalid visual vectors for {raw['scene_id']}")
            mean = [sum(items) / len(items) for items in zip(*vectors, strict=True)]
            norm = math.sqrt(sum(x * x for x in mean))
            vector = [x / norm for x in mean] if norm else mean
            rows.append({"id": raw["scene_id"], "vector": vector, "payload": {
                "scene_id": raw["scene_id"], "video_id": raw["video_id"], "scene_idx": raw["scene_idx"],
                "start_sec": raw["start_sec"], "end_sec": raw["end_sec"],
                "has_ocr": any(frame.get("ocr_instances") for frame in raw["keyframes"]), "has_asr": bool(raw.get("asr_segments")),
            }})
    return rows


def build_local_index(rows: list[dict], output: Path) -> str:
    """Use FAISS when installed; always write a portable JSON fallback."""
    output.parent.mkdir(parents=True, exist_ok=True)
    portable = output.with_suffix(".json")
    portable.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    try:
        import faiss  # type: ignore
        import numpy as np
        matrix = np.asarray([row["vector"] for row in rows], dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(output.with_suffix(".faiss")))
        output.with_suffix(".ids.json").write_text(json.dumps([row["id"] for row in rows]), encoding="utf-8")
        return "faiss"
    except ImportError:
        return "portable-json"


class QdrantIndexer:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        timeout: float = 30,
        *,
        default_vector_name: str = "visual",
    ) -> None:
        self.url, self.collection, self.api_key, self.timeout = url.rstrip("/"), collection, api_key, timeout
        self.default_vector_name = default_vector_name

    def _request(self, method: str, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = Request(self.url + path, data=json.dumps(body).encode(), headers=headers, method=method)
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    async def provision(
        self,
        dimension: int | None = None,
        *,
        vectors: dict[str, int] | None = None,
        payload_fields: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """Tạo collection với một hoặc nhiều named vector.

        `dimension` (một vector tên "visual") giữ lại cho index scene cũ;
        `vectors={"openclip_l14": 768, ...}` dùng cho collection frame nhiều
        model — đổi model không được ghi đè vector của model khác.
        """

        if vectors is None:
            if dimension is None:
                raise ValueError("provision requires either `dimension` or `vectors`")
            vectors = {self.default_vector_name: dimension}
        await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}", {
            "vectors": {
                name: {"size": size, "distance": "Cosine"} for name, size in vectors.items()
            },
            "on_disk_payload": True,
        })
        fields = payload_fields or (
            ("video_id", "keyword"), ("scene_id", "keyword"), ("scene_idx", "integer"),
            ("has_ocr", "bool"), ("has_asr", "bool"),
        )
        for field, schema in fields:
            await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}/index", {"field_name": field, "field_schema": schema})

    async def upsert(self, rows: list[dict], batch_size: int = 128) -> None:
        for start in range(0, len(rows), batch_size):
            points = [
                {
                    "id": qdrant_point_id(row["id"]),
                    # `vector_name` cho phép cùng một hàm upsert cả collection
                    # scene (visual) lẫn collection frame (siglip2_frame/...).
                    "vector": {row.get("vector_name") or self.default_vector_name: row["vector"]},
                    "payload": row["payload"],
                }
                for row in rows[start:start + batch_size]
            ]
            await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}/points?wait=true", {"points": points})


FRAME_PAYLOAD_FIELDS: tuple[tuple[str, str], ...] = (
    ("video_id", "keyword"),
    ("scene_id", "keyword"),
    ("event_id", "keyword"),
    ("keyframe_id", "keyword"),
    ("frame_idx", "integer"),
    ("has_ocr", "bool"),
    ("has_asr", "bool"),
)
