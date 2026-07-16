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
    def __init__(self, url: str, collection: str, api_key: str | None = None, timeout: float = 30) -> None:
        self.url, self.collection, self.api_key, self.timeout = url.rstrip("/"), collection, api_key, timeout

    def _request(self, method: str, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = Request(self.url + path, data=json.dumps(body).encode(), headers=headers, method=method)
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    async def provision(self, dimension: int) -> None:
        await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}", {
            "vectors": {"visual": {"size": dimension, "distance": "Cosine"}},
            "on_disk_payload": True,
        })
        for field, schema in (("video_id", "keyword"), ("scene_id", "keyword"), ("scene_idx", "integer"), ("has_ocr", "bool"), ("has_asr", "bool")):
            await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}/index", {"field_name": field, "field_schema": schema})

    async def upsert(self, rows: list[dict], batch_size: int = 128) -> None:
        for start in range(0, len(rows), batch_size):
            points = [{"id": qdrant_point_id(row["id"]), "vector": {"visual": row["vector"]}, "payload": row["payload"]} for row in rows[start:start + batch_size]]
            await asyncio.to_thread(self._request, "PUT", f"/collections/{self.collection}/points?wait=true", {"points": points})
