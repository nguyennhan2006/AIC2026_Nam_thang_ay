"""In-memory cosine baseline and Qdrant Query API adapter."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from online.domain.models import Candidate, Modality, SearchFilters
from online.errors import DependencyUnavailableError


def qdrant_point_id(entity_id: str) -> str:
    """Map business IDs to Qdrant-compatible deterministic UUID strings."""

    return str(uuid5(NAMESPACE_URL, f"aic2026:v1:{entity_id}"))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class InMemoryVectorStore:
    def __init__(self, rows: list[tuple[str, str, list[float], dict]]) -> None:
        self.rows = rows

    async def health(self) -> bool:
        return True

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[Candidate]:
        results: list[tuple[str, str, float, dict]] = []
        for scene_id, video_id, candidate_vector, payload in self.rows:
            if filters.video_ids and video_id not in filters.video_ids:
                continue
            if filters.scene_ids and scene_id not in filters.scene_ids:
                continue
            if filters.has_ocr is not None and bool(payload.get("has_ocr")) != filters.has_ocr:
                continue
            if filters.has_asr is not None and bool(payload.get("has_asr")) != filters.has_asr:
                continue
            score = cosine(vector, candidate_vector)
            results.append((scene_id, video_id, score, payload))
        results.sort(key=lambda item: (-item[2], item[0]))
        return [
            Candidate(
                entity_id=scene_id,
                scene_id=scene_id,
                video_id=video_id,
                source="local_dense",
                modality=Modality.VISUAL,
                score=score,
                rank=rank,
                payload=payload,
            )
            for rank, (scene_id, video_id, score, payload) in enumerate(
                results[:limit], start=1
            )
        ]


def _qdrant_filter(filters: SearchFilters) -> dict | None:
    must: list[dict] = []
    if filters.video_ids:
        must.append({"key": "video_id", "match": {"any": filters.video_ids}})
    if filters.scene_ids:
        must.append({"key": "scene_id", "match": {"any": filters.scene_ids}})
    if filters.has_ocr is not None:
        must.append({"key": "has_ocr", "match": {"value": filters.has_ocr}})
    if filters.has_asr is not None:
        must.append({"key": "has_asr", "match": {"value": filters.has_asr}})
    if filters.start_sec_gte is not None:
        must.append({"key": "start_sec", "range": {"gte": filters.start_sec_gte}})
    if filters.end_sec_lte is not None:
        must.append({"key": "end_sec", "range": {"lte": filters.end_sec_lte}})
    return {"must": must} if must else None


class QdrantVectorStore:
    """Minimal stdlib HTTP adapter for Qdrant's universal Query API."""

    def __init__(
        self,
        url: str,
        collection: str,
        vector_name: str,
        *,
        api_key: str | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.collection = collection
        self.endpoint = f"{self.base_url}/collections/{collection}/points/query"
        self.vector_name = vector_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    async def health(self) -> bool:
        def perform() -> bool:
            headers = {"api-key": self.api_key} if self.api_key else {}
            request = Request(f"{self.base_url}/collections/{self.collection}", headers=headers, method="GET")
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    return response.status == 200 and payload.get("status") == "ok"
            except (HTTPError, URLError, TimeoutError):
                return False
        return await asyncio.to_thread(perform)

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[Candidate]:
        body: dict = {
            "query": list(vector),
            "using": self.vector_name,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        query_filter = _qdrant_filter(filters)
        if query_filter:
            body["filter"] = query_filter

        def perform() -> dict:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["api-key"] = self.api_key
            request = Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                raise DependencyUnavailableError(f"Qdrant unavailable: {exc}") from exc

        response = await asyncio.to_thread(perform)
        result = response.get("result", {})
        if isinstance(result, dict):
            points = result.get("points", [])
        elif isinstance(result, list):
            points = result
        else:
            points = []
        candidates: list[Candidate] = []
        for rank, point in enumerate(points, start=1):
            payload = point.get("payload") or {}
            scene_id = payload.get("scene_id")
            video_id = payload.get("video_id")
            if not scene_id or not video_id:
                continue
            candidates.append(
                Candidate(
                    entity_id=scene_id,
                    scene_id=scene_id,
                    video_id=video_id,
                    source="qdrant_dense",
                    modality=Modality.VISUAL,
                    score=float(point["score"]),
                    rank=rank,
                    payload=payload,
                )
            )
        return candidates
