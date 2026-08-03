"""Adapter reranker qua HTTP: BGE (text) và Qwen3-VL (đa phương thức) — PR-06.

Cả hai model đều nằm sau HTTP theo đúng topology ở
`docs/11_SERVER_IMPLEMENTATION.md` (vLLM/worker riêng), nên tầng online không
phụ thuộc vào việc model chạy ở đâu hay bằng runtime nào.

Nguyên tắc chung: **lỗi model không được làm hỏng kết quả retrieval**. Adapter
ném `DependencyUnavailableError`, và `rerank_pipeline` bắt lại rồi giữ nguyên
thứ hạng của stage trước kèm warning — degrade nhưng nói rõ, không im lặng.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.domain.evidence import EvidencePack
from online.errors import DependencyUnavailableError


def _post(url: str, body: dict, *, timeout_sec: float, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DependencyUnavailableError(f"rerank service unavailable: {exc}") from exc


class BgeTextReranker:
    """Cross-encoder văn bản (mặc định `bge-reranker-v2-m3`).

    Contract HTTP::

        POST {url}  {"query": str, "documents": [str, ...]}
        -> {"scores": [float, ...]}   # cùng thứ tự với documents
    """

    stage = "text"

    def __init__(
        self,
        url: str,
        *,
        model_id: str = "bge-reranker-v2-m3",
        timeout_sec: float = 15.0,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.model_id = model_id
        self.timeout_sec = timeout_sec
        self.api_key = api_key

    async def score(self, query: str, packs: list[EvidencePack]) -> list[float]:
        if not packs:
            return []
        documents = [pack.rerank_text() for pack in packs]

        def call() -> dict:
            return _post(
                self.url,
                {"model": self.model_id, "query": query, "documents": documents},
                timeout_sec=self.timeout_sec,
                api_key=self.api_key,
            )

        payload = await asyncio.to_thread(call)
        scores = payload.get("scores")
        if not isinstance(scores, list) or len(scores) != len(packs):
            raise DependencyUnavailableError(
                f"text reranker phải trả {{'scores': [...]}} đúng {len(packs)} phần tử, "
                f"nhận được {type(scores).__name__} độ dài "
                f"{len(scores) if isinstance(scores, list) else 'n/a'}"
            )
        return [float(item) for item in scores]


class QwenVlReranker:
    """Reranker đa phương thức trên vài frame đại diện.

    Contract HTTP::

        POST {url}  {"query": str, "candidates": [
                        {"candidate_id": str, "frames": [path, ...], "evidence": str}
                    ]}
        -> {"results": [{"candidate_id": str, "relevance": float,
                         "must_match_coverage": float?, "contradictions": [str]?,
                         "evidence_summary": str?, "confidence": float?}]}

    Đường ảnh gửi đi là đường TƯƠNG ĐỐI so với data root — worker tự phân giải,
    tầng online không phát tán đường tuyệt đối của máy chủ.
    """

    stage = "vlm"

    def __init__(
        self,
        url: str,
        *,
        model_id: str = "qwen3-vl-32b",
        frames_per_candidate: int = 3,
        timeout_sec: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.model_id = model_id
        self.frames_per_candidate = frames_per_candidate
        self.timeout_sec = timeout_sec
        self.api_key = api_key

    async def score(self, query: str, packs: list[EvidencePack]) -> list[dict]:
        if not packs:
            return []
        candidates = [
            {
                "candidate_id": pack.candidate_id,
                "frames": [
                    Path(frame.image_path).as_posix()
                    for frame in pack.representative_frames(self.frames_per_candidate)
                ],
                "evidence": pack.rerank_text(max_chars=600),
            }
            for pack in packs
        ]

        def call() -> dict:
            return _post(
                self.url,
                {"model": self.model_id, "query": query, "candidates": candidates},
                timeout_sec=self.timeout_sec,
                api_key=self.api_key,
            )

        payload = await asyncio.to_thread(call)
        results = payload.get("results")
        if not isinstance(results, list):
            raise DependencyUnavailableError(
                "VLM reranker phải trả {'results': [{'candidate_id', 'relevance'}, ...]}"
            )
        by_id = {str(item["candidate_id"]): item for item in results if "candidate_id" in item}
        missing = [pack.candidate_id for pack in packs if pack.candidate_id not in by_id]
        if missing:
            raise DependencyUnavailableError(
                f"VLM reranker bỏ sót {len(missing)} candidate (vd {missing[:3]}) — "
                "kết quả một phần sẽ làm thứ hạng lệch, không dùng được"
            )
        return [by_id[pack.candidate_id] for pack in packs]


class FptTextReranker:
    """Text reranker qua FPT AI Marketplace `/rerank` — PR-15.

    KHÁC `BgeTextReranker` (contract tự đặt `{"scores": [...]}`, dành cho
    worker tự host): FPT dùng schema Cohere/Jina-style thật
    (`{"results": [{"index", "relevance_score"}]}`, đã xác nhận bằng probe
    thủ công + `FptClient.rerank()` tự sắp lại theo `index`). Cùng interface
    `.score(query, packs) -> list[float]` với `BgeTextReranker` nên là
    drop-in cho `RerankPipeline`, không cần sửa gì ở đó.
    """

    stage = "text"

    def __init__(self, client: FptClient, *, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    async def score(self, query: str, packs: list[EvidencePack]) -> list[float]:
        if not packs:
            return []
        documents = [pack.rerank_text() for pack in packs]

        def call() -> list[float]:
            return self.client.rerank(query, documents, model=self.model_id).scores

        try:
            return await asyncio.to_thread(call)
        except ProviderError as exc:
            raise DependencyUnavailableError(f"FPT rerank unavailable: {exc}") from exc


__all__ = ["BgeTextReranker", "FptTextReranker", "QwenVlReranker"]
