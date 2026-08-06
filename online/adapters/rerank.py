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
import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.domain.evidence import EvidencePack
from online.errors import DependencyUnavailableError
from online.prompts import VLM_RERANK


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


def _image_data_url(path: Path, *, max_pixels: int = 1_200_000) -> str:
    """Đọc ảnh thành `data:` URL để nhúng thẳng vào `messages`.

    Thu nhỏ ảnh lớn trước khi gửi: keyframe gốc có thể vài MB, mà rerank gọi
    hàng chục lần mỗi truy vấn nên ảnh nguyên cỡ vừa chậm vừa dễ đụng giới hạn
    payload. Thiếu Pillow thì gửi nguyên bytes — mất tối ưu chứ không hỏng.
    """

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width * image.height > max_pixels:
                ratio = (max_pixels / (image.width * image.height)) ** 0.5
                image = image.resize(
                    (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            payload, mime = buffer.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 - Pillow thiếu hoặc ảnh lạ: gửi nguyên bản
        payload = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


class FptVlmReranker:
    """VLM rerank qua FPT `/chat/completions` — nhánh CHƯA TỪNG được đo.

    Vì sao không dùng được `QwenVlReranker`: adapter đó POST
    `{"query", "candidates"}` vào thẳng URL gốc, tức giả định có một worker tự
    host nói đúng contract ấy. FPT chỉ có `/chat/completions` kiểu OpenAI, nên
    đặt `AIC_RERANK_VLM_URL=https://mkp-api.fptcloud.com` chỉ làm nhánh này
    hỏng ở mọi request chứ không bật được gì.

    Hai ràng buộc của FPT định hình thiết kế:

    1. **Mỗi prompt chỉ được một ảnh** (HTTP 400 `At most 1 image(s) may be
       provided in one prompt`). Nên mỗi frame là một lệnh gọi riêng, rồi gộp
       lại theo candidate — không thể đưa cả 3 frame vào một prompt so sánh.
    2. Không có endpoint rerank đa phương thức, nên điểm số là do model tự
       chấm theo thang trong prompt, không phải logit của một cross-encoder.

    Cố tình **chỉ đưa ẢNH, không đưa caption/OCR** vào prompt: caption đã là
    thứ mà `bm25_caption` và text rerank dùng rồi. Cho VLM đọc lại caption thì
    nhánh này chỉ lặp lại tín hiệu cũ và không ai biết nó có thêm giá trị thị
    giác hay không. Giữ nó thuần thị giác thì con số đo được mới diễn giải được.

    Gộp nhiều frame về một điểm bằng `max`: scene đúng chỉ cần MỘT khung hình
    chứng minh, còn các khung khác trong cùng scene có thể là cảnh chuyển.
    """

    stage = "vlm"
    spec = VLM_RERANK

    def __init__(
        self,
        client: FptClient,
        *,
        model_id: str,
        data_root: Path,
        frames_per_candidate: int = 3,
        max_concurrency: int = 4,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.data_root = Path(data_root)
        self.frames_per_candidate = frames_per_candidate
        self.max_concurrency = max(1, max_concurrency)

    def _score_frame(self, query: str, image_path: Path) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.spec.render(query=query)},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            }
        ]
        text = self.client.chat_completion(
            messages,
            model=self.model_id,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
        ).text
        return _parse_vlm_json(text)

    async def score(self, query: str, packs: list[EvidencePack]) -> list[dict]:
        if not packs:
            return []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def score_frame(image_path: Path) -> dict | None:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self._score_frame, query, image_path)
                except (ProviderError, OSError, ValueError):
                    # Một frame hỏng không được giết cả stage; số frame hỏng
                    # đi vào `vlm_verdict` bên dưới để vẫn nhìn thấy được.
                    return None

        jobs: list[tuple[int, Path]] = []
        for index, pack in enumerate(packs):
            for frame in pack.representative_frames(self.frames_per_candidate):
                path = self.data_root / frame.image_path
                if path.exists():
                    jobs.append((index, path))

        if not jobs:
            raise DependencyUnavailableError(
                f"VLM rerank: không tìm thấy file ảnh nào dưới {self.data_root} "
                "— kiểm tra AIC_DATA_ROOT và image_path trong keyframes.jsonl"
            )

        results = await asyncio.gather(*(score_frame(path) for _, path in jobs))
        if all(item is None for item in results):
            raise DependencyUnavailableError(
                f"VLM rerank: cả {len(jobs)} lệnh gọi FPT đều lỗi"
            )

        merged: list[dict] = [
            {
                "candidate_id": pack.candidate_id,
                "relevance": 0.0,
                "must_match_coverage": 0.0,
                "contradictions": [],
                "evidence_summary": "",
                "frames_scored": 0,
                "frames_failed": 0,
            }
            for pack in packs
        ]
        for (index, _), item in zip(jobs, results, strict=True):
            entry = merged[index]
            if item is None:
                entry["frames_failed"] += 1
                continue
            entry["frames_scored"] += 1
            relevance = item["relevance"]
            if relevance >= entry["relevance"]:
                entry["relevance"] = relevance
                entry["evidence_summary"] = item["evidence_summary"]
            entry["must_match_coverage"] = max(
                entry["must_match_coverage"], item["must_match_coverage"]
            )
            for contradiction in item["contradictions"]:
                if contradiction not in entry["contradictions"]:
                    entry["contradictions"].append(contradiction)
        return merged


def _parse_vlm_json(text: str) -> dict:
    """Bóc object JSON khỏi câu trả lời của VLM và ép về đúng kiểu.

    Model hay bọc JSON trong ```json ... ``` hoặc thêm một câu dẫn, nên cắt
    theo cặp ngoặc ngoài cùng thay vì `json.loads` thẳng.
    """

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"VLM không trả JSON: {text[:120]!r}")
    payload = json.loads(text[start : end + 1])

    def as_unit_float(value) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    contradictions = payload.get("contradictions")
    return {
        "relevance": as_unit_float(payload.get("relevance")),
        "must_match_coverage": as_unit_float(payload.get("must_match_coverage")),
        "contradictions": [str(item) for item in contradictions]
        if isinstance(contradictions, list)
        else [],
        "evidence_summary": str(payload.get("evidence_summary") or ""),
    }


__all__ = ["BgeTextReranker", "FptTextReranker", "FptVlmReranker", "QwenVlReranker"]
