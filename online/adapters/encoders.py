"""Local deterministic and remote production text encoders."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from online.errors import DependencyUnavailableError


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class HashingTextEncoder:
    """Dependency-free deterministic encoder for smoke tests, never a quality model."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    async def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_RE.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector


class LocalClipTextEncoder:
    """CLIP text tower chạy tại chỗ (CPU), cùng không gian embedding với vector
    ảnh do `scripts/embed_keyframes_local.py` sinh — bắt buộc để dense_visual
    local so khớp được query text với keyframe, thay vì hash trên caption.

    `torch`/`transformers` chỉ import lúc gọi `encode()` lần đầu (không phải
    ở module import) — nhiều test dựng container mà không bao giờ gọi tới
    nhánh này, import sớm sẽ làm chậm toàn bộ test suite vô ích.
    """

    def __init__(self, model_path: str, *, revision: str | None = None) -> None:
        self.model_path = model_path
        self.revision = revision
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_path, revision=self.revision).to("cpu").eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_path, revision=self.revision)
        return self._model, self._processor

    async def encode(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> list[float]:
        import torch

        model, processor = self._load()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            vector = model.get_text_features(**inputs)[0]
            vector = vector / vector.norm(p=2)
        return vector.detach().cpu().float().tolist()


class RemoteTextEncoder:
    """HTTP adapter for a separately deployed CLIP/SigLIP text encoder."""

    def __init__(self, url: str, timeout_sec: float = 10.0, api_key: str | None = None) -> None:
        self.url = url
        self.timeout_sec = timeout_sec
        self.api_key = api_key

    async def encode(self, text: str) -> list[float]:
        def request_vector() -> list[float]:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = Request(
                self.url,
                data=json.dumps({"text": text}).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                raise DependencyUnavailableError(
                    f"embedding service unavailable: {exc}"
                ) from exc
            vector = payload.get("vector")
            if not isinstance(vector, list) or not vector:
                raise DependencyUnavailableError(
                    "embedding service must return {'vector': [float, ...]}"
                )
            return [float(item) for item in vector]

        return await asyncio.to_thread(request_vector)
