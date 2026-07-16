"""Inference providers: deterministic smoke mode and retrying Vast.ai HTTP mode."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MockInferenceProvider:
    model_name = "mock/aic-v1"
    revision = "deterministic"

    async def image(self, task: str, path: Path, **context) -> dict:
        if task == "caption":
            return {"captions": [{"text": f"Khung hình {path.stem}", "language": "vi", "confidence": 0.5}]}
        if task == "ocr":
            return {"instances": []}
        if task == "object":
            return {"objects": []}
        if task == "embedding":
            digest = hashlib.sha256(path.read_bytes()).digest()
            vector = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(256)]
            norm = sum(x * x for x in vector) ** 0.5
            return {"vector": [x / norm for x in vector], "dimension": 256}
        raise ValueError(f"unsupported task: {task}")

    async def video(self, task: str, uri: str) -> dict:
        return {"segments": []} if task == "asr" else {}


class RemoteInferenceProvider:
    """Idempotent JSON client; retries only transient failures with jitter."""

    model_name = "remote/vast-worker"
    revision = "server-reported"

    def __init__(self, base_url: str, api_key: str | None, timeout: float, retries: int) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    async def _post(self, task: str, payload: dict, idempotency_key: str) -> dict:
        def call() -> dict:
            body = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json", "Idempotency-Key": idempotency_key}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = Request(f"{self.base_url}/v1/inference/{task}", data=body, headers=headers, method="POST")
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return await asyncio.to_thread(call)
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                last = exc
            except (URLError, TimeoutError) as exc:
                last = exc
            if attempt + 1 < self.retries:
                await asyncio.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.2)
        raise RuntimeError(f"GPU worker failed after {self.retries} attempts: {last}") from last

    async def image(self, task: str, path: Path, **context) -> dict:
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {"image_base64": base64.b64encode(path.read_bytes()).decode("ascii"), "filename": path.name, **context}
        return await self._post(task, payload, f"{task}:{checksum}")

    async def video(self, task: str, uri: str) -> dict:
        checksum = hashlib.sha256(uri.encode()).hexdigest()
        return await self._post(task, {"video_uri": uri}, f"{task}:{checksum}")
