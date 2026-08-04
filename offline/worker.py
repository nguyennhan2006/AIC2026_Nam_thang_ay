"""GPU inference gateway for Vast.ai.

The gateway owns authentication, concurrency and response contracts. Model code
is isolated behind ``InferenceEngine`` so teams can replace models without
changing Data or Online.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import os
import secrets
from pathlib import Path
import tempfile

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from offline.providers import MockInferenceProvider


class ImageRequest(BaseModel):
    image_base64: str
    filename: str = "image.jpg"
    caption: str | None = None
    candidate_labels: list[str] = Field(default_factory=list)


class VideoRequest(BaseModel):
    video_uri: str


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class InferenceEngine:
    """Dependency-light reference engine; replace in production via one class."""

    def __init__(self) -> None:
        self.mock = MockInferenceProvider()

    async def image(self, task: str, request: ImageRequest) -> dict:
        try:
            raw = base64.b64decode(request.image_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(422, "invalid image_base64") from exc
        maximum = int(os.getenv("AIC_MAX_IMAGE_BYTES", "20000000"))
        if len(raw) > maximum:
            raise HTTPException(413, "image exceeds AIC_MAX_IMAGE_BYTES")
        suffix = Path(request.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            handle.write(raw)
            handle.flush()
            # The default is intentionally deterministic. Install a team model
            # adapter here; the API response contract remains unchanged.
            return await self.mock.image(
                task, Path(handle.name), caption=request.caption,
                candidate_labels=request.candidate_labels,
            )

    async def text_embedding(self, text: str) -> dict:
        from offline.indexing import hashing_vector
        return {"vector": hashing_vector(text), "dimension": 256, "model": "hashing-v1"}

    async def video(self, task: str, uri: str) -> dict:
        return await self.mock.video(task, uri)


def create_worker() -> FastAPI:
    app = FastAPI(title="AIC V1 GPU Worker", version="1.0.0")
    if os.getenv("AIC_GPU_PROVIDER", "mock").lower() == "transformers":
        from offline.gpu_engine import TransformersGpuEngine
        engine = TransformersGpuEngine()
    else:
        engine = InferenceEngine()
    semaphore = asyncio.Semaphore(int(os.getenv("AIC_GPU_CONCURRENCY", "2")))
    expected_key = os.getenv("AIC_GPU_API_KEY")

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if expected_key and not secrets.compare_digest(authorization or "", f"Bearer {expected_key}"):
            raise HTTPException(401, "invalid worker token")

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ok", "provider": os.getenv("AIC_GPU_PROVIDER", "mock"), "concurrency": semaphore._value}

    @app.post("/v1/inference/asr", dependencies=[Depends(authorize)])
    async def infer_asr(request: VideoRequest) -> dict:
        async with semaphore:
            return await engine.video("asr", request.video_uri)

    @app.post("/v1/inference/{task}", dependencies=[Depends(authorize)])
    async def infer_image(task: str, request: ImageRequest) -> dict:
        if task not in {"caption", "ocr", "object", "embedding", "color"}:
            raise HTTPException(404, "unsupported image task")
        async with semaphore:
            return await engine.image(task, request)

    @app.post("/v1/embed/text", dependencies=[Depends(authorize)])
    async def embed_text(request: TextRequest) -> dict:
        async with semaphore:
            return await engine.text_embedding(request.text)

    return app


app = create_worker()
