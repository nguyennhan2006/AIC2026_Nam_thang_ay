"""Environment-driven configuration without an extra settings dependency."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for local or Qdrant-backed operation."""

    app_name: str
    environment: str
    log_level: str
    backend: str
    metadata_jsonl: Path
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_scene_collection: str
    qdrant_vector_name: str
    embedding_url: str | None
    request_timeout_sec: float
    candidate_limit: int
    rrf_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("AIC_ONLINE_BACKEND", "local").lower()
        if backend not in {"local", "qdrant"}:
            raise ValueError("AIC_ONLINE_BACKEND must be 'local' or 'qdrant'")
        qdrant_url = os.getenv("AIC_QDRANT_URL")
        embedding_url = os.getenv("AIC_EMBEDDING_URL")
        if backend == "qdrant" and not qdrant_url:
            raise ValueError("AIC_QDRANT_URL is required for qdrant backend")
        if backend == "qdrant" and not embedding_url:
            raise ValueError("AIC_EMBEDDING_URL is required for qdrant backend")
        timeout = float(os.getenv("AIC_REQUEST_TIMEOUT_SEC", "10"))
        if timeout <= 0:
            raise ValueError("AIC_REQUEST_TIMEOUT_SEC must be positive")
        return cls(
            app_name=os.getenv("AIC_APP_NAME", "AIC 2026 Online V1"),
            environment=os.getenv("AIC_ENV", "development"),
            log_level=os.getenv("AIC_LOG_LEVEL", "INFO").upper(),
            backend=backend,
            metadata_jsonl=Path(
                os.getenv("AIC_METADATA_JSONL", "datasection/exports/scenes.jsonl")
            ),
            qdrant_url=qdrant_url.rstrip("/") if qdrant_url else None,
            qdrant_api_key=os.getenv("AIC_QDRANT_API_KEY"),
            qdrant_scene_collection=os.getenv(
                "AIC_QDRANT_SCENE_COLLECTION", "aic_scenes_v1"
            ),
            qdrant_vector_name=os.getenv("AIC_QDRANT_VECTOR_NAME", "visual"),
            embedding_url=embedding_url.rstrip("/") if embedding_url else None,
            request_timeout_sec=timeout,
            candidate_limit=_env_int("AIC_CANDIDATE_LIMIT", 100),
            rrf_k=_env_int("AIC_RRF_K", 60),
        )
