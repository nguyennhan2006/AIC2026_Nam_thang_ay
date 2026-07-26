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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    embedding_api_key: str | None
    request_timeout_sec: float
    candidate_limit: int
    rrf_k: int
    data_root: Path
    cors_origins: tuple[str, ...]
    api_key: str | None
    enable_ocr_fuzzy: bool
    enable_query_prep: bool
    enable_expansion: bool
    enable_rules: bool

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
                os.getenv("AIC_METADATA_JSONL", "storage/exports/scenes.jsonl")
            ),
            qdrant_url=qdrant_url.rstrip("/") if qdrant_url else None,
            qdrant_api_key=os.getenv("AIC_QDRANT_API_KEY"),
            qdrant_scene_collection=os.getenv(
                "AIC_QDRANT_SCENE_COLLECTION", "aic_scenes_v1"
            ),
            qdrant_vector_name=os.getenv("AIC_QDRANT_VECTOR_NAME", "visual"),
            embedding_url=embedding_url.rstrip("/") if embedding_url else None,
            embedding_api_key=os.getenv("AIC_EMBEDDING_API_KEY") or os.getenv("AIC_GPU_API_KEY"),
            request_timeout_sec=timeout,
            candidate_limit=_env_int("AIC_CANDIDATE_LIMIT", 100),
            rrf_k=_env_int("AIC_RRF_K", 60),
            data_root=Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve(),
            cors_origins=tuple(x.strip() for x in os.getenv("AIC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()),
            api_key=os.getenv("AIC_ONLINE_API_KEY"),
            enable_ocr_fuzzy=_env_bool("AIC_ENABLE_OCR_FUZZY", False),
            enable_query_prep=_env_bool("AIC_ENABLE_QUERY_PREP", False),
            enable_expansion=_env_bool("AIC_ENABLE_EXPANSION", False),
            enable_rules=_env_bool("AIC_ENABLE_RULES", False),
        )
