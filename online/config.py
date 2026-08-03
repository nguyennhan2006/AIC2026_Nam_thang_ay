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


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _fpt_settings_kwargs() -> dict[str, object]:
    """Đọc toàn bộ `AIC_FPT_*`/`AIC_LOG_*` — tách riêng khỏi `Settings.from_env`
    (đã dài) và để lỗi cấu hình FPT hiện ra ngay khi bật, không phải lúc gọi
    API giữa thí nghiệm (Gate 0/Gate 1 của
    AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md)."""

    enabled = _env_bool("AIC_FPT_ENABLED", False)
    api_key = os.getenv("AIC_FPT_API_KEY") or None
    if enabled and not api_key:
        raise ValueError("AIC_FPT_ENABLED=true requires AIC_FPT_API_KEY")
    return {
        "fpt_enabled": enabled,
        "fpt_base_url": (os.getenv("AIC_FPT_BASE_URL") or "https://mkp-api.fptcloud.com").rstrip("/"),
        "fpt_api_key": api_key,
        "fpt_llm_model": os.getenv("AIC_FPT_LLM_MODEL") or None,
        "fpt_vlm_model": os.getenv("AIC_FPT_VLM_MODEL") or None,
        "fpt_rerank_model": os.getenv("AIC_FPT_RERANK_MODEL") or None,
        "fpt_timeout_sec": _env_float("AIC_FPT_TIMEOUT_SEC", 90.0),
        "fpt_connect_timeout_sec": _env_float("AIC_FPT_CONNECT_TIMEOUT_SEC", 10.0),
        "fpt_max_retries": _env_int("AIC_FPT_MAX_RETRIES", 3),
        "fpt_max_concurrency": _env_int("AIC_FPT_MAX_CONCURRENCY", 2),
        "fpt_retry_backoff_base_sec": _env_float("AIC_FPT_RETRY_BACKOFF_BASE_SEC", 1.0),
        "fpt_retry_backoff_max_sec": _env_float("AIC_FPT_RETRY_BACKOFF_MAX_SEC", 8.0),
        "log_request_body": _env_bool("AIC_LOG_REQUEST_BODY", False),
        "log_provider_response": _env_bool("AIC_LOG_PROVIDER_RESPONSE", False),
    }


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
    # Search Mixing Console W3 — mỗi cờ mặc định tắt, xem online/api/container.py.
    enable_object_search: bool
    enable_action_search: bool
    enable_color_search: bool
    enable_event_search: bool
    # PR-06 — rerank cascade. None = tầng đó không tồn tại; capabilities báo
    # False và request bật nó sẽ bị 422 thay vì im lặng không chạy.
    # Có default để chỗ nào dựng Settings trực tiếp (không qua from_env, vd
    # tests/test_container_flags.py) vẫn chạy — cùng cách OfflineSettings làm
    # với nhóm field clip pooling.
    rerank_text_url: str | None = None
    rerank_vlm_url: str | None = None
    rerank_api_key: str | None = None
    rerank_text_model: str = "bge-reranker-v2-m3"
    rerank_vlm_model: str = "qwen3-vl-32b"
    # Dense visual local (PR-13) — text tower phải cùng model với vector ảnh
    # đã sinh (`scripts/embed_keyframes_local.py`), nếu không hai không gian
    # embedding lệch nhau và cosine similarity vô nghĩa. Chỉ thật sự nạp model
    # khi export có embedding thật (online/adapters/frame_vector_store.py);
    # không có thì container vẫn dùng lexical_hash_fallback như trước.
    visual_embedding_model: str = "openai/clip-vit-large-patch14"
    visual_embedding_model_revision: str | None = None
    # PR-12 — FPT AI Marketplace, dùng TẠM thay server A100 tự host để test/
    # tune prompt (xem AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md).
    # `fpt_enabled=False` là mặc định an toàn: không có field nào ở đây được
    # đọc nếu tắt, và bật lên mà thiếu key phải lỗi ngay lúc load config chứ
    # không phải lúc gọi API giữa chừng thí nghiệm.
    fpt_enabled: bool = False
    fpt_base_url: str = "https://mkp-api.fptcloud.com"
    fpt_api_key: str | None = None
    fpt_llm_model: str | None = None
    fpt_vlm_model: str | None = None
    fpt_rerank_model: str | None = None
    fpt_timeout_sec: float = 90.0
    fpt_connect_timeout_sec: float = 10.0
    fpt_max_retries: int = 3
    fpt_max_concurrency: int = 2
    fpt_retry_backoff_base_sec: float = 1.0
    fpt_retry_backoff_max_sec: float = 8.0
    # Gate 0 (bảo mật) — mặc định tắt ghi log, chỉ bật thủ công khi debug.
    log_request_body: bool = False
    log_provider_response: bool = False

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
            enable_object_search=_env_bool("AIC_ENABLE_OBJECT_SEARCH", False),
            enable_action_search=_env_bool("AIC_ENABLE_ACTION_SEARCH", False),
            enable_color_search=_env_bool("AIC_ENABLE_COLOR_SEARCH", False),
            enable_event_search=_env_bool("AIC_ENABLE_EVENT_SEARCH", False),
            rerank_text_url=(os.getenv("AIC_RERANK_TEXT_URL") or None),
            rerank_vlm_url=(os.getenv("AIC_RERANK_VLM_URL") or None),
            rerank_api_key=os.getenv("AIC_RERANK_API_KEY") or os.getenv("AIC_GPU_API_KEY"),
            rerank_text_model=os.getenv("AIC_RERANK_TEXT_MODEL", "bge-reranker-v2-m3"),
            rerank_vlm_model=os.getenv("AIC_RERANK_VLM_MODEL", "qwen3-vl-32b"),
            visual_embedding_model=os.getenv("AIC_VISUAL_EMBEDDING_MODEL", "openai/clip-vit-large-patch14"),
            visual_embedding_model_revision=os.getenv("AIC_VISUAL_EMBEDDING_MODEL_REVISION") or None,
            **_fpt_settings_kwargs(),
        )
