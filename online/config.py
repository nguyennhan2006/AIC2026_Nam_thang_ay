"""Environment-driven configuration without an extra settings dependency."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Nạp file kiểu `.env` vào `os.environ`. Trả về tên các biến đã đặt.

    Tồn tại vì luồng online KHÔNG hề đọc `.env.*` ở đâu cả: `uvicorn
    online.api.app:app` khởi động với `fpt_enabled=False` và không một dòng
    cảnh báo nào, nên một buổi đo tưởng là "có FPT" thực ra chạy hoàn toàn
    không FPT. Chỉ có `scripts/enrich_keyframes_fpt.py` và
    `scripts/fpt_api_preflight.py` tự parse `--env-file` của riêng chúng.

    Nạp TƯỜNG MINH qua `AIC_ENV_FILE`, không tự dò `.env` trong thư mục hiện
    tại: file này chứa key thật, nên việc nó có được nạp hay không phải là một
    lựa chọn nhìn thấy được trên dòng lệnh chứ không phải tác dụng phụ của
    việc bạn đang đứng ở thư mục nào.

    `override=False` để biến đặt sẵn trên dòng lệnh luôn thắng file — cần cho
    ablation kiểu `AIC_ENABLE_EXPANSION=false python -m scripts.eval_kis ...`.
    Không thêm phụ thuộc `python-dotenv` cho một định dạng 6 dòng là đủ đọc.
    """

    applied: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        # `KEY = "giá trị"` và `KEY = 'giá trị'` đều gặp trong .env.fpt.local.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def _load_env_file_from_environ() -> None:
    """Đọc `AIC_ENV_FILE` nếu có. Thiếu file là LỖI, không phải bỏ qua âm thầm.

    Gõ sai đường dẫn mà vẫn khởi động được là đúng cái bẫy đang muốn tránh:
    server lên bình thường, mọi thứ FPT tắt hết, và không ai biết.
    """

    raw = os.getenv("AIC_ENV_FILE")
    if not raw:
        return
    path = Path(raw)
    if not path.exists():
        raise ValueError(f"AIC_ENV_FILE trỏ tới {path} nhưng file không tồn tại")
    load_env_file(path)


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
        "fpt_fast_llm_model": os.getenv("AIC_FPT_FAST_LLM_MODEL") or None,
        "fpt_vlm_model": os.getenv("AIC_FPT_VLM_MODEL") or None,
        "fpt_rerank_model": os.getenv("AIC_FPT_RERANK_MODEL") or None,
        "qa_llm_rank_mode": os.getenv("AIC_QA_LLM_RANK_MODE", "keep"),
        "avs_grade_mode": os.getenv("AIC_AVS_GRADE_MODE", "hard_gate"),
        "avs_soft_lambda": float(os.getenv("AIC_AVS_SOFT_LAMBDA", "0.3")),
        "avs_semantic_tau": float(os.getenv("AIC_AVS_SEMANTIC_TAU", "0.35")),
        "avs_max_per_video": _env_int("AIC_AVS_MAX_RESULTS_PER_VIDEO", 3),
        "fpt_qa_top_n": _env_int("AIC_FPT_QA_TOP_N", 5),
        "fpt_qa_max_tokens": _env_int("AIC_FPT_QA_MAX_TOKENS", 3000),
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
    # Mỗi frame là MỘT lệnh gọi FPT (model chỉ nhận 1 ảnh/prompt), nên số
    # này nhân thẳng vào chi phí và độ trễ của stage VLM: top-20 candidate
    # x 3 frame = 60 lệnh gọi cho một truy vấn.
    # Stage VLM rerank tốn `input_top_k x frames_per_candidate` lệnh gọi MỖI
    # truy vấn (top-20 x 3 frame = 60), nên phải bật có chủ đích chứ không
    # để mặc định: một lần chạy eval 23 truy vấn là ~1400 lệnh gọi.
    enable_vlm_rerank: bool = False
    rerank_vlm_frames_per_candidate: int = 3
    # Dịch truy vấn VI→EN trước khi vào text tower của CLIP. Mặc định TẮT
    # vì nó thêm một lệnh gọi LLM vào đường request và biến nhánh dense
    # thành phụ thuộc mạng — bật có chủ đích rồi đo, xem online/adapters/fpt_query.py.
    # Cache bản dịch trên đĩa: mỗi lần eval trước đây dịch lại cả 40 truy vấn
    # dù nhiệt độ 0 nên kết quả tất định. Cũng che được lỗi dịch ~1/40 lượt.
    query_translation_cache_dir: Path | None = Path("storage/cache/query_translation")
    enable_query_translation: bool = False
    # Mở rộng query bằng LLM (đồng nghĩa TIẾNG VIỆT) thay cho lexicon VI→EN
    # cứng, vốn đã lỗi thời từ khi caption chuyển sang tiếng Việt.
    enable_llm_expansion: bool = False
    # Tách keyword từ truy vấn trước khi vào nhánh `bm25_keyword`. Phía tài
    # liệu là nhãn object 2-3 từ, phía truy vấn là cả câu — lệch hạt.
    enable_keyword_extraction: bool = False
    # Deadline mỗi nhánh retrieval. Xem retrieval_orchestrator: nhánh nặng CPU
    # tranh chấp nhau và 3000ms là quá chặt khi corpus lớn lên.
    # Bỏ chữ lớp phủ khỏi index OCR: chuỗi có mặt ở hơn X tỉ lệ scene của
    # cùng video (logo đài, chữ chạy) không phân biệt được gì. None = tắt.
    ocr_overlay_df: float | None = None
    # >0: bỏ chuỗi OCR dài hơn ngần ấy từ (khung tiêu đề / chữ chạy).
    ocr_overlay_max_words: int = 0
    branch_timeout_ms: int = 8000
    playback_pad_sec: float = 5.0
    trake_engine: str = "sequences"
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
    # Model "nhanh" cho việc máy móc ngắn (dịch truy vấn, sinh từ đồng nghĩa).
    # PHẢI là model trả thẳng `content`, KHÔNG phải model reasoning: đo trên
    # FPT, Qwen3.6-27B tốn ~1650 token chỉ để dịch một câu, còn gemma-4-31B-it
    # tốn 9. Bỏ trống thì dùng lại `fpt_llm_model`.
    fpt_fast_llm_model: str | None = None
    fpt_vlm_model: str | None = None
    fpt_rerank_model: str | None = None
    # QA-JOINT-01: keep | scale | boost | answer_first (xem QaProcessor).
    qa_llm_rank_mode: str = "keep"
    # AVS-GRADE-01: hard_gate | no_gate | soft | semantic_or_lexical
    avs_grade_mode: str = "hard_gate"
    avs_soft_lambda: float = 0.3
    avs_semantic_tau: float = 0.35
    # Trần số kết quả MỖI VIDEO. Với dataset một video nó chặn cứng AVS ở 3
    # kết quả — khớp đúng `result_count` quan sát được (0,0,0,1,2,3,3,3).
    # `AIC_AVS_MAX_RESULTS_PER_VIDEO` đã có trong .env từ lâu nhưng chưa
    # từng được code đọc.
    avs_max_per_video: int = 3
    fpt_qa_top_n: int = 5
    # Ngân sách token cho một câu trả lời QA. Mặc định rộng vì model QA
    # thường là model reasoning — thiếu ngân sách thì `content` về None và
    # MỌI lệnh gọi QA đều hỏng (đã xảy ra thật với max_tokens=200).
    fpt_qa_max_tokens: int = 3000
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
        _load_env_file_from_environ()
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
            rerank_vlm_frames_per_candidate=_env_int(
                "AIC_RERANK_VLM_MAX_FRAMES_PER_CANDIDATE", 3
            ),
            enable_vlm_rerank=_env_bool("AIC_RERANK_VLM_ENABLED", False),
            query_translation_cache_dir=(
                Path(os.environ["AIC_QUERY_TRANSLATION_CACHE_DIR"])
                if os.getenv("AIC_QUERY_TRANSLATION_CACHE_DIR")
                else Path("storage/cache/query_translation")
            ),
            enable_query_translation=_env_bool("AIC_ENABLE_QUERY_TRANSLATION", False),
            enable_llm_expansion=_env_bool("AIC_ENABLE_LLM_EXPANSION", False),
            enable_keyword_extraction=_env_bool("AIC_ENABLE_KEYWORD_EXTRACTION", False),
            ocr_overlay_df=(
                float(os.environ["AIC_OCR_OVERLAY_DF"])
                if os.getenv("AIC_OCR_OVERLAY_DF") else None
            ),
            ocr_overlay_max_words=int(os.getenv("AIC_OCR_OVERLAY_MAX_WORDS", "0")),
            branch_timeout_ms=_env_int("AIC_BRANCH_TIMEOUT_MS", 8000),
            playback_pad_sec=float(os.getenv("AIC_PLAYBACK_PAD_SEC", "5.0")),
            trake_engine=os.getenv("AIC_TRAKE_ENGINE", "sequences").lower(),
            visual_embedding_model=os.getenv("AIC_VISUAL_EMBEDDING_MODEL", "openai/clip-vit-large-patch14"),
            visual_embedding_model_revision=os.getenv("AIC_VISUAL_EMBEDDING_MODEL_REVISION") or None,
            **_fpt_settings_kwargs(),
        )
