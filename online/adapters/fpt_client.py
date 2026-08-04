"""Client OpenAI-compatible cho FPT AI Marketplace (PR-12).

Dùng tạm thay cho server A100 tự host (xem
`AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md`). Ba việc module này làm:

1. Gọi `/chat/completions` (LLM text hoặc VLM đa phương thức — cùng endpoint,
   khác ở `messages` có `image_url` hay không) và `/embeddings`.
2. Retry CHỈ lỗi transient (429/5xx/timeout/connection-reset), bounded, có
   jitter — đúng §21.3, cùng logic `RemoteInferenceProvider` đã có ở
   `offline/providers.py` (không phát minh chính sách retry mới).
3. Trả kèm `FptUsage` (token, latency, retry_count, cache_hit) cho MỌI call —
   PR-16 (cost accounting) đọc thẳng từ đây, không phải instrument lại.

**Không log `Authorization` hay body** — chỉ log độ dài, không log nội dung,
trừ khi `AIC_LOG_REQUEST_BODY=true` (mặc định false, xem `online/config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import ssl
from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from online.config import Settings

from online.adapters.provider_errors import (
    MalformedResponseError,
    ProviderError,
    ProviderTimeoutError,
    SchemaInvalidError,
    classify_http_status,
)


@dataclass(frozen=True, slots=True)
class FptUsage:
    """Kèm theo mọi response — nguồn duy nhất cho cost/latency accounting."""

    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retry_count: int
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class FptChatResult:
    text: str
    usage: FptUsage
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class FptEmbeddingResult:
    vector: list[float]
    usage: FptUsage


@dataclass(frozen=True, slots=True)
class RerankProbeResult:
    """Kết quả dò shape API rerank — §PR-12 mục 4 kế hoạch.

    FPT có thể có endpoint `/rerank` kiểu Cohere/Jina, hoặc không có gì và
    phải dùng LLM-as-reranker qua `/chat/completions`. Không đoán trước;
    `scripts/fpt_api_preflight.py` gọi `probe_rerank` để biết chắc.
    """

    native_rerank_available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class FptRerankResult:
    # scores[i] khớp documents[i] gốc (đã sắp lại theo `index` — FPT trả
    # `results` sắp theo relevance giảm dần, không theo thứ tự documents gửi
    # lên; xác nhận thật bằng probe PR-12: index trả về không tăng dần).
    scores: list[float]
    usage: FptUsage


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("***" if k.lower() == "authorization" else v) for k, v in headers.items()}


def _build_ssl_context() -> ssl.SSLContext:
    """`ssl.create_default_context()` nhưng bỏ `VERIFY_X509_STRICT`.

    Python 3.13+/OpenSSL 3.x bật cờ này theo mặc định: nó từ chối handshake
    nếu BẤT KỲ CA cert nào trong chain có basicConstraints không đánh dấu
    `critical` — một số proxy/antivirus chặn-giữa (TLS inspection) phát hành
    cert không tuân thủ chi tiết RFC 5280 này dù bản thân chain vẫn hợp lệ.
    Windows Schannel (dùng bởi curl.exe) không áp check này nên vẫn kết nối
    bình thường, còn Python thì lỗi "Basic Constraints of CA cert not marked
    critical" trên MỌI HTTPS request, không riêng FPT.
    CERT_REQUIRED và kiểm tra hostname vẫn giữ nguyên — chỉ bỏ đúng một pedantic
    check dôi thừa này, không tắt xác thực chứng chỉ.
    """

    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


class FptClient:
    """Client mỏng, retry bounded, không giữ state giữa các call."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_sec: float = 90.0,
        connect_timeout_sec: float = 10.0,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        retry_backoff_max_sec: float = 8.0,
        log_request_body: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("FPT API key is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.connect_timeout_sec = connect_timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.retry_backoff_max_sec = retry_backoff_max_sec
        self.log_request_body = log_request_body
        self._ssl_context = _build_ssl_context()

    # -- HTTP core ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _request_once(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Một lần gọi HTTP thô. Trả `(body_json, elapsed_ms)` hoặc raise ProviderError.

        KHÔNG retry ở đây — retry là trách nhiệm của `_call_with_retry`, để
        chính sách retry (transient-only, bounded, jittered) nằm ở một chỗ.
        """

        body = json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=body, headers=self._headers(), method="POST")
        started = perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_sec, context=self._ssl_context) as response:
                raw = response.read().decode("utf-8")
                elapsed_ms = int((perf_counter() - started) * 1000)
        except HTTPError as exc:
            elapsed_ms = int((perf_counter() - started) * 1000)
            error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            error_cls = classify_http_status(exc.code)
            raise error_cls(
                f"FPT API {path} -> HTTP {exc.code} sau {elapsed_ms}ms: {error_body[:500]}",
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError) as exc:
            elapsed_ms = int((perf_counter() - started) * 1000)
            raise ProviderTimeoutError(
                f"FPT API {path} không phản hồi sau {elapsed_ms}ms: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(
                f"FPT API {path} trả body không phải JSON hợp lệ: {raw[:300]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise MalformedResponseError(f"FPT API {path} trả JSON không phải object: {type(parsed).__name__}")
        return parsed, elapsed_ms

    def _call_with_retry(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Retry CHỈ lỗi transient. Trả `(body, elapsed_ms, retry_count)`.

        Cùng chính sách với `RemoteInferenceProvider._post`
        (`offline/providers.py`): backoff `min(max, base * 2**attempt)` +
        jitter, không retry lỗi permanent (401/403/404/422/schema).
        """

        last: ProviderError | None = None
        for attempt in range(self.max_retries):
            try:
                body, elapsed_ms = self._request_once(path, payload)
                return body, elapsed_ms, attempt
            except ProviderError as exc:
                if not exc.transient:
                    raise
                last = exc
            if attempt + 1 < self.max_retries:
                import time

                delay = min(self.retry_backoff_max_sec, self.retry_backoff_base_sec * 2**attempt)
                time.sleep(delay + random.random() * 0.2)
        assert last is not None
        raise last

    @classmethod
    def from_settings(cls, settings: "Settings") -> "FptClient":
        """Dựng client từ `online.config.Settings` — một chỗ duy nhất map
        field, để `scripts/fpt_api_preflight.py` và các script enrichment sau
        này (PR-13+) không tự lặp lại việc đọc field."""

        if not settings.fpt_enabled:
            raise ValueError("AIC_FPT_ENABLED is false — bật nó trước khi dựng FptClient")
        if not settings.fpt_api_key:
            raise ValueError("Settings.fpt_api_key is empty")
        return cls(
            settings.fpt_base_url,
            settings.fpt_api_key,
            timeout_sec=settings.fpt_timeout_sec,
            connect_timeout_sec=settings.fpt_connect_timeout_sec,
            max_retries=settings.fpt_max_retries,
            retry_backoff_base_sec=settings.fpt_retry_backoff_base_sec,
            retry_backoff_max_sec=settings.fpt_retry_backoff_max_sec,
            log_request_body=settings.log_request_body,
        )

    # -- Public API -----------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
        top_p: float = 1.0,
        response_format: dict[str, Any] | None = None,
    ) -> FptChatResult:
        """POST /chat/completions. Dùng cho cả LLM text lẫn VLM (`messages`
        chứa `image_url` thì model tự xử lý đa phương thức — cùng endpoint)."""

        payload: dict[str, Any] = {
            "model": model, "messages": messages,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        body, elapsed_ms, retries = self._call_with_retry("/chat/completions", payload)

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SchemaInvalidError(f"/chat/completions thiếu 'choices': {list(body.keys())}")
        message = choices[0].get("message", {})
        text = message.get("content")
        if not isinstance(text, str):
            raise SchemaInvalidError(f"/chat/completions choices[0].message.content không phải string: {type(text).__name__}")
        usage_raw = body.get("usage", {})
        usage = FptUsage(
            model_id=body.get("model", model),
            input_tokens=int(usage_raw.get("prompt_tokens", 0)),
            output_tokens=int(usage_raw.get("completion_tokens", 0)),
            latency_ms=elapsed_ms,
            retry_count=retries,
        )
        return FptChatResult(text=text, usage=usage, raw=body)

    def embedding(self, text: str, *, model: str) -> FptEmbeddingResult:
        """POST /embeddings — dùng nếu chọn nhánh dense qua text-embedding
        FPT (xem quyết định "Dense branch" trong kế hoạch PR-12..18)."""

        body, elapsed_ms, retries = self._call_with_retry(
            "/embeddings", {"model": model, "input": text}
        )
        data = body.get("data")
        if not isinstance(data, list) or not data or "embedding" not in data[0]:
            raise SchemaInvalidError(f"/embeddings thiếu 'data[0].embedding': {list(body.keys())}")
        vector = [float(x) for x in data[0]["embedding"]]
        usage_raw = body.get("usage", {})
        usage = FptUsage(
            model_id=body.get("model", model),
            input_tokens=int(usage_raw.get("prompt_tokens", 0)),
            output_tokens=0,
            latency_ms=elapsed_ms,
            retry_count=retries,
        )
        return FptEmbeddingResult(vector=vector, usage=usage)

    def rerank(self, query: str, documents: list[str], *, model: str) -> FptRerankResult:
        """POST /rerank — schema Cohere/Jina-style thật (xác nhận bằng
        `scripts/fpt_api_preflight` + probe thủ công PR-15): trả `results:
        [{index, relevance_score}]` sắp theo relevance giảm dần, KHÔNG theo
        thứ tự `documents` gửi lên. Sắp lại theo `index` để `scores[i]` khớp
        đúng `documents[i]` gốc — bắt buộc, không được giả định thứ tự giữ
        nguyên."""

        if not documents:
            return FptRerankResult(scores=[], usage=FptUsage(model, 0, 0, 0, 0))
        body, elapsed_ms, retries = self._call_with_retry(
            "/rerank", {"model": model, "query": query, "documents": documents}
        )
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise SchemaInvalidError(
                f"/rerank phải trả 'results' đúng {len(documents)} phần tử, "
                f"nhận được {type(results).__name__} độ dài "
                f"{len(results) if isinstance(results, list) else 'n/a'}"
            )
        scores = [0.0] * len(documents)
        seen = set()
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or index in seen or not (0 <= index < len(documents)):
                raise SchemaInvalidError(f"/rerank trả index không hợp lệ: {item!r}")
            seen.add(index)
            scores[index] = float(score)
        usage_raw = body.get("usage", {})
        usage = FptUsage(
            model_id=body.get("model", model),
            input_tokens=int(usage_raw.get("prompt_tokens", 0)),
            output_tokens=0,
            latency_ms=elapsed_ms,
            retry_count=retries,
        )
        return FptRerankResult(scores=scores, usage=usage)

    def probe_rerank(self, query: str, documents: list[str], *, model: str) -> RerankProbeResult:
        """Dò xem FPT có endpoint `/rerank` kiểu Cohere/Jina không.

        404/405 -> không có, phải dùng LLM-as-reranker qua `chat_completion`.
        Lỗi khác (auth/permission/5xx) -> ném nguyên, đó không phải câu trả
        lời "không có endpoint" mà là "có sự cố khác cần biết ngay".
        """

        try:
            body, _elapsed_ms, _retries = self._call_with_retry(
                "/rerank", {"model": model, "query": query, "documents": documents}
            )
        except ProviderError as exc:
            # 404 (route không tồn tại) hoặc 405 (method không hỗ trợ trên
            # route đó) đều nghĩa là "không có rerank endpoint" — không phải
            # sự cố provider cần dừng preflight lại.
            if exc.status_code in (404, 405):
                return RerankProbeResult(False, f"/rerank -> HTTP {exc.status_code}: {exc}")
            raise
        if "results" not in body:
            return RerankProbeResult(False, f"/rerank trả 200 nhưng thiếu 'results': {list(body.keys())}")
        return RerankProbeResult(True, "/rerank khả dụng, đúng schema Cohere/Jina-style")


def image_to_data_url(path: Path) -> str:
    """Base64 data URL cho `messages[].content[].image_url.url` (chuẩn OpenAI VLM)."""

    import base64

    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


__all__ = [
    "FptChatResult",
    "FptClient",
    "FptEmbeddingResult",
    "FptRerankResult",
    "FptUsage",
    "RerankProbeResult",
    "image_to_data_url",
]
