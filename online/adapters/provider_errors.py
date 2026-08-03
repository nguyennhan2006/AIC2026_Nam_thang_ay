"""Error taxonomy cho external inference provider (PR-12).

Khớp đúng §24 "Provider/system" của
`AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md`. Tách riêng khỏi
`online/errors.py` (lỗi API nội bộ) vì đây là lỗi của MỘT PHÍA THỨ BA — cần
phân biệt rõ để retry đúng chính sách (§21.3): chỉ retry lỗi transient
(429/5xx/timeout/connection-reset), không bao giờ retry lỗi permanent
(401/403/404-model/422/schema sai).
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base cho mọi lỗi gọi provider ngoài. `code` khớp taxonomy §24."""

    code = "provider_error"
    # True nếu lỗi này ĐÁNG retry (transient); False nếu retry vô ích (permanent).
    transient = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        # HTTP status gốc (None nếu lỗi không tới từ HTTP, vd JSON hỏng sau
        # response 200). Giữ lại để caller phân biệt 404 route-not-found với
        # 405 method-not-allowed mà không phải string-match message lỗi.
        self.status_code = status_code


class AuthError(ProviderError):
    code = "AUTH_ERROR"


class PermissionDeniedError(ProviderError):
    code = "PERMISSION_DENIED"


class ModelNotFoundError(ProviderError):
    code = "MODEL_NOT_FOUND"


class RateLimitedError(ProviderError):
    code = "RATE_LIMITED"
    transient = True


class ProviderTimeoutError(ProviderError):
    code = "TIMEOUT"
    transient = True


class UpstreamServerError(ProviderError):
    code = "UPSTREAM_5XX"
    transient = True


class MalformedResponseError(ProviderError):
    """HTTP OK nhưng body không phải JSON hợp lệ, hoặc thiếu field bắt buộc."""

    code = "MALFORMED_RESPONSE"


class SchemaInvalidError(ProviderError):
    """JSON hợp lệ nhưng không khớp schema kỳ vọng (vd thiếu key `choices`)."""

    code = "SCHEMA_INVALID"


def classify_http_status(status: int) -> type[ProviderError]:
    """Map HTTP status -> lớp lỗi taxonomy. Dùng bởi mọi provider client."""

    if status in (401,):
        return AuthError
    if status in (403,):
        return PermissionDeniedError
    if status == 404:
        return ModelNotFoundError
    if status == 429:
        return RateLimitedError
    if 500 <= status < 600:
        return UpstreamServerError
    # 400/422/... : lỗi request/schema phía mình, không phải lỗi provider —
    # coi là SchemaInvalidError vì gần như luôn do payload gửi lên sai contract.
    return SchemaInvalidError


__all__ = [
    "AuthError",
    "MalformedResponseError",
    "ModelNotFoundError",
    "PermissionDeniedError",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "SchemaInvalidError",
    "UpstreamServerError",
    "classify_http_status",
]
