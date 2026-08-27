"""Query normalization - basic preprocessing without model calls.

QUAN TRỌNG (bug 2026-08-27): mọi so khớp từ khoá ở tầng này phải đi qua
`strip_diacritics` VÀ có biên từ. Bản trước viết danh sách keyword bằng ASCII
không dấu rồi `in` thẳng vào query có dấu:

    "ca" in "hình ảnh một con cá..."   -> False  (mất hết entity thật)
    "ao" in "... là bao nhiêu?"        -> True   (khớp nhầm trong "bao")

nên visual query của truy vấn cá/cân ra đúng hai chữ "ao sau".
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

# Patterns for basic extraction
SPACE_RE = re.compile(r"\s+")
QUOTED_RE = re.compile(r'["“”‘’]([^"“”‘’]{2,}?)["“”‘’]')
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Câu hỏi trừu tượng — KHÔNG mang thông tin thị giác, bỏ khỏi visual query.
# Viết bằng dạng KHÔNG DẤU: mọi text đều được `strip_diacritics` trước khi khớp.
ABSTRACT_PATTERNS = [
    r"\bla\s+bao\s+nhieu\b",
    r"\bbao\s+nhieu\b",
    # "<danh từ> gì" — thứ được hỏi là ẨN SỐ, không phải mô tả. "Cốc … có màu
    # gì?" thì "cốc" mới là tín hiệu thị giác; "màu gì" không mô tả màu nào cả.
    r"\b(?:mau|hinh dang|hinh|loai|kieu|ten|chu|noi dung|hanh dong|so)\s+gi\b",
    r"\b(?:con|cai|vat|nguoi)\s+gi\b",
    r"\bco\s+gi\b",
    r"\bla\s+gi\b",
    r"\bla\s+ai\b",
    r"\bcua\s+ai\b",
    r"\bo\s+dau\b",
    r"\bkhi\s+nao\b",
    r"\bnhu\s+the\s+nao\b",
    r"\btai\s+sao\b",
    r"\bvi\s+sao\b",
    r"\bco\s+phai\s+khong\b",
    r"\bhay\s+cho\s+biet\b",
    r"\bhoi\b",
    r"\bwhat\s+is\b",
    r"\bhow\s+(?:many|much|long|far|tall)\b",
    r"\bwho\s+(?:is|was|are|were)\b",
    r"\bwhy\b",
]

# Từ mở đầu vô nghĩa với visual retrieval ("Hình ảnh ...", "Tìm cảnh ...").
LEAD_IN_PATTERNS = [
    r"^\s*hinh\s+anh\s+(?:cua\s+)?",
    r"^\s*tim\s+(?:canh|video|khoanh\s+khac)\s+",
    r"^\s*canh\s+quay\s+",
    r"^\s*find\s+(?:the\s+)?(?:scene|video|moment)\s+",
]

# Strong temporal markers - CHẮC CHẮN là transition (dạng không dấu).
STRONG_TEMPORAL = {
    "sau do", "tiep theo", "ke tiep", "roi thi",
    "then", "next", "after that",
}

# Weak temporal markers - có thể là ĐỊNH NGỮ của danh từ, không phải marker.
# "Con số cuối cùng trên cân" = attribute; "Cuối cùng, ông ấy đi vào" = temporal.
WEAK_TEMPORAL = {
    "cuoi cung", "finally",
}


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp keyword; giữ nguyên độ dài từ.

    `đ`/`Đ` không phải tổ hợp dấu Unicode nên phải thay tay.
    """

    lowered = text.casefold().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_query(text: str) -> str:
    """Normalize query: NFC + gộp khoảng trắng. GIỮ NGUYÊN dấu và hoa/thường."""

    normalized = unicodedata.normalize("NFC", text).strip()
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized


def extract_quotes(text: str) -> list[str]:
    """Extract quoted phrases for OCR exact matching."""

    return [item.strip() for item in QUOTED_RE.findall(text) if item.strip()]


def extract_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def contains_term(text: str, term: str) -> bool:
    """Có chứa `term` như một CỤM TỪ trọn vẹn, bỏ qua dấu.

    Dùng thay cho `term in text`: `"ao" in "bao nhiêu"` là True và chính nó
    sinh ra visual query rác.
    """

    haystack = strip_diacritics(text)
    needle = strip_diacritics(term)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def strip_question_words(text: str) -> str:
    """Bỏ phần hỏi trừu tượng, GIỮ NGUYÊN phần mô tả thị giác.

    "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
    -> "Con số hiển thị cuối cùng trên cân"

    Khớp trên bản không dấu nhưng CẮT trên chuỗi gốc (hai chuỗi cùng độ dài
    vì `strip_diacritics` chỉ bỏ ký tự tổ hợp, không đổi số ký tự cơ sở).
    """

    result = text
    for pattern in LEAD_IN_PATTERNS + ABSTRACT_PATTERNS:
        while True:
            match = re.search(pattern, strip_diacritics(result))
            if match is None:
                break
            result = result[: match.start()] + " " + result[match.end() :]
    return SPACE_RE.sub(" ", result).strip(" ,.;:?!")


def split_temporal_weak(text: str) -> tuple[str, str]:
    """Tách theo marker YẾU ("cuối cùng"), chỉ khi nó đứng đầu mệnh đề.

    "Người A đổ nước, sau đó B rót. Cuối cùng, C uống."
    -> ("Người A đổ nước, sau đó B rót.", "C uống")

    "Con số cuối cùng trên cân là bao nhiêu?"
    -> (nguyên câu, "")   # KHÔNG tách: đây là định ngữ của "con số"

    Returns:
        (target_query, context_query); context rỗng nghĩa là không tách.
    """

    ascii_text = strip_diacritics(text)
    weak_pattern = r"(?<=[,;:.])\s*(?:cuoi cung|finally)\s*,?\s*"
    matches = list(re.finditer(weak_pattern, ascii_text))
    if not matches:
        return text, ""

    first = matches[0]
    before = text[: first.start()].strip()
    after = text[first.end() :].strip()
    if before and after:
        return before, after
    return text, ""


def token_count(text: str) -> int:
    return len(text.split())


def estimate_complexity(bundle: SearchQueryBundle) -> int:
    """0-1 đơn giản, 2-3 trung bình, 4+ phức tạp."""

    complexity = 0
    if token_count(bundle.normalized_query) > 25:
        complexity += 1
    if token_count(bundle.normalized_query) > 50:
        complexity += 1
    if len(bundle.events) >= 2:
        complexity += 1
    if len(bundle.events) >= 3:
        complexity += 1
    if bundle.exact_phrases:
        complexity += 1
    if bundle.answer_type.name == "NUMERIC":
        complexity += 1
    return min(complexity, 5)
