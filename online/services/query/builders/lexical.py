"""Lexical query builders cho BM25 caption / OCR / ASR.

- caption: giữ nguyên câu (BM25 tự lo tf-idf) + expansion synonym có kiểm soát
- ocr:     CHỈ chữ có thể hiện trên màn hình (đơn vị, cụm trong ngoặc kép)
- asr:     phần ngữ nghĩa của câu hỏi, bỏ mô tả thị giác thuần tuý

Mọi so khớp dùng `contains_term` (bỏ dấu + biên từ) — xem ghi chú trong
`online/services/query/normalize.py`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

from online.services.query.normalize import (
    SPACE_RE,
    contains_term,
    strip_question_words,
)


# Synonym cho caption BM25. Khoá viết không dấu; giá trị GIỮ dấu vì chúng được
# nối thẳng vào query và BM25 index là văn bản tiếng Việt có dấu.
# Tối đa 2 synonym mỗi concept — nhiều hơn thì query loãng.
SYNONYMS = {
    "ca": ("con cá", "fish"),
    "can": ("cân điện tử", "weighing scale"),
    "can dien tu": ("cân", "digital scale"),
    "dat": ("đặt lên", "placed on"),
    "cam": ("nắm", "holding"),
    "duoi": ("đuôi cá", "tail"),
    "hien thi": ("hiển thị số", "display"),
    "man hinh": ("màn hình số", "screen"),
    "con so": ("chỉ số", "number"),
    "nguoi": ("người đàn ông", "person"),
    "do": ("rót", "pouring"),
    "bat": ("tô", "bowl"),
    "nuoc": ("chất lỏng", "water"),
}

# Chữ có khả năng XUẤT HIỆN TRÊN MÀN HÌNH.
OCR_KEYWORDS = (
    "kg", "g", "gam", "gram", "ml", "lit", "km", "cm", "mm", "met",
    "diem", "phut", "giay", "gio", "do c", "vnd", "dong", "trieu",
    "so dien thoai", "bien so", "logo", "tieu de",
)

ASR_KEYWORDS = (
    "noi", "phat bieu", "tra loi", "phong van", "hoi thoai",
    "loi noi", "cau noi", "giong noi", "am thanh", "nghe thay",
)

# Từ mô tả thị giác thuần tuý — ASR gần như không bao giờ đọc lên.
VISUAL_ONLY_TERMS = (
    "hinh anh", "canh quay", "goc nhin", "ben trai", "ben phai",
    "phia sau", "phia truoc", "o giua", "mau sac", "cận cảnh", "can canh",
)


def build_caption_query(bundle: SearchQueryBundle) -> str:
    """Caption BM25: câu gốc + synonym của các concept thật sự có mặt."""

    query = bundle.normalized_query
    extras: list[str] = []
    for concept, synonyms in SYNONYMS.items():
        if contains_term(query, concept):
            extras.extend(synonyms)

    # Bản EN của visual query đã là các từ khoá rời — dùng luôn làm expansion.
    if bundle.visual_query_en:
        extras.extend(bundle.visual_query_en.split())

    combined = f"{query} {' '.join(dict.fromkeys(extras))}"
    return SPACE_RE.sub(" ", combined).strip()


# Số ĐÁNH DẤU của đề bài, không phải chữ trên màn hình:
#   "(1) Đoạn video..."      số thứ tự đoạn
#   "Tầng 1: 2 khối hộp"     số tầng / số lượng trong mô tả bố cục
#   "bước 3", "cảnh 2"       số bước
# Đưa chúng vào ocr_query là đưa nhiễu thuần tuý — MỌI video có chữ số đều
# khớp, và tín hiệu thật (vd "200 gr", chỉ có ở 1/873 video) bị chìm.
_MARKER_NUMBER_RE = re.compile(
    r"\(\s*\d+\s*\)"                                    # (1) (2)
    r"|(?:tầng|bước|cảnh|phần|mục|câu|hình|đoạn|lớp)\s*\d+"
    r"|\d+\s*(?=khối|hộp|cái|chiếc|người|con|tầng)",    # "2 khối hộp"
    flags=re.IGNORECASE,
)

# Số ĐÁNG tìm trên màn hình: có đơn vị đi kèm, hoặc đủ dài để là mã/giá/năm.
_CONTENT_NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:kg|g|gr|gam|mg|ml|l|km|cm|mm|m|%|độ|đồng|vnd|usd|tuổi|điểm)\b"
    r"|\b\d{3,}\b",
    flags=re.IGNORECASE,
)


def _content_numbers(query: str) -> list[str]:
    """Số THẬT SỰ có thể hiện trên màn hình, đã bỏ số đánh dấu của đề bài."""

    stripped = _MARKER_NUMBER_RE.sub(" ", query)
    return [match.strip() for match in _CONTENT_NUMBER_RE.findall(stripped)]


def build_ocr_query(bundle: SearchQueryBundle) -> str:
    """OCR: chỉ chữ kỳ vọng thấy trên màn hình, KHÔNG mô tả thị giác."""

    parts: list[str] = []

    # Cụm trong ngoặc kép là tín hiệu OCR mạnh nhất.
    parts.extend(bundle.exact_phrases)
    parts.extend(bundle.expected_units)
    parts.extend(keyword for keyword in OCR_KEYWORDS if contains_term(bundle.normalized_query, keyword))
    parts.extend(_content_numbers(bundle.normalized_query))

    # Truy vấn hỏi số mà không nêu đơn vị: đưa các đơn vị cân nặng thường gặp
    # vào để nhánh OCR còn có gì để khớp (số trên cân hầu như luôn kèm kg/g).
    if bundle.is_numeric_qa and not bundle.expected_units:
        parts.extend(("kg", "g"))

    ordered = list(dict.fromkeys(part for part in parts if part))
    if ordered:
        return " ".join(ordered)

    # Không suy ra được gì: KHÔNG dội cả câu mô tả vào OCR (đó là nguồn nhiễu
    # chính của nhánh này). Trả rỗng để nhánh tự bỏ qua.
    return ""


def build_asr_query(bundle: SearchQueryBundle) -> str:
    """ASR: phần ngữ nghĩa, bỏ các cụm chỉ tồn tại trong hình."""

    query = strip_question_words(bundle.normalized_query) or bundle.normalized_query
    words = query.split()
    kept = [
        word
        for word in words
        if not any(contains_term(word, term) for term in VISUAL_ONLY_TERMS)
    ]
    base = " ".join(kept) if kept else query

    extras = [keyword for keyword in ASR_KEYWORDS if contains_term(bundle.normalized_query, keyword)]
    combined = f"{base} {' '.join(extras)}" if extras else base
    return SPACE_RE.sub(" ", combined).strip()


__all__ = ["build_asr_query", "build_caption_query", "build_ocr_query"]
