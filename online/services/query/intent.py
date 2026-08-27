"""Query intent classification - deterministic rule-based.

Mọi so khớp đi qua `contains_term` (bỏ dấu + biên từ). Bản trước dùng
`keyword in text` với keyword không dấu nên vừa bỏ sót ("ca" không khớp "cá")
vừa khớp nhầm ("ai" khớp trong "loài", "do" khớp trong "đó").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from online.services.query.models import AnswerType, QueryIntent
from online.services.query.normalize import contains_term, strip_diacritics

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

# OCR cue — truy vấn hỏi CHỮ trên màn hình.
OCR_HINTS = (
    "ghi gi", "viet gi", "doc duoc", "chu gi", "dong chu", "hang chu",
    "bien hieu", "bien bao", "bien so", "bang hieu", "khau hieu",
    "nhan", "logo", "tieu de", "phu de", "man hinh hien thi",
    "so dien thoai", "text", "sign", "caption", "subtitle", "label", "written",
)

# ASR cue — truy vấn hỏi LỜI NÓI.
ASR_HINTS = (
    "noi gi", "phat bieu", "tra loi", "duoc hoi", "phong van",
    "loi noi", "cau noi", "hoi thoai", "doi thoai", "giong noi",
    "nghe thay", "am thanh", "says", "speaks", "said", "voice", "interview",
)

# Visual cue — truy vấn hỏi thứ NHÌN THẤY.
VISUAL_HINTS = (
    "mau gi", "mau sac", "hinh gi", "hinh dang", "loai nao", "kieu gi",
    "dang lam gi", "lam gi", "con gi", "cai gi", "vat gi", "nguoi nao",
    "o dau", "ben nao", "phia nao", "mac gi", "deo gi", "cam gi",
    "what color", "what shape", "what kind", "where is", "doing what",
)

# Câu hỏi số. `bao nhieu`/`may` là cue mạnh nhất.
NUMERIC_HINTS = (
    "bao nhieu", "la bao nhieu", "may", "thu may", "so luong",
    "gia bao nhieu", "nang bao nhieu", "cao bao nhieu", "dai bao nhieu",
    "rong bao nhieu", "ket qua la", "con so", "chi so",
    "how many", "how much", "how long", "how tall", "how far",
)

# Đơn vị -> gợi ý OCR. Khớp theo biên từ nên "g" KHÔNG còn khớp mọi chữ g.
UNITS = {
    "weight": ("kg", "g", "gam", "gram", "kilogram", "tan", "lbs", "oz"),
    "volume": ("ml", "lit", "cc"),
    "score": ("diem", "score"),
    "time": ("phut", "giay", "gio", "ngay", "thang", "nam"),
    "temperature": ("do c", "°c", "°f"),
    "distance": ("km", "cm", "mm", "met"),
    "money": ("dong", "vnd", "usd", "trieu", "ty"),
    "age": ("tuoi",),
}

ANSWER_TYPE_KEYWORDS = {
    AnswerType.COLOR: ("mau gi", "mau sac", "mau nao"),
    AnswerType.LOCATION: ("o dau", "vi tri nao", "dia diem nao", "noi nao", "cho nao"),
    AnswerType.TIME: ("khi nao", "luc nao", "may gio", "ngay nao", "thang nao", "nam nao"),
    AnswerType.PERSON: ("la ai", "nguoi nao", "cua ai", "ai da", "ai la"),
    AnswerType.ACTION: ("lam gi", "dang lam gi", "hanh dong gi", "lam cach nao"),
    AnswerType.TEXT: ("ghi gi", "viet gi", "ten gi", "chu gi", "noi dung gi"),
    AnswerType.OBJECT: ("con gi", "cai gi", "vat gi", "loai gi", "loai nao"),
}


def _count_hints(text: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if contains_term(text, hint))


def is_numeric_question(text: str) -> bool:
    """Truy vấn có hỏi một CON SỐ không."""

    if any(contains_term(text, hint) for hint in NUMERIC_HINTS):
        return True
    # Đơn vị đứng một mình chưa đủ (vd "một tấn hàng" là mô tả, không phải hỏi).
    # Chỉ tính khi đi kèm cue hỏi.
    return False


def extract_expected_units(text: str) -> list[str]:
    """Đơn vị mà đáp án có thể mang — gợi ý cho nhánh OCR."""

    found: list[str] = []
    for units in UNITS.values():
        found.extend(unit for unit in units if contains_term(text, unit))
    return list(dict.fromkeys(found))


def classify_answer_type(bundle: SearchQueryBundle) -> AnswerType:
    """Kiểu đáp án mong đợi. Cue CỤ THỂ được xét TRƯỚC cue số.

    Thứ tự này quan trọng: "Cốc cuối cùng có màu gì?" phải ra COLOR chứ không
    phải NUMERIC chỉ vì đâu đó có từ mang nghĩa lượng.
    """

    text = bundle.normalized_query

    for answer_type, keywords in ANSWER_TYPE_KEYWORDS.items():
        if any(contains_term(text, keyword) for keyword in keywords):
            return answer_type

    if is_numeric_question(text):
        return AnswerType.NUMERIC

    return AnswerType.UNKNOWN


def classify_intent(bundle: SearchQueryBundle) -> QueryIntent:
    """Modality NÀO là chính để trả lời truy vấn này."""

    text = bundle.normalized_query
    answer_type = classify_answer_type(bundle)

    # Số hiển thị trên màn hình -> OCR đọc số, nhưng visual vẫn phải định vị cảnh.
    if answer_type == AnswerType.NUMERIC:
        return QueryIntent.NUMERIC_OCR

    ocr_count = _count_hints(text, OCR_HINTS)
    asr_count = _count_hints(text, ASR_HINTS)
    visual_count = _count_hints(text, VISUAL_HINTS)

    if ocr_count and ocr_count >= asr_count and ocr_count >= visual_count:
        return QueryIntent.OCR
    if asr_count and asr_count >= visual_count:
        return QueryIntent.ASR
    if len(bundle.events) >= 2 and not visual_count:
        return QueryIntent.TEMPORAL
    if visual_count and (ocr_count or asr_count):
        return QueryIntent.MIXED
    return QueryIntent.VISUAL


__all__ = [
    "classify_answer_type",
    "classify_intent",
    "extract_expected_units",
    "is_numeric_question",
]
