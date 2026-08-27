"""Query intent classification — deterministic rule-based."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from online.services.query.models import AnswerType, QueryIntent

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

# Visual hints - query asks about what can be SEEN
VISUAL_HINTS = {
    "mau gi", "mau sac", "mau do", "mau xanh", "mau vang", "mau trang", "mau den",
    "hinh gi", "hinh dang", "kieu cach", "loai nao", "o dau", "o phia nao",
    "ben nao", "tren duoi", "trai phai", "o giua", "truoc sau",
    "ai dang", "nguoi nao", "con gi", "cai gi", "vat gi",
    "what color", "what shape", "what type", "what kind", "where is", "who is",
}

# OCR hints - query asks about TEXT
OCR_HINTS = {
    "ghi gi", "viet gi", "noi gi", "doc duoc", "chu gi", "dong chu",
    "bien", "bang", "khau hieu", "bien hieu", "bien bao", "nhan",
    "so dien thoai", "ten gi", "logo", "tieu de", "phu de",
    "text", "sign", "caption", "label", "written", "says",
}

# ASR hints - query asks about SPEECH
ASR_HINTS = {
    "noi gi", "phat bieu", "tra loi", "hoi", "duoc hoi", "phong van",
    "loi noi", "giọng", "hoi thoai", "doi thoai",
    "tieng", "am thanh", "nghe thay", "am",
    "says", "speaks", "speech", "heard", "said", "voice", "answer",
}

# Numeric question patterns
NUMERIC_PATTERNS = [
    "bao nhieu",
    "may",
    r"\d+\s*(kg|g|ml|l|diem|phut|giay|gio|ngay|thang|nam|tuoi|do|%)",
    r"gia\s+(bao|-la)",
    r"nang\s+bao",
    r"cao\s+bao",
    r"rong\s+bao",
    r"dai\s+bao",
    r"so\s+(la|tren)",
    "ket qua",
    r"thu\s+(may|bao)",
    "how many", "how much", "how long", "how far", "how tall",
]

# Answer type keywords
ANSWER_TYPE_KEYWORDS = {
    AnswerType.NUMERIC: ["bao nhieu", "may", "gia", "nang", "cao", "rong", "dai", "diem", "phut", "giay", "tuoi", "kg", "g"],
    AnswerType.COLOR: ["mau gi", "mau gi?", "to mau", "sac"],
    AnswerType.TEXT: ["ten", "ghi gi", "viet gi", "noi gi", "bien", "bang", "logo", "nhan"],
    AnswerType.OBJECT: ["con gi", "cai gi", "nguoi nao", "ai", "xe", "cay", "nha"],
    AnswerType.ACTION: ["lam gi", "dang lam", "da lam", "lam cach nao", "bang cach nao"],
    AnswerType.LOCATION: ["o dau", "vi tri", "dia diem", "noi nao", "cho nao"],
    AnswerType.PERSON: ["ai", "nguoi nao", "cua ai"],
    AnswerType.TIME: ["khi nao", "luc nao", "may gio", "ngay nao", "thang nao", "nam nao"],
}

# Units to extract
UNITS = {
    "weight": ["kg", "g", "mg", "tan", "pounds", "lbs", "oz"],
    "volume": ["ml", "l", "cc", "dl"],
    "score": ["diem", "diem so", "score"],
    "time": ["phut", "giay", "gio", "ngay", "thang", "nam"],
    "temperature": ["do", "°C", "°F", "℃"],
    "distance": ["km", "m", "cm", "mm"],
    "age": ["tuoi", "nam tuoi"],
}


def classify_intent(bundle: SearchQueryBundle) -> QueryIntent:
    """Classify query intent - which modality is PRIMARY for answering."""
    text = bundle.normalized_query.lower()

    # Check for numeric OCR first (high confidence)
    if is_numeric_question(text):
        return QueryIntent.NUMERIC_OCR

    # Count hints
    visual_count = sum(1 for h in VISUAL_HINTS if h in text)
    ocr_count = sum(1 for h in OCR_HINTS if h in text)
    asr_count = sum(1 for h in ASR_HINTS if h in text)

    # Determine dominant intent
    max_count = max(visual_count, ocr_count, asr_count)

    # If OCR phrase detected (e.g., "tren bien ghi...") -> OCR intent
    if any(hint in text for hint in ["ghi gi", "viet gi", "dong chu", "bien"]):
        return QueryIntent.OCR

    # If ASR phrase detected -> ASR intent
    if any(hint in text for hint in ["noi gi", "phat bieu", "tra loi"]):
        return QueryIntent.ASR

    # If strong visual question
    if visual_count >= 2:
        return QueryIntent.VISUAL

    # If has temporal structure
    if len(bundle.events) >= 2:
        return QueryIntent.TEMPORAL

    # Mixed if multiple modalities
    if max_count >= 1:
        intents = []
        if visual_count > 0:
            intents.append(QueryIntent.VISUAL)
        if ocr_count > 0:
            intents.append(QueryIntent.OCR)
        if asr_count > 0:
            intents.append(QueryIntent.ASR)

        if len(intents) > 1:
            return QueryIntent.MIXED
        if intents:
            return intents[0]

    return QueryIntent.VISUAL  # Default to visual


def classify_answer_type(bundle: SearchQueryBundle) -> AnswerType:
    """Classify expected answer type for QA tasks."""
    text = bundle.normalized_query.lower()

    # Check keyword patterns FIRST (before numeric) to avoid "gi" matching numeric
    for answer_type, keywords in ANSWER_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return answer_type

    # Numeric question - checked AFTER specific types
    if is_numeric_question(text):
        return AnswerType.NUMERIC

    # Color patterns
    color_words = ["mau", "to", "xanh", "do", "vang", "trang", "den", "cam", "tim", "hong"]
    if any(c in text for c in color_words):
        return AnswerType.COLOR

    # Object patterns
    object_words = ["con gi", "cai gi", "nguoi nao", "ai", "vat gi"]
    if any(o in text for o in object_words):
        return AnswerType.OBJECT

    return AnswerType.UNKNOWN


def is_numeric_question(text: str) -> bool:
    """Check if query is asking for a numeric answer."""
    text_lower = text.lower()

    # Check patterns
    for pattern in NUMERIC_PATTERNS:
        try:
            if re.search(pattern, text_lower):
                return True
        except re.error:
            # If pattern is not a valid regex, check as literal
            if pattern in text_lower:
                return True

    # Check units
    for unit_list in UNITS.values():
        if any(unit in text_lower for unit in unit_list):
            return True

    return False


def extract_expected_units(text: str) -> list[str]:
    """Extract expected units from query."""
    text_lower = text.lower()
    found = []

    for unit_type, unit_list in UNITS.items():
        for unit in unit_list:
            if unit in text_lower:
                found.append(unit)

    return list(set(found))
