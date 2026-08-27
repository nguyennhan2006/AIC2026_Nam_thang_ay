"""Visual query builder cho Jina CLIP v2.

Nguyên tắc (sửa 2026-08-27):

BẢN TRƯỚC dựng lại visual query bằng cách TRÍCH danh từ/động từ khớp một
danh sách keyword cố định rồi nối chúng lại. Cách đó hỏng theo hai đường:

  1. Corpus mở có hàng nghìn danh từ; danh sách nào cũng thủng. Truy vấn nào
     không có từ trong list thì visual query gần như rỗng.
  2. Nối keyword rời phá cấu trúc câu — Jina CLIP là encoder CÂU, "cá cân
     người cầm" mất hết quan hệ so với "con cá đặt trên cân".

BẢN NÀY giữ nguyên câu mô tả, chỉ CẮT phần hỏi trừu tượng ("là bao nhiêu",
"là gì") và phần dẫn ("Hình ảnh ..."). Đây là phép trừ, không phải phép dựng
lại, nên không phụ thuộc độ phủ của bất kỳ từ điển nào.

Từ điển VN->EN chỉ còn dùng cho `visual_query_en` (augmentation phụ) và để
trích entity phục vụ debug — KHÔNG còn nằm trên đường sinh visual query chính.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

from online.services.query.normalize import (
    SPACE_RE,
    contains_term,
    strip_diacritics,
    strip_question_words,
)


# Gợi ý dịch cho `visual_query_en`. Khoá viết KHÔNG DẤU vì so khớp chạy trên
# bản đã `strip_diacritics`. Thiếu từ ở đây chỉ làm bản EN nghèo hơn, KHÔNG
# ảnh hưởng visual query tiếng Việt.
VN_TO_EN = {
    "con ca": "fish",
    "ca": "fish",
    "can dien tu": "digital scale",
    "can": "weighing scale",
    "man hinh": "screen",
    "con so": "number",
    "duoi": "tail",
    "nguoi dan ong": "man",
    "nguoi phu nu": "woman",
    "dan ong": "man",
    "phu nu": "woman",
    "tre em": "child",
    "nguoi": "person",
    "dat": "placed",
    "cam": "holding",
    "hien thi": "displaying",
    "do": "pouring",
    "bat": "bowl",
    "coc": "cup",
    "nuoc": "water",
    "chat long": "liquid",
    "ban tay": "hand",
    "dong ho": "watch",
    "xe": "car",
    "nha": "house",
    "cay": "tree",
    "duong": "road",
    "song": "river",
    "bien": "sea",
    "trau": "buffalo",
    "muoi": "salt",
    "vay tay": "waving",
    "hoc sinh": "student",
    "giao vien": "teacher",
    "bang": "board",
    "ao": "shirt",
    "mau do": "red",
    "mau xanh": "blue",
    "mau vang": "yellow",
    "mau trang": "white",
    "mau den": "black",
}

# Entity/action/attribute — CHỈ để hiển thị debug và làm secondary query.
VISUAL_NOUNS = (
    "con ca", "ca", "can dien tu", "can", "man hinh", "con so", "duoi",
    "nguoi dan ong", "nguoi phu nu", "dan ong", "phu nu", "tre em", "nguoi",
    "bat", "coc", "ly", "nuoc", "chat long", "ban tay", "dong ho",
    "xe", "o to", "nha", "can nha", "cay", "hoa", "qua", "duong", "song",
    "bien", "trau", "bo", "muoi", "laptop", "may tinh", "ban", "ghe",
    "truong", "lop", "hoc sinh", "giao vien", "bang", "ao", "quan",
)

# Động từ đơn âm tiết trùng với hư từ tiếng Việt ("đó", "đi", "an") gây khớp
# nhầm ngay cả khi có biên từ, nên chỉ nhận chúng ở dạng CỤM.
ACTION_VERBS = (
    "dat len", "dat tren", "dat", "cam", "nam giu", "hien thi",
    "do nuoc", "do chat long", "rot", "uong", "an com",
    "nem", "quang", "chay", "nhay", "ngoi", "dung", "vay tay",
    "cao", "cuoc", "lai xe", "deo", "mang", "om", "boi", "bay",
)

SPATIAL_TERMS = (
    "ben tren", "ben duoi", "ben trai", "ben phai", "o giua",
    "tren", "duoi", "truoc", "sau", "trai", "phai", "giua",
)

COLOR_TERMS = (
    "mau do", "mau xanh", "mau vang", "mau trang", "mau den",
    "mau cam", "mau tim", "mau hong", "mau nau", "mau xam",
)


def build_visual_query(bundle: SearchQueryBundle) -> tuple[str, str]:
    """Visual query cho Jina CLIP v2: GIỮ câu, chỉ trừ phần hỏi trừu tượng.

    Returns:
        (visual_query_vn, visual_query_en)
    """

    stripped = strip_question_words(bundle.normalized_query)

    # Phép trừ có thể ăn hết câu với truy vấn thuần câu hỏi ("Là gì?").
    # Khi đó thà giữ nguyên câu còn hơn đưa chuỗi rỗng vào encoder.
    visual_vn = stripped if len(stripped.split()) >= 2 else bundle.normalized_query

    return visual_vn.strip(), translate_to_english(visual_vn).strip()


def extract_visual_entities(text: str) -> list[str]:
    """Entity xuất hiện trong query — dùng cho debug UI, khớp theo biên từ."""

    found = [noun for noun in VISUAL_NOUNS if contains_term(text, noun)]
    # Bỏ cụm bị chứa trong cụm dài hơn đã khớp ("ca" khi đã có "con ca").
    return _drop_subsumed(found)


def extract_actions(text: str) -> list[str]:
    return _drop_subsumed([verb for verb in ACTION_VERBS if contains_term(text, verb)])


def extract_attributes(text: str) -> list[str]:
    colors = [term for term in COLOR_TERMS if contains_term(text, term)]
    spatial = [term for term in SPATIAL_TERMS if contains_term(text, term)]
    return _drop_subsumed(colors + spatial)


def _drop_subsumed(terms: list[str]) -> list[str]:
    """Bỏ cụm ngắn đã nằm trong cụm dài hơn cùng danh sách."""

    ordered = sorted(dict.fromkeys(terms), key=len, reverse=True)
    kept: list[str] = []
    for term in ordered:
        if not any(
            term != longer and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", longer)
            for longer in kept
        ):
            kept.append(term)
    return kept


def translate_to_english(vn_text: str) -> str:
    """Dịch thô VN->EN cho augmentation; khớp trên bản không dấu.

    Chỉ dịch được phần có trong từ điển, phần còn lại bị bỏ — bản EN vì thế
    là tín hiệu PHỤ, không bao giờ thay thế bản tiếng Việt.
    """

    if not vn_text.strip():
        return ""

    ascii_text = strip_diacritics(vn_text)
    pieces: list[str] = []
    for vn, en in sorted(VN_TO_EN.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(vn)}(?!\w)", ascii_text):
            pieces.append(en)
            # Xoá để cụm ngắn hơn không khớp lại phần đã dịch.
            ascii_text = re.sub(rf"(?<!\w){re.escape(vn)}(?!\w)", " ", ascii_text)

    return SPACE_RE.sub(" ", " ".join(pieces)).strip()


def build_multi_visual_queries(bundle: SearchQueryBundle) -> list[dict]:
    """Nhiều visual query cho truy vấn phức tạp (full / target / context).

    Fusion nên lấy `max` các điểm, xem docs — full giữ recall, target giữ
    độ chính xác của khoảnh khắc đích.
    """

    queries: list[dict] = []
    full_vn, full_en = build_visual_query(bundle)
    if full_vn:
        queries.append({"query_vn": full_vn, "query_en": full_en, "type": "full"})

    for part, kind in ((bundle.target_query, "target"), (bundle.context_query, "context")):
        if not part:
            continue
        part_vn = strip_question_words(part)
        if not part_vn or part_vn == full_vn:
            continue
        queries.append({
            "query_vn": part_vn,
            "query_en": translate_to_english(part_vn),
            "type": kind,
        })
    return queries
