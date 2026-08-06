"""Negative constraints — Search Mixing Console W4/W5 (`enable_negative_constraints`).

Baseline: rule-based, no LLM. Detects an explicit negation phrase ("không
có X", "không X", "not X", "without X") and extracts X as a short phrase (up
to the next punctuation). A candidate is excluded outright — not
down-weighted — when every word of an extracted phrase appears in that
scene's own text (captions/object_labels/keywords/action_tags/ocr_texts),
matching the "must not contain" semantics of the plan's parsed-query
`negative_constraints` field. This is a hard filter, unlike
`online/services/rules.py`'s soft bonus/penalty.
"""

from __future__ import annotations

import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.models import Candidate, SceneDocument

_NEGATION_RE = re.compile(
    r"\b(?:không có|không|chẳng có|chẳng|without|not)\s+([^,.;:!?\n]+)", flags=re.IGNORECASE,
)

# `không` chỉ là phủ định khi nó đứng trước một vị từ. Nó cũng là ÂM TIẾT của
# nhiều danh từ ghép — và ở đó nghĩa hoàn toàn ngược lại. Đo trên 120 truy vấn
# gold: 3/5 constraint trích ra là dương tính giả, cả ba đều thuộc kiểu này.
#
#   "không gian bảo tàng"   -> cấm "gian bảo tàng"  (đúng ra là KHÔNG GIAN)
#   "bay trên không và mang túi nước" -> cấm "mang túi nước ..."
#
# Trường hợp thứ hai nguy hiểm nhất: cụm bị cấm chính là đặc điểm nhận dạng
# của đáp án đúng, mà đây lại là LỌC CỨNG — candidate đúng bị xoá khỏi pool.

# `không` + một trong các âm tiết này là danh từ ghép, không phải phủ định.
_COMPOUND_TAIL = frozenset("gian khí khi quân quan trung tặc tac vận van lực luc phận phan".split())

# Đứng sau các từ này thì `không` là danh từ ("trên không", "hàng không",
# "bầu không khí", "phòng không").
_COMPOUND_HEAD = frozenset("trên tren dưới duoi giữa giua hàng hang bầu bau phòng phong vùng vung".split())


def _is_compound_noun(query: str, match: re.Match[str]) -> bool:
    """`không` ở đây là âm tiết của danh từ ghép chứ không phải phủ định."""

    if normalize_vi(match.group(0).split()[0]) != "khong":
        return False  # "không có", "chẳng", "without", "not" — luôn là phủ định
    tail = match.group(1).split()
    if tail and normalize_vi(tail[0]) in {normalize_vi(w) for w in _COMPOUND_TAIL}:
        return True
    before = query[: match.start()].split()
    return bool(before) and normalize_vi(before[-1]) in {normalize_vi(w) for w in _COMPOUND_HEAD}


def iter_negative_constraints(query: str) -> list[tuple[str, tuple[int, int]]]:
    """`(phrase, span)` cho từng mệnh đề phủ định THẬT.

    Trả kèm span để bên gọi cắt đúng đoạn đã sinh ra constraint. `avs.py` từng
    cắt bằng cách chạy lại regex thô, nên nó cũng cắt luôn những cụm mà guard
    ở trên đã cố tình bỏ qua — hai nơi phải nhìn cùng một danh sách.
    """

    out: list[tuple[str, tuple[int, int]]] = []
    for match in _NEGATION_RE.finditer(query):
        if _is_compound_noun(query, match):
            continue
        phrase = normalize_vi(match.group(1))
        if phrase:
            out.append((phrase, match.span()))
    return out


def extract_negative_constraints(query: str) -> list[str]:
    """Return the phrases the query says must NOT be present, normalized."""

    return [phrase for phrase, _span in iter_negative_constraints(query)]


def apply_negative_constraints(
    candidates: list[Candidate], documents: dict[str, SceneDocument], constraints: list[str],
) -> list[Candidate]:
    """Drop candidates whose scene text fully contains any constraint phrase."""

    word_sets = [words for item in constraints if (words := set(normalize_vi(item).split()))]
    if not word_sets:
        return candidates

    kept: list[Candidate] = []
    for candidate in candidates:
        document = documents.get(candidate.scene_id or "")
        if document is None:
            kept.append(candidate)
            continue
        scene_words = set(normalize_vi(" ".join([
            *document.captions, *document.object_labels, *document.keywords,
            *document.action_tags, *document.ocr_texts,
        ])).split())
        if any(words <= scene_words for words in word_sets):
            continue
        kept.append(candidate)
    return [candidate.model_copy(update={"rank": rank}) for rank, candidate in enumerate(kept, start=1)]


__all__ = ["extract_negative_constraints", "apply_negative_constraints"]
