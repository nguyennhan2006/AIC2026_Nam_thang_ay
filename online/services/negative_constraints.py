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


def extract_negative_constraints(query: str) -> list[str]:
    """Return the phrases the query says must NOT be present, normalized."""

    constraints: list[str] = []
    for match in _NEGATION_RE.findall(query):
        phrase = normalize_vi(match)
        if phrase:
            constraints.append(phrase)
    return constraints


def apply_negative_constraints(
    candidates: list[Candidate], documents: dict[str, SceneDocument], constraints: list[str],
) -> list[Candidate]:
    """Drop candidates whose scene text fully contains any constraint phrase."""

    word_sets = [words for item in constraints if (words := set(normalize_vi(item).split()))]
    if not word_sets:
        return candidates

    kept: list[Candidate] = []
    for candidate in candidates:
        document = documents.get(candidate.scene_id)
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
