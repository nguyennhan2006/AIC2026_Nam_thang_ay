"""Rule-based action tag extraction — Search Mixing Console W1 (action tags).

Baseline V1 does not run an action-recognition model. It matches a curated
Vietnamese/English action-verb lexicon against caption text already produced
during keyframe enrichment — the same "cheap baseline first" approach used
for color features and clip pooling. Whether a real action-recognition model
is worth its cost is an open research-agenda question (see
docs/15_RESEARCH_AGENDA.md), not something to guess at here.

Multi-word surface forms (e.g. "vẫy tay") are matched as substrings of the
casefolded caption text; single-word forms are matched as whole tokens so a
short verb like "đi" cannot spuriously match inside an unrelated word.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+")

ACTION_LEXICON: dict[str, list[str]] = {
    "walking": ["đi bộ", "đi", "walking", "walk"],
    "running": ["chạy", "running", "run"],
    "standing": ["đứng", "standing", "stand"],
    "sitting": ["ngồi", "sitting", "sit"],
    "waving": ["vẫy tay", "vẫy", "waving", "wave"],
    "talking": ["nói chuyện", "nói", "talking", "speaking", "speak"],
    "holding": ["cầm", "giữ", "holding", "hold"],
    "cooking": ["nấu ăn", "nấu", "cooking", "cook"],
    "raking": ["cào", "raking", "rake"],
    "riding": ["đi xe", "cưỡi", "riding", "ride"],
    "driving": ["lái xe", "driving", "drive"],
    "dancing": ["nhảy múa", "múa", "dancing", "dance"],
    "singing": ["hát", "singing", "sing"],
    "eating": ["ăn", "eating", "eat"],
    "drinking": ["uống", "drinking", "drink"],
    "writing": ["viết", "writing", "write"],
    "reading": ["đọc", "reading", "read"],
    "playing": ["chơi", "playing", "play"],
    "working": ["làm việc", "working", "work"],
    "greeting": ["chào hỏi", "chào", "greeting", "greet"],
}


def extract_action_tags(caption_text: str) -> list[str]:
    """Return the sorted, deduplicated action tags found in `caption_text`."""

    if not caption_text:
        return []
    text = caption_text.casefold()
    tokens = set(TOKEN_RE.findall(text))
    tags: set[str] = set()
    for tag, surface_forms in ACTION_LEXICON.items():
        for form in surface_forms:
            form_cf = form.casefold()
            matched = form_cf in tokens if " " not in form_cf else form_cf in text
            if matched:
                tags.add(tag)
                break
    return sorted(tags)


__all__ = ["ACTION_LEXICON", "extract_action_tags"]
