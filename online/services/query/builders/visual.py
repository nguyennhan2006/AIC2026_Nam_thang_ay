"""Visual query builder for Jina CLIP v2.

Principles:
- Visual query only contains VISUAL INFORMATION: objects, actions, scenes, spatial relations
- REMOVE: abstract questions ("bao nhieu", "la gi", "tai sao")
- KEEP: nouns, verbs, adjectives about appearance, colors, positions
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

from online.services.query.normalize import (
    QUESTION_WORDS,
    SPACE_RE,
    strip_question_words,
)


# English translation hints for visual concepts (ASCII Vietnamese for encoding safety)
VN_TO_EN = {
    "con ca": "fish",
    "ca": "fish",
    "can": "weighing scale",
    "can dien tu": "digital scale",
    "nguoi": "person",
    "nguoi dan ong": "man",
    "nguoi phu nu": "woman",
    "dan ong": "man",
    "phu nu": "woman",
    "tre em": "child",
    "tre": "child",
    "dat": "placed",
    "cam": "holding",
    "nam": "holding",
    "duoi": "tail",
    "hien thi": "display",
    "so": "number",
    "man hinh": "screen",
    "ao": "shirt",
    "quan": "pants",
    "mau": "color",
    "do": "red",
    "xanh": "blue",
    "vang": "yellow",
    "trang": "white",
    "den": "black",
    "cam": "orange",
    "tim": "purple",
    "hong": "pink",
    "xe": "car",
    "o to": "car",
    "may": "machine",
    "nuoc": "water",
    "do": "pouring",
    "coc": "cup",
    "ly": "glass",
    "chat long": "liquid",
    "nha": "house",
    "can nha": "house",
    "duong": "road",
    "bo": "shore",
    "song": "river",
    "bien": "sea",
    "dan": "herd",
    "bay": "flock",
    "bo": "cattle",
    "trau": "buffalo",
    "bay trau": "buffalo herd",
    "nguoi cao": "person raking",
    "cao": "raking",
    "muoi": "salt",
    "doan nguoi": "group of people",
    "vay tay": "waving",
    "di": "walking",
    "vao": "entering",
    "cay": "tree",
    "hoa": "flower",
    "qua": "fruit",
    "chuoi": "banana",
    "laptop": "laptop",
    "may tinh": "computer",
    "ban": "table",
    "ghe": "chair",
    "truong": "school",
    "lop": "classroom",
    "hoc sinh": "student",
    "giao vien": "teacher",
    "bang": "blackboard",
    "trinh chieu": "presentation",
    "anh": "photo",
    "hinh": "image",
}


# Visual concepts to extract (ASCII Vietnamese)
VISUAL_NOUNS = {
    "nguoi", "nguoi dan ong", "phu nu", "tre em", "dan ong", "phu nu",
    "con ca", "ca", "xe", "o to", "may", "nuoc", "chat long", "coc", "ly",
    "nha", "can nha", "duong", "bo", "song", "bien", "bay trau", "trau", "bo",
    "cay", "hoa", "qua", "chuoi", "laptop", "may tinh", "ban", "ghe",
    "truong", "lop", "hoc sinh", "giao vien", "bang", "ao", "quan",
    "can", "can dien tu", "man hinh", "duoi",
}

# Action verbs (ASCII Vietnamese)
ACTION_VERBS = {
    "dat", "cam", "nam", "cam", "hien thi", "do", "nem", "quang",
    "di", "chay", "nhay", "ngoi", "dung", "vay tay", "cao", "cuoc",
    "noi", "hat", "nhin", "chi", "chi tay", "rot", "muc", "uong",
    "an", "nau", "lai", "ngoi", "deo", "mang", "cam", "om",
}

# Spatial relations (ASCII Vietnamese)
SPATIAL_TERMS = {
    "tren", "duoi", "truoc", "sau", "trai", "phai", "giua",
    "ben tren", "ben duoi", "ben trai", "ben phai",
    "o giua", "o tren", "o duoi",
    "tren cung", "duoi cung",
}


def build_visual_query(bundle: SearchQueryBundle) -> tuple[str, str]:
    """Build visual query for Jina CLIP v2.

    Returns:
        Tuple of (visual_query_vn, visual_query_en).
    """
    # Start from normalized query
    query = bundle.normalized_query

    # Step 1: Remove abstract question patterns
    query = strip_question_words(query)

    # Step 2: Extract visual entities
    entities = extract_visual_entities(query)

    # Step 3: Extract actions
    actions = extract_actions(query)

    # Step 4: Extract attributes
    attributes = extract_attributes(query)

    # Step 5: Build Vietnamese visual query
    visual_parts = []
    if entities:
        visual_parts.extend(entities)
    if actions:
        visual_parts.extend(actions)
    if attributes:
        visual_parts.extend(attributes)

    visual_vn = " ".join(visual_parts) if visual_parts else query

    # Step 6: Build English translation
    visual_en = translate_to_english(visual_vn)

    return visual_vn.strip(), visual_en.strip()


def extract_visual_entities(text: str) -> list[str]:
    """Extract visual entities from query."""
    text_lower = text.lower()
    found = []

    # Multi-word entities first (longer match)
    multi_word = sorted(VISUAL_NOUNS, key=len, reverse=True)
    for noun in multi_word:
        if noun in text_lower:
            found.append(noun)

    return list(dict.fromkeys(found))  # Preserve order, remove duplicates


def extract_actions(text: str) -> list[str]:
    """Extract action verbs from query."""
    text_lower = text.lower()
    found = []

    for verb in ACTION_VERBS:
        # Match whole word
        pattern = r'\b' + re.escape(verb) + r'\b'
        if re.search(pattern, text_lower):
            found.append(verb)

    return list(dict.fromkeys(found))


def extract_attributes(text: str) -> list[str]:
    """Extract visual attributes (colors, sizes, positions)."""
    text_lower = text.lower()
    found = []

    # Colors
    colors = ["do", "xanh", "vang", "trang", "den", "cam", "tim", "hong", "nau", "xam"]
    for color in colors:
        if color in text_lower:
            found.append(color)

    # Spatial relations
    for spatial in SPATIAL_TERMS:
        if spatial in text_lower:
            found.append(spatial)

    return list(dict.fromkeys(found))


def translate_to_english(vn_text: str) -> str:
    """Simple rule-based Vietnamese to English translation for visual terms."""
    if not vn_text.strip():
        return ""

    result = vn_text
    # Sort by length (longest first) to avoid partial matches
    sorted_terms = sorted(VN_TO_EN.items(), key=lambda x: len(x[0]), reverse=True)

    for vn, en in sorted_terms:
        result = re.sub(r'\b' + re.escape(vn) + r'\b', en, result, flags=re.IGNORECASE)

    # Clean up multiple spaces
    result = SPACE_RE.sub(" ", result).strip()

    return result


def build_multi_visual_queries(bundle: SearchQueryBundle) -> list[dict]:
    """Build multiple visual queries for complex queries.

    For queries with multiple events or complex structure,
    create separate queries for different parts.

    Returns:
        List of dicts with 'query_vn', 'query_en', 'type' (full/target/context).
    """
    queries = []

    # Full query (compressed)
    visual_full_vn, visual_full_en = build_visual_query(bundle)
    if visual_full_vn:
        queries.append({
            "query_vn": visual_full_vn,
            "query_en": visual_full_en,
            "type": "full",
        })

    # Target query (if events exist)
    if bundle.target_query:
        target_vn = strip_question_words(bundle.target_query)
        target_en = translate_to_english(target_vn)
        if target_vn != visual_full_vn:
            queries.append({
                "query_vn": target_vn,
                "query_en": target_en,
                "type": "target",
            })

    # Context query
    if bundle.context_query:
        context_vn = strip_question_words(bundle.context_query)
        context_en = translate_to_english(context_vn)
        queries.append({
            "query_vn": context_vn,
            "query_en": context_en,
            "type": "context",
        })

    return queries
