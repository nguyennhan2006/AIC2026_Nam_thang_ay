"""Lexical query builders for BM25 caption/OCR/ASR.

Principles:
- Caption: keep nouns/verbs, have synonym expansion, can keep context
- OCR: only expected text/keywords, no visual description
- ASR: semantic question + speech keywords, no visual description
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

from online.services.query.normalize import SPACE_RE


# Synonyms for caption expansion (ASCII Vietnamese)
SYNONYMS = {
    # Fish / weighing
    "ca": ["ca", "ca con", "con ca"],
    "can": ["can", "can dien tu", "can so", "weighing scale", "scale", "weight"],
    "dat": ["dat", "de", "bo", "cho", "placed", "put", "placed on"],
    "hien thi": ["hien thi", "show", "display", "hien"],
    "so": ["so", "con so", "number", "chu so"],

    # People / actions
    "nguoi": ["nguoi", "nguoi dan ong", "nguoi phu nu", "man", "woman", "person"],
    "cam": ["cam", "nam", "nam giu", "om", "holding", "holding by"],
    "duoi": ["duoi", "tail"],
    "do": ["do", "rot", "pouring", "pour", "do nuoc"],

    # Scale / display
    "man hinh": ["man hinh", "screen", "display"],
    "kg": ["kg", "kilogram"],
    "g": ["g", "gram"],

    # Common
    "sau do": ["sau do", "roi", "tiep theo", "then", "after that"],
    "cuoi cung": ["cuoi cung", "finally", "last"],
    "dau tien": ["dau tien", "first", "dau"],
}


# OCR-specific keywords - only text-related (ASCII Vietnamese)
OCR_KEYWORDS = {
    "kg", "g", "gram", "kilogram", "mg",
    "ml", "l", "liter",
    "diem", "diem so", "score",
    "phut", "giay", "gio", "thoi gian",
    "km", "m", "cm", "mm", "khoang cach",
    "do", "doC", "nhiet do",
    "gia", "price", "vnd", "dong",
    "so dien thoai", "phone", "tel",
    "ten", "name",
}


# ASR-specific keywords - speech-related (ASCII Vietnamese)
ASR_KEYWORDS = {
    "noi", "phat bieu", "tra loi", "hoi", "dap",
    "noi gi", "noi dieu gi", "cau noi", "loi noi",
    "tieng", "am thanh", "giọng", "hoi thoai", "doi thoai",
    "says", "speaks", "said", "voice",
}


def build_caption_query(bundle: SearchQueryBundle) -> str:
    """Build query for BM25 caption retrieval.

    Caption retrieval uses lexical matching so query should:
    - Keep main nouns and verbs
    - Have light synonym expansion
    - Can keep short context
    - No English needed
    """
    # Start from normalized query
    query = bundle.normalized_query

    # Step 1: Expand with synonyms
    expanded = expand_with_synonyms(query)

    # Step 2: Add English terms if helpful
    if bundle.visual_query_en:
        # Extract key terms from English visual query
        en_terms = extract_key_terms(bundle.visual_query_en)
        if en_terms:
            expanded += " " + " ".join(en_terms)

    # Step 3: Clean up
    result = SPACE_RE.sub(" ", expanded).strip()

    return result


def build_ocr_query(bundle: SearchQueryBundle) -> str:
    """Build query for OCR retrieval.

    OCR only needs to find text appearing on screen.
    Query should:
    - Focus on expected text/keywords
    - No visual description
    - Prioritize units and numbers
    """
    parts = []

    # Step 1: Expected units
    if bundle.expected_units:
        parts.extend(bundle.expected_units)

    # Step 2: Extract OCR keywords from query
    text_lower = bundle.normalized_query.lower()
    for keyword in OCR_KEYWORDS:
        if keyword in text_lower:
            parts.append(keyword)

    # Step 3: Add exact phrases (quoted text)
    if bundle.exact_phrases:
        parts.extend(bundle.exact_phrases)

    # Step 4: For numeric QA, also add number patterns
    if bundle.is_numeric_qa:
        # Look for number references
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', bundle.normalized_query)
        parts.extend(numbers)

    # Step 5: Build result
    if parts:
        return " ".join(parts)

    # Fallback: use normalized query but stripped of visual terms
    fallback = strip_visual_terms(bundle.normalized_query)
    return fallback if fallback else bundle.normalized_query


def build_asr_query(bundle: SearchQueryBundle) -> str:
    """Build query for ASR retrieval.

    ASR finds speech content, so query should:
    - Keep semantic question
    - Have speech-related keywords
    - No visual description
    """
    parts = []

    # Step 1: Semantic question (stripped of visual description)
    semantic = extract_semantic_question(bundle.normalized_query)
    if semantic:
        parts.append(semantic)

    # Step 2: ASR keywords
    text_lower = bundle.normalized_query.lower()
    for keyword in ASR_KEYWORDS:
        if keyword in text_lower:
            parts.append(keyword)

    # Step 3: Add expected answer type context
    if bundle.answer_type.name == "NUMERIC":
        parts.extend(["bao nhieu", "so", "nang", "diem"])

    # Build result
    result = " ".join(parts)
    return result if result else bundle.normalized_query


def expand_with_synonyms(text: str) -> str:
    """Expand text with synonyms for lexical retrieval."""
    result = text.lower()

    # Sort by length (longest first) to avoid partial replacements
    sorted_syns = sorted(SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)

    for term, synonyms in sorted_syns:
        if term in result:
            # Add all synonyms except the original
            additional = [s for s in synonyms if s != term]
            if additional:
                result += " " + " ".join(additional[:2])  # Max 2 synonyms

    return result


def extract_key_terms(en_text: str) -> list[str]:
    """Extract key English terms for caption expansion."""
    # Simple extraction - just split and filter
    words = en_text.split()
    # Filter out common stop words
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "on", "in", "at", "by", "with", "of", "to", "for", "and", "or", "but"}
    return [w for w in words if w.lower() not in stop_words and len(w) > 2]


def strip_visual_terms(text: str) -> str:
    """Remove visual description terms, keep only text-related terms."""
    visual_terms = {
        "nguoi", "dan ong", "phu nu", "tre", "con", "cai",
        "mau", "hinh", "anh", "nhin", "thay",
        "tren", "duoi", "trai", "phai", "giua",
        "dat", "cam", "nam", "do", "rot",
        "di", "chay", "nhay", "ngoi", "dung",
    }

    words = text.split()
    filtered = [w for w in words if w not in visual_terms]
    return " ".join(filtered)


def extract_semantic_question(text: str) -> str:
    """Extract the semantic question part from query."""
    # Remove visual description, keep question
    question_patterns = [
        r'([^?]*?)(?:la gi|co gi|o dau|ai|bao nhieu|may|khi nao|nhu the nao|tai sao)',
        r'.*(noi gi|tra loi|phat bieu)',
    ]

    for pattern in question_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return text
