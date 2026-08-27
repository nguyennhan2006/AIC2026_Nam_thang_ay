"""Query normalization - basic preprocessing without model calls."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle

# Patterns for basic extraction
SPACE_RE = re.compile(r"\s+")
QUOTED_RE = re.compile(r'"([^"]+)"')
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Question words - those that should NOT be in visual query
QUESTION_WORDS = {
    "bao nhieu", "bao nhieu?", "la bao nhieu", "may", "bang bao nhieu",
    "gi", "la gi", "co gi", "o dau", "la o dau", "ai", "la ai", "cua ai",
    "khi nao", "la khi nao", "nhu the nao", "bang cach nao",
    "sao", "tai sao", "vi sao", "nhu the nao",
    "what", "how", "when", "where", "who", "why", "which",
}

# Abstract question patterns - don't contain visual information
ABSTRACT_PATTERNS = [
    r"\bba?o\s+nhieu\b",
    r"\bla\s+gi\b",
    r"\bco\s+gi\b",
    r"\bo\s+dau\b",
    r"\bla\s+ai\b",
    r"\bnhu\s+the\s+nao\b",
    r"\bco\s+phai\s+khong\b",
    r"\bwhat\s+is\b",
    r"\bhow\s+many\b",
    r"\bhow\s+much\b",
    r"\bwho\s+(?:is|was|are|were)\b",
    r"\bwhy\b",
]

# Strong temporal markers - CERTAINLY transitions
STRONG_TEMPORAL = {
    "sau do", "tiep theo", "ke tiep", "roi", "roi thi",
    "then", "next", "after that",
}

# Weak temporal markers - may be attribute of noun, not marker
WEAK_TEMPORAL = {
    "cuoi cung", "finally",
}


def normalize_query(text: str) -> str:
    """Normalize query: NFC, lowercase, strip."""
    normalized = unicodedata.normalize("NFC", text).strip()
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized


def extract_quotes(text: str) -> list[str]:
    """Extract quoted phrases for OCR exact matching."""
    return [item.strip() for item in QUOTED_RE.findall(text) if item.strip()]


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return NUMBER_RE.findall(text)


def strip_question_words(text: str) -> str:
    """Remove abstract question words from visual query.

    "Con so hien thi cuoi cung tren can la bao nhieu?"
    -> "Con so hien thi cuoi cung tren can"
    """
    result = text
    for pattern in ABSTRACT_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    # Clean up extra spaces
    result = SPACE_RE.sub(" ", result).strip()
    return result


def is_temporal_marker(text: str, position: int = -1) -> bool:
    """Check if text contains temporal marker.

    Args:
        text: The text to check.
        position: Position of marker in original sentence (-1 = at end).

    Returns:
        True if strong temporal marker, False otherwise.
    """
    lowered = text.lower().strip()
    # Strong markers always temporal
    if lowered in STRONG_TEMPORAL:
        return True
    # Weak markers - check context
    if lowered in WEAK_TEMPORAL:
        # "cuoi cung" is temporal when at clause start
        # "cuoi cung, nguoi dan ong..." = temporal
        # "Con so cuoi cung tren can" = attribute
        return position <= 5  # near start of sentence
    return False


def split_temporal_weak(text: str) -> tuple[str, str]:
    """Split query by weak temporal markers (cuoi cung, finally).

    Unlike strong markers, these should only split when clearly at clause start.

    "Nguoi A do nuoc, sau do B rot nuoc. Cuoi cung, C uong."
    -> ("Nguoi A do nuoc, sau do B rot nuoc", "C uong")

    "Con so cuoi cung tren can la bao nhieu?"
    -> ("Con so cuoi cung tren can la bao nhieu?", "")  # NO split

    Returns:
        Tuple of (target_query, context_query). Empty context if no split.
    """
    # Find "cuoi cung" / "finally" positions
    text_lower = text.lower()

    # Check if weak temporal is at clause boundary (preceded by comma or period)
    # Pattern: ... marker , ... or ... marker .
    weak_pattern = r"(?<=[,;:])\s*(cuoi cung|finally)\s*,?\s*"
    matches = list(re.finditer(weak_pattern, text_lower))

    if not matches:
        return text, ""

    # Split at first weak temporal that's at clause boundary
    first_match = matches[0]
    before = text[:first_match.start()].strip()
    after = text[first_match.end():].strip()

    if after:
        return before, after
    return text, ""


def token_count(text: str) -> int:
    """Count tokens (rough word count)."""
    return len(text.split())


def estimate_complexity(bundle: SearchQueryBundle) -> int:
    """Estimate query complexity for deciding processing strategy.

    Returns:
        0-1: Simple query, rule-only processing
        2-3: Medium complexity, consider enhanced processing
        4+: Complex, definitely use enhanced processing
    """
    complexity = 0

    # Token count
    if token_count(bundle.normalized_query) > 25:
        complexity += 1
    if token_count(bundle.normalized_query) > 50:
        complexity += 1

    # Multiple events
    if len(bundle.events) >= 2:
        complexity += 1
    if len(bundle.events) >= 3:
        complexity += 1

    # Has quotes (OCR evidence)
    if bundle.exact_phrases:
        complexity += 1

    # Numeric question
    if bundle.answer_type.name == "NUMERIC":
        complexity += 1

    return min(complexity, 5)  # Cap at 5
