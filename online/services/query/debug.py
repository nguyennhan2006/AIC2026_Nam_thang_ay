"""Query debug utilities for UI and logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.query.models import SearchQueryBundle


def bundle_to_debug_ui(bundle: SearchQueryBundle) -> dict:
    """Convert SearchQueryBundle to debug dict for UI display.

    Returns a structured dict suitable for StreamLog and QueryStudio display.
    """
    return bundle.to_debug_dict()


def bundle_to_stream_events(bundle: SearchQueryBundle) -> list[dict]:
    """Convert SearchQueryBundle to stream events for search_stream().

    Returns events that mirror the existing query_prepared event format
    but with additional per-engine queries.
    """
    return [
        {
            "type": "query_raw",
            "query": bundle.raw_query,
        },
        {
            "type": "query_normalized",
            "query": bundle.normalized_query,
        },
        {
            "type": "query_visual",
            "query": bundle.visual_query,
            "query_en": bundle.visual_query_en,
        },
        {
            "type": "query_caption",
            "query": bundle.caption_query,
        },
        {
            "type": "query_ocr",
            "query": bundle.ocr_query,
            "expected_units": bundle.expected_units,
            "exact_phrases": bundle.exact_phrases,
        },
        {
            "type": "query_asr",
            "query": bundle.asr_query,
        },
        {
            "type": "query_intent",
            "intent": bundle.intent.value,
            "answer_type": bundle.answer_type.value,
            "complexity": bundle.complexity_score,
        },
    ]


def log_bundle(logger, bundle: SearchQueryBundle, prefix: str = "") -> None:
    """Log bundle details for debugging.

    Usage:
        from online.services.query.debug import log_bundle
        log_bundle(logger, bundle)
    """
    logger.info(
        f"{prefix}QueryBundle: "
        f"task={bundle.debug_info.get('task')}, "
        f"intent={bundle.intent.value}, "
        f"answer_type={bundle.answer_type.value}"
    )
    logger.info(f"{prefix}  raw: {bundle.raw_query[:100]}...")
    logger.info(f"{prefix}  visual: {bundle.visual_query[:100]}...")
    logger.info(f"{prefix}  visual_en: {bundle.visual_query_en[:100]}...")
    logger.info(f"{prefix}  caption: {bundle.caption_query[:100]}...")
    logger.info(f"{prefix}  ocr: {bundle.ocr_query[:100]}...")
    logger.info(f"{prefix}  asr: {bundle.asr_query[:100]}...")
    logger.info(
        f"{prefix}  entities={bundle.visual_entities}, "
        f"actions={bundle.actions}, "
        f"attributes={bundle.attributes}"
    )
    logger.info(
        f"{prefix}  complexity={bundle.complexity_score}, "
        f"is_complex={bundle.is_complex}"
    )
