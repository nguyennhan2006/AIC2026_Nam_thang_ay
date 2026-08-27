"""Query Router V2 — Modality-Aware Query Preparation.

Main entry point for Query Routing V2.
Takes a raw query and returns a SearchQueryBundle with specialized queries
for each retrieval engine.

Usage:
    router = QueryRouter()
    bundle = await router.prepare(request)

    # Dense visual retrieval
    results = await dense.search(bundle.visual_query)

    # Caption BM25
    results = await caption.search(bundle.caption_query)

    # OCR
    results = await ocr.search(bundle.ocr_query)

    # ASR
    results = await asr.search(bundle.asr_query)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from online.domain.models import SearchRequest, TaskType
from online.services.query.models import (
    AnswerType,
    QueryIntent,
    SearchQueryBundle,
)
from online.services.query.normalize import (
    extract_numbers,
    extract_quotes,
    normalize_query,
    split_temporal_weak,
)
from online.services.query.intent import (
    classify_answer_type,
    classify_intent,
    extract_expected_units,
)
from online.services.query.builders import (
    build_visual_query,
    build_caption_query,
    build_ocr_query,
    build_asr_query,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QueryRouter:
    """Main router for Query Routing V2.

    Takes raw query and produces specialized queries per retrieval engine.
    """

    def __init__(self) -> None:
        pass

    async def prepare(self, request: SearchRequest) -> SearchQueryBundle:
        """Async version of prepare - same as prepare_sync."""
        return self.prepare_sync(request)

    def prepare_sync(self, request: SearchRequest) -> SearchQueryBundle:
        """Prepare query bundle from search request.

        Args:
            request: The search request with raw query.

        Returns:
            SearchQueryBundle with specialized queries for each retriever.
        """
        # Step 1: Basic normalization
        raw_query = request.query.strip()
        normalized = normalize_query(raw_query)

        # Step 2: Extract structure
        exact_phrases = extract_quotes(normalized)
        numbers = extract_numbers(normalized)

        # Step 3: Detect events (for TRAKE/multi-event queries)
        events = self._detect_events(normalized, request.task)

        # Step 4: Determine target vs context
        target_query, context_query = self._split_target_context(
            normalized, events, request.task
        )

        # Step 5: Classify intent and answer type
        # Build initial bundle for classification
        initial_bundle = SearchQueryBundle(
            raw_query=raw_query,
            normalized_query=normalized,
            exact_phrases=exact_phrases,
            events=events,
            target_query=target_query,
            context_query=context_query,
        )

        intent = classify_intent(initial_bundle)
        answer_type = classify_answer_type(initial_bundle)
        expected_units = extract_expected_units(normalized)

        # Step 6: Build specialized queries
        visual_query, visual_query_en = build_visual_query(initial_bundle)
        caption_query = build_caption_query(initial_bundle)
        ocr_query = build_ocr_query(initial_bundle)
        asr_query = build_asr_query(initial_bundle)

        # Step 7: Extract entities for debugging
        entities = self._extract_entities(normalized)
        actions = self._extract_actions(normalized)
        attributes = self._extract_attributes(normalized)

        # Step 8: Estimate complexity
        complexity = self._estimate_complexity(
            normalized, events, exact_phrases, answer_type
        )

        # Build final bundle
        bundle = SearchQueryBundle(
            raw_query=raw_query,
            normalized_query=normalized,
            visual_query=visual_query,
            visual_query_en=visual_query_en,
            caption_query=caption_query,
            ocr_query=ocr_query,
            asr_query=asr_query,
            context_query=context_query,
            target_query=target_query,
            exact_phrases=exact_phrases,
            events=events,
            intent=intent,
            answer_type=answer_type,
            expected_units=expected_units,
            visual_entities=entities,
            actions=actions,
            attributes=attributes,
            complexity_score=complexity,
            debug_info={
                "task": (request.task or TaskType.TEXTUAL_KIS).value,
                "numbers": numbers,
            },
        )

        # Log for debugging
        logger.debug(
            f"QueryRouter: task={bundle.debug_info['task']}, "
            f"intent={bundle.intent.value}, "
            f"answer_type={bundle.answer_type.value}, "
            f"complexity={bundle.complexity_score}, "
            f"visual='{bundle.visual_query[:50]}...'" if len(bundle.visual_query) > 50 else f"visual='{bundle.visual_query}'"
        )

        return bundle

    def _detect_events(self, query: str, task: TaskType | None) -> list[str]:
        """Detect event boundaries in query."""
        # Only for TRAKE
        if task != TaskType.TRAKE:
            return []

        # Split by strong temporal markers
        from online.services.query.normalize import STRONG_TEMPORAL

        pieces = [query]
        for marker in STRONG_TEMPORAL:
            new_pieces = []
            for piece in pieces:
                parts = piece.split(marker)
                new_pieces.extend([p.strip() for p in parts if p.strip()])
            pieces = new_pieces

        # Filter empty
        return [p for p in pieces if p]

    def _split_target_context(
        self, query: str, events: list[str], task: TaskType | None
    ) -> tuple[str, str]:
        """Split query into target (moment to find) and context (background).

        For KIS: target is usually the last event
        For QA: keep full query as-is (no splitting)

        Args:
            query: The normalized query.
            events: Detected events.
            task: The task type.

        Returns:
            Tuple of (target_query, context_query).
        """
        # QA keeps full query - no splitting for QA
        if task == TaskType.QA:
            return query, ""

        # TRAKE uses events directly
        if task == TaskType.TRAKE and len(events) >= 2:
            return events[-1] if events else query, ""

        # AVS - similar to KIS, use target only
        # KIS - use weak temporal split if applicable
        target, context = split_temporal_weak(query)

        # If context is too similar to target, don't use it
        if context and len(context) < len(target) * 0.3:
            return target, ""

        return target, context

    def _extract_entities(self, query: str) -> list[str]:
        """Extract key entities from query for debugging."""
        # Simple keyword extraction (ASCII Vietnamese)
        entities = []

        entity_keywords = [
            "ca", "xe", "nguoi", "dan ong", "phu nu", "tre", "hoc sinh",
            "giao vien", "can", "coc", "nuoc", "nha", "truong", "bang",
        ]

        query_lower = query.lower()
        for entity in entity_keywords:
            if entity in query_lower:
                entities.append(entity)

        return entities

    def _extract_actions(self, query: str) -> list[str]:
        """Extract action verbs from query."""
        actions = []

        action_keywords = [
            "dat", "cam", "nam", "do", "rot", "uong", "an", "noi",
            "hat", "nhay", "chay", "di", "ngoi", "dung", "vay tay",
            "cao", "cuoc", "lai", "nem", "quang",
        ]

        query_lower = query.lower()
        for action in action_keywords:
            if action in query_lower:
                actions.append(action)

        return actions

    def _extract_attributes(self, query: str) -> list[str]:
        """Extract visual attributes from query."""
        attributes = []

        # Colors (ASCII Vietnamese)
        colors = ["do", "xanh", "vang", "trang", "den", "cam", "tim", "hong"]
        query_lower = query.lower()
        for color in colors:
            if color in query_lower:
                attributes.append(color)

        # Spatial
        spatial = ["tren", "duoi", "trai", "phai", "giua"]
        for sp in spatial:
            if sp in query_lower:
                attributes.append(sp)

        return attributes

    def _estimate_complexity(
        self, query: str, events: list[str], exact_phrases: list[str], answer_type: AnswerType
    ) -> int:
        """Estimate query complexity for deciding processing strategy."""
        complexity = 0

        # Token count
        token_count = len(query.split())
        if token_count > 25:
            complexity += 1
        if token_count > 50:
            complexity += 1

        # Multiple events
        if len(events) >= 2:
            complexity += 1
        if len(events) >= 3:
            complexity += 1

        # Has quotes (OCR evidence)
        if exact_phrases:
            complexity += 1

        # Numeric question
        if answer_type == AnswerType.NUMERIC:
            complexity += 1

        return min(complexity, 5)


# Convenience function
async def prepare_query(request: SearchRequest) -> SearchQueryBundle:
    """Prepare query bundle from search request.

    Shortcut for QueryRouter().prepare(request).
    """
    router = QueryRouter()
    return await router.prepare(request)
