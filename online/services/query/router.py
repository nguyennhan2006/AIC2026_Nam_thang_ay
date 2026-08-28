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

import asyncio
from dataclasses import replace
import logging
import re
from typing import TYPE_CHECKING

from online.domain.models import SearchRequest, TaskType
from online.services.query.models import (
    AnswerType,
    QueryIntent,
    SearchQueryBundle,
)
from online.services.query.normalize import (
    STRONG_TEMPORAL,
    extract_numbers,
    extract_quotes,
    normalize_query,
    split_temporal_weak,
    strip_diacritics,
)

# Gold TRAKE dùng format liệt kê "(1) ...; (2) ...", giống query_planner.py.
NUMBERED_STEP_RE = re.compile(r"\(\d+\)\s*")
# Đề sơ tuyển thật dùng format "E1 ... E2 ... E3 ..." thay vì ngoặc — cùng lý
# do và cùng cách xử lý như `online/services/query_planner.py::LETTERED_STEP_RE`.
LETTERED_STEP_RE = re.compile(r"(?<!\w)E\d+\s+")
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
from online.services.query.builders.visual import (
    extract_actions,
    extract_attributes,
    extract_visual_entities,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QueryRouter:
    """Main router for Query Routing V2.

    Hai tầng, cố ý tách rời:

    Tier 1 — rule (luôn chạy, <1ms, tất định)
        Cắt phần hỏi trừu tượng, tách event, phân loại intent. Đây là NỀN:
        mọi truy vấn đều có bundle dùng được kể cả khi không có mạng.

    Tier 2 — LLM (tuỳ chọn, `refiner`)
        VIẾT LẠI từng trường cho đúng thế mạnh của từng engine — việc rule
        không làm được. Hỏng thì bundle rule đi tiếp nguyên vẹn.

    `refiner=None` giữ nguyên hành vi thuần rule.
    """

    def __init__(self, refiner=None) -> None:
        # refiner: đối tượng có `.refine(bundle, task=...) -> SearchQueryBundle`
        # (xem online/adapters/fpt_query_bundle.py::FptQueryBundlePreparer).
        self.refiner = refiner

    async def prepare(self, request: SearchRequest) -> SearchQueryBundle:
        """Bundle cho request; chạy phần LLM ngoài event loop nếu có cấu hình."""

        bundle = self.prepare_sync(request)
        if self.refiner is None:
            return bundle

        task = (request.task or TaskType.TEXTUAL_KIS).value
        try:
            # `refine` gọi HTTP đồng bộ; chạy thẳng trên event loop sẽ khoá cả
            # server trong lúc chờ LLM — đúng lỗi mà bm25.py đã phải sửa bằng
            # to_thread. Lỗi ngoài dự kiến cũng chỉ được phép mất phần cải thiện.
            return await asyncio.to_thread(self.refiner.refine, bundle, task=task)
        except Exception as exc:  # noqa: BLE001 - không truy vấn nào được chết vì refiner
            logger.warning("LLM refine bundle hỏng, dùng bundle rule: %s", exc)
            return bundle

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

        # Step 6: Build specialized queries.
        #
        # Builder ĐỌC `intent`/`answer_type`/`expected_units` (vd `build_ocr_query`
        # kiểm tra `is_numeric_qa` để bơm đơn vị cân nặng), nên chúng phải nằm
        # sẵn trong bundle truyền vào. Truyền `initial_bundle` — vốn chưa phân
        # loại — làm nhánh đó im lặng không chạy và ocr_query ra rỗng.
        classified = replace(
            initial_bundle,
            intent=intent,
            answer_type=answer_type,
            expected_units=expected_units,
        )

        # visual chạy trước: caption expansion dùng lại `visual_query_en`.
        visual_query, visual_query_en = build_visual_query(classified)
        classified = replace(
            classified, visual_query=visual_query, visual_query_en=visual_query_en
        )

        caption_query = build_caption_query(classified)
        ocr_query = build_ocr_query(classified)
        asr_query = build_asr_query(classified)

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
        """Tách query thành các event theo marker thời gian.

        Chạy cho MỌI task, không riêng TRAKE: `events` còn được dùng để tính
        complexity và để chọn intent TEMPORAL. Việc có tách `normalized_query`
        hay không là quyết định RIÊNG của `_split_target_context`.

        Ưu tiên format đánh số "(1) ... (2) ..." — đúng format gold TRAKE —
        rồi mới tới marker "sau đó"/"tiếp theo".
        """

        numbered = [part.strip(" ,.;:") for part in NUMBERED_STEP_RE.split(query)[1:]]
        numbered = [part for part in numbered if part]
        if len(numbered) >= 2:
            return numbered

        lettered = [part.strip(" ,.;:") for part in LETTERED_STEP_RE.split(query)[1:]]
        lettered = [part for part in lettered if part]
        if len(lettered) >= 2:
            return lettered

        # Tách trên bản KHÔNG DẤU rồi ánh xạ offset về chuỗi gốc: hai chuỗi
        # cùng độ dài nên offset khớp 1-1.
        ascii_query = strip_diacritics(query)
        pattern = "|".join(re.escape(marker) for marker in sorted(STRONG_TEMPORAL, key=len, reverse=True))
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", ascii_query):
            piece = query[cursor : match.start()].strip(" ,.;:")
            if piece:
                pieces.append(piece)
            cursor = match.end()
        tail = query[cursor:].strip(" ,.;:")
        if tail:
            pieces.append(tail)

        return pieces if len(pieces) >= 2 else []

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

    # Trích entity/action/attribute dùng CHUNG một nguồn với visual builder
    # (online/services/query/builders/visual.py) — trước đây router giữ bảng
    # keyword riêng, nên sửa một chỗ không sửa chỗ kia.
    def _extract_entities(self, query: str) -> list[str]:
        return extract_visual_entities(query)

    def _extract_actions(self, query: str) -> list[str]:
        return extract_actions(query)

    def _extract_attributes(self, query: str) -> list[str]:
        return extract_attributes(query)

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
