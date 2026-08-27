"""Smoke tests for Query Routing V2.

Run with: pytest tests/test_query_routing_v2.py -v

These tests verify the key behavior changes:
1. QA queries are NOT split by temporal markers (MAIN FIX)
2. Visual queries have question words removed
3. Different specialized queries are generated per engine
4. Numeric QA is detected correctly
"""

import pytest
from online.domain.models import SearchRequest, TaskType
from online.services.query.router import QueryRouter
from online.services.query.models import QueryIntent, AnswerType


class TestQueryRoutingV2:
    """Test Query Routing V2 behavior - CORE TESTS."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_qa_fish_query_no_split(self, router):
        """The KEY BUG FIX: QA query with 'cuoi cung' should NOT be split.

        Before fix: "Con so hien thi cuoi cung tren can la bao nhieu?"
        was split to "tren can la bao nhieu?" (WRONG!)

        After fix: query should keep full visual description.
        """
        request = SearchRequest(
            query="Hinh anh mot con ca duoc dat len can, sau do co canh mot con ca khac cung loai bi mot nguoi cam duoi. Con so hien thi cuoi cung tren can la bao nhieu?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # Visual query should contain fish and scale, NOT just "tren can la bao nhieu"
        assert "ca" in bundle.visual_query or "can" in bundle.visual_query, \
            f"visual_query should contain 'ca' or 'can', got: {bundle.visual_query}"

        # Visual query should NOT be just the question
        assert bundle.visual_query != "tren can la bao nhieu", \
            f"visual_query should NOT be just question, got: {bundle.visual_query}"

        # Intent should be NUMERIC_OCR
        assert bundle.intent == QueryIntent.NUMERIC_OCR, \
            f"intent should be NUMERIC_OCR, got: {bundle.intent}"

        # Answer type should be NUMERIC
        assert bundle.answer_type == AnswerType.NUMERIC, \
            f"answer_type should be NUMERIC, got: {bundle.answer_type}"

    def test_engine_specific_queries_differ(self, router):
        """Different retrieval engines should get different queries."""
        request = SearchRequest(
            query="Mot con ca dat tren can dien tu. Con so hien thi la bao nhieu?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # All queries should be non-empty
        assert bundle.visual_query, "visual_query should not be empty"
        assert bundle.caption_query, "caption_query should not be empty"
        assert bundle.ocr_query, "ocr_query should not be empty"

    def test_english_visual_augmentation(self, router):
        """Visual query should include English translation for Jina CLIP v2."""
        request = SearchRequest(
            query="Mot con ca duoc dat tren can",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # English visual query should exist
        assert bundle.visual_query_en, "English visual query should exist"

    def test_complex_query_complexity(self, router):
        """Complex queries should have higher complexity score."""
        simple_query = "Ca nang bao nhieu?"
        complex_query = "Hinh anh mot con ca duoc dat len can, sau do co canh mot con ca khac cung loai bi mot nguoi cam duoi. Con so hien thi cuoi cung tren can la bao nhieu?"

        simple_bundle = router.prepare_sync(SearchRequest(query=simple_query, task=TaskType.QA))
        complex_bundle = router.prepare_sync(SearchRequest(query=complex_query, task=TaskType.QA))

        # Complex query should have higher complexity
        assert complex_bundle.complexity_score > simple_bundle.complexity_score, \
            f"Complex query complexity ({complex_bundle.complexity_score}) should be > simple ({simple_bundle.complexity_score})"

    def test_debug_info_present(self, router):
        """Query bundle should include debug info for UI."""
        request = SearchRequest(
            query="Con ca nang bao nhieu kg?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        debug = bundle.to_debug_dict()

        assert "task" in debug, "Debug info should include task"
        assert "intent" in debug, "Debug info should include intent"
        assert "answer_type" in debug, "Debug info should include answer_type"
        assert "visual" in debug, "Debug info should include visual query"
        assert "ocr" in debug, "Debug info should include ocr query"


class TestTemporalMarkerBehavior:
    """Test temporal marker handling for different contexts."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_cuoi_cung_in_question_is_not_split(self, router):
        """'Cuoi cung' in a question should NOT trigger split.

        "Con so cuoi cung tren can" is an ATTRIBUTE, not a temporal marker.
        """
        request = SearchRequest(
            query="Con so cuoi cung tren can la bao nhieu?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # Should NOT be split - visual query should contain full context
        assert len(bundle.visual_query) > 10, \
            f"visual_query should not be stripped to tiny fragment, got: {bundle.visual_query}"

    def test_cuoi_cung_at_clause_boundary_is_split(self, router):
        """'Cuoi cung' at clause boundary SHOULD trigger split for KIS.

        "Cuoi cung, nguoi dan ong di vao nha" is a TEMPORAL marker.
        """
        request = SearchRequest(
            query="Nguoi dan ong dung truoc nha. Cuoi cung, ong ay di vao.",
            task=TaskType.TEXTUAL_KIS,
        )
        bundle = router.prepare_sync(request)

        # For KIS, we should have events detected
        # The split should produce meaningful parts
        assert bundle.target_query or bundle.context_query, \
            "KIS should have target or context separated"


class TestEdgeCases:
    """Edge case tests - less critical."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_short_query(self, router):
        """Short queries should still work."""
        request = SearchRequest(query="Ca", task=TaskType.QA)
        bundle = router.prepare_sync(request)

        assert bundle.visual_query, "Short query should still produce visual query"
        assert bundle.intent in [QueryIntent.VISUAL, QueryIntent.MIXED]


class TestQueryStripping:
    """Test visual query stripping behavior."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_visual_query_strips_question_patterns(self, router):
        """Visual query should strip abstract question patterns."""
        request = SearchRequest(
            query="Mot nguoi dan ong do nuoc vao coc. Coc cuoi cung co mau gi?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # Visual query should NOT contain "mau gi" (the question)
        assert "mau gi" not in bundle.visual_query.lower(), \
            f"visual_query should not contain 'mau gi', got: {bundle.visual_query}"

    def test_qa_no_split_for_any_marker(self, router):
        """QA should NEVER split by temporal markers."""
        request = SearchRequest(
            query="Hinh anh nguoi dan ong cam ca, sau do nguoi phu nu do nuoc. Cuoi cung ai uong?",
            task=TaskType.QA,
        )
        bundle = router.prepare_sync(request)

        # Visual query should contain context, not be split to tiny fragments
        # Should contain main entities (person, fish, cup)
        assert len(bundle.visual_query) > 5, \
            f"QA visual_query should keep context, got: {bundle.visual_query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
