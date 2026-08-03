"""PR-07: QA phải trả (video, frame, answer), không phải bằng chứng thô.

`EvidenceOnlyAnswerGenerator` (trước PR-07) nối caption/OCR/ASR thành chuỗi —
mọi item QA khi đó chấm 0 điểm vì thiếu answer. Test dưới đây khóa lại: có
answer thật, verify độc lập với tool sinh ra nó, và joint ranking không cho
answer rỗng lọt qua.
"""

from __future__ import annotations

import unittest

from online.domain.evidence import EvidencePack
from online.services.qa import (
    ANSWER_TOOLS,
    QaProcessor,
    QuestionParser,
    normalize_answer,
    verify_answer,
)
from online.domain.task_results import AnswerCandidate
from online.errors import DependencyUnavailableError


def pack(
    candidate_id: str,
    *,
    ocr: str | None = None,
    caption: str | None = None,
    asr: str | None = None,
    colors: list[str] | None = None,
    objects: list[str] | None = None,
    best_frame_idx: int | None = 100,
) -> EvidencePack:
    from online.domain.candidate import FrameEvidence

    frame = FrameEvidence(
        keyframe_id=f"{candidate_id}_F000100", video_id="L21_V001", scene_id=candidate_id,
        frame_idx=100, timestamp_sec=4.0, image_path="processed/keyframes/f.jpg",
        dominant_colors=colors or [], object_labels=objects or [],
    )
    return EvidencePack(
        candidate_id=candidate_id, video_id="L21_V001", scene_id=candidate_id,
        start_frame=0, end_frame_exclusive=200, start_sec=0.0, end_sec=8.0,
        keyframes=[frame], best_frame_idx=best_frame_idx,
        ocr_text=ocr, caption_text=caption, asr_window=asr,
    )


class ParserTests(unittest.TestCase):
    def test_count_question_is_routed_to_count(self) -> None:
        parsed = QuestionParser().parse("Có bao nhiêu xe máy tham gia vụ va chạm?")
        self.assertEqual(parsed.answer_type, "count")

    def test_color_question_is_routed_to_color(self) -> None:
        parsed = QuestionParser().parse("Chiếc áo của người đó màu gì?")
        self.assertEqual(parsed.answer_type, "color")

    def test_explicit_event_description_becomes_the_retrieval_context(self) -> None:
        parsed = QuestionParser().parse(
            "Thiệt hại là bao nhiêu tiền?", event_description="Bản tin về sạt lở bờ sông"
        )
        self.assertEqual(parsed.event_query, "Bản tin về sạt lở bờ sông")
        self.assertIn("sạt lở", parsed.retrieval_query)

    def test_single_sentence_question_uses_itself_as_both_parts(self) -> None:
        parsed = QuestionParser().parse("Người đó đang làm gì?")
        self.assertEqual(parsed.event_query, parsed.question_target)


class AnswerNormalizationTests(unittest.TestCase):
    def test_diacritics_and_case_are_stripped(self) -> None:
        self.assertEqual(normalize_answer("Hơn 14,5 Tỷ Đồng"), "hon 14 5 ty dong")

    def test_normalized_forms_of_synonyms_match(self) -> None:
        self.assertEqual(normalize_answer("Đỏ"), normalize_answer("do"))


class ToolTests(unittest.TestCase):
    def test_count_tool_extracts_number_from_ocr(self) -> None:
        candidates = ANSWER_TOOLS["count"](
            QuestionParser().parse("Có bao nhiêu xe máy?"),
            pack("p1", ocr="Va chạm liên hoàn 5 xe máy trong đêm"),
        )
        self.assertIn("5", [item.canonical for item in candidates])

    def test_color_tool_uses_dominant_colors(self) -> None:
        candidates = ANSWER_TOOLS["color"](
            QuestionParser().parse("Áo màu gì?"), pack("p1", colors=["red", "red", "blue"]),
        )
        self.assertEqual(candidates[0].canonical, "red")

    def test_ocr_tool_prefers_the_longest_text(self) -> None:
        from online.domain.candidate import FrameEvidence

        frame = FrameEvidence(
            keyframe_id="p1_F000100", video_id="L21_V001", scene_id="p1", frame_idx=100,
            timestamp_sec=4.0, image_path="f.jpg",
            ocr_texts=["OK", "Cảnh báo sạt lở nguy hiểm khu dân cư"],
        )
        evidence = pack("p1")
        evidence = evidence.model_copy(update={"keyframes": [frame]})
        candidates = ANSWER_TOOLS["ocr_text"](QuestionParser().parse("Biển ghi gì?"), evidence)
        self.assertEqual(candidates[0].canonical, "Cảnh báo sạt lở nguy hiểm khu dân cư")

    def test_yes_no_tool_answers_no_when_evidence_lacks_the_terms(self) -> None:
        parsed = QuestionParser().parse("Có phải xe tải gây ra vụ va chạm không?")
        candidates = ANSWER_TOOLS["yes_no"](parsed, pack("p1", caption="hai xe máy va vào nhau"))
        self.assertEqual(candidates[0].canonical, "không")

    def test_entity_tool_falls_back_to_caption_when_no_objects(self) -> None:
        candidates = ANSWER_TOOLS["entity"](
            QuestionParser().parse("Đó là gì?"), pack("p1", caption="một chiếc thuyền đánh cá"),
        )
        self.assertTrue(candidates)
        self.assertIn("thuyền", candidates[0].canonical)


class VerifierTests(unittest.TestCase):
    def test_answer_present_in_evidence_is_supported(self) -> None:
        answer = AnswerCandidate(canonical="5", surface="5", confidence=0.7, answer_type="count")
        status = verify_answer(answer, pack("p1", ocr="5 xe máy va chạm"))
        self.assertEqual(status, "SUPPORTED")

    def test_contradicting_number_is_contradicted(self) -> None:
        answer = AnswerCandidate(canonical="9", surface="9", confidence=0.7, answer_type="count")
        status = verify_answer(answer, pack("p1", ocr="5 xe máy va chạm"))
        self.assertEqual(status, "CONTRADICTED")

    def test_no_evidence_at_all_is_insufficient(self) -> None:
        answer = AnswerCandidate(canonical="5", surface="5", confidence=0.7, answer_type="count")
        status = verify_answer(answer, pack("p1"))
        self.assertEqual(status, "INSUFFICIENT")

    def test_yes_no_without_exact_string_is_partial_not_insufficient(self) -> None:
        answer = AnswerCandidate(canonical="có", surface="có", confidence=0.6, answer_type="yes_no")
        status = verify_answer(answer, pack("p1", caption="một sự kiện nào đó"))
        self.assertEqual(status, "PARTIAL")


class QaProcessorTests(unittest.TestCase):
    def test_no_result_has_an_empty_answer(self) -> None:
        packs = [
            pack("p1", ocr="5 xe máy va chạm liên hoàn"),
            pack("p2", caption="một cảnh không liên quan"),
        ]
        results = QaProcessor().answer(
            "Có bao nhiêu xe máy tham gia vụ va chạm?", packs,
            frame_scores={"p1": 0.9, "p2": 0.3},
        )
        self.assertTrue(results)
        self.assertTrue(all(item.answer.strip() for item in results))

    def test_only_evidence_matching_the_question_target_yields_a_candidate(self) -> None:
        # p_unrelated không nhắc tới "xe máy" -> count tool không sinh candidate
        # từ nó, dù có chứa một con số. Việc lọc theo target-word tránh việc
        # đếm nhầm một con số không liên quan tới câu hỏi.
        packs = [
            pack("p_unrelated", ocr="9 người đứng xem pháo hoa"),
            pack("p_right", ocr="5 xe máy va chạm liên hoàn"),
        ]
        results = QaProcessor().answer(
            "Có bao nhiêu xe máy va chạm?", packs,
            frame_scores={"p_unrelated": 0.9, "p_right": 0.5},
        )
        self.assertTrue(results)
        self.assertTrue(all(item.canonical_answer == "5" for item in results))

    def test_contradicted_status_is_reachable_when_a_wrong_number_is_proposed(self) -> None:
        from online.services.qa import verify_answer
        from online.domain.task_results import AnswerCandidate

        wrong = AnswerCandidate(canonical="9", surface="9", confidence=0.7, answer_type="count")
        status = verify_answer(wrong, pack("p_right", ocr="5 xe máy va chạm liên hoàn"))
        self.assertEqual(status, "CONTRADICTED")

    def test_duplicate_video_frame_answer_triples_are_collapsed(self) -> None:
        packs = [pack("p1", ocr="5 xe máy"), pack("p1", ocr="5 xe may")]
        results = QaProcessor().answer("bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9})
        keys = {(item.video_id, item.frame_idx, item.canonical_answer) for item in results}
        self.assertEqual(len(keys), len(results))

    def test_empty_pack_list_returns_no_results(self) -> None:
        self.assertEqual(QaProcessor().answer("câu hỏi bất kỳ", []), [])


class _FakeLlmAnswerer:
    def __init__(self, candidate: AnswerCandidate | None = None, error: Exception | None = None) -> None:
        self.candidate = candidate
        self.error = error
        self.calls = 0

    async def answer(self, question: str, pack: EvidencePack) -> AnswerCandidate | None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.candidate


class QaProcessorAsyncLlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_llm_answerer_behaves_like_sync_answer(self) -> None:
        packs = [pack("p1", ocr="5 xe máy va chạm liên hoàn")]
        results, warnings = await QaProcessor().answer_async(
            "Có bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9},
        )
        self.assertTrue(results)
        self.assertEqual(warnings, [])

    async def test_llm_candidate_replaces_the_rule_based_answer(self) -> None:
        llm = _FakeLlmAnswerer(
            AnswerCandidate(canonical="5", surface="5 chiếc", confidence=0.9, answer_type="count", source="fpt_llm")
        )
        processor = QaProcessor(llm_answerer=llm, llm_top_n=5)
        packs = [pack("p1", ocr="5 xe máy va chạm liên hoàn")]
        results, warnings = await processor.answer_async(
            "Có bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9},
        )
        self.assertEqual(warnings, [])
        self.assertTrue(any(item.answer == "5 chiếc" for item in results))

    async def test_contradicted_llm_answer_keeps_rule_based_answer(self) -> None:
        llm = _FakeLlmAnswerer(
            AnswerCandidate(canonical="9", surface="9", confidence=0.9, answer_type="count", source="fpt_llm")
        )
        processor = QaProcessor(llm_answerer=llm)
        packs = [pack("p1", ocr="5 xe máy va chạm liên hoàn")]
        results, warnings = await processor.answer_async(
            "Có bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9},
        )
        self.assertTrue(all(item.canonical_answer != "9" for item in results))

    async def test_llm_error_falls_back_and_records_a_warning(self) -> None:
        llm = _FakeLlmAnswerer(error=DependencyUnavailableError("boom"))
        processor = QaProcessor(llm_answerer=llm)
        packs = [pack("p1", ocr="5 xe máy va chạm liên hoàn")]
        results, warnings = await processor.answer_async(
            "Có bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9},
        )
        self.assertTrue(results)
        self.assertEqual(len(warnings), 1)

    async def test_llm_is_capped_to_llm_top_n_distinct_packs(self) -> None:
        llm = _FakeLlmAnswerer(
            AnswerCandidate(canonical="5", surface="5", confidence=0.9, answer_type="count", source="fpt_llm")
        )
        processor = QaProcessor(llm_answerer=llm, llm_top_n=1)
        packs = [
            pack("p1", ocr="5 xe máy va chạm liên hoàn"),
            pack("p2", ocr="5 xe máy tông nhau"),
        ]
        await processor.answer_async("Có bao nhiêu xe máy?", packs, frame_scores={"p1": 0.9, "p2": 0.8})
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
