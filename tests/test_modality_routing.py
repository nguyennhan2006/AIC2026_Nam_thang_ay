"""ROUTE-01: modality không có cue phải về đúng 0 và branch KHÔNG được chạy.

Trước đây OCR/ASR có sàn 0.35/0.25 nên luôn góp candidate. Hệ quả tái hiện
được: truy vấn thuần thị giác "cột nước phun lên từ lòng đất" nhận một bản
tin cháy rừng và một cảnh lở đất vào top-5 qua OCR/ASR khớp vài token phổ
biến ("nước", "đất").

Khác biệt quan trọng: weight 0 phải nghĩa là *không chạy branch*, không phải
*chạy rồi nhân 0* — chạy rồi nhân 0 vẫn tốn latency và vẫn đổi union pool.
"""

from __future__ import annotations

import unittest

from online.domain.models import Modality, SearchRequest, TaskType
from online.services.query_planner import (
    RuleBasedQueryPlanner,
    compute_modality_weights,
    has_speech_cue,
    has_text_cue,
)


def _plan(query: str, *, allow_zero: bool = True):
    """Mặc định planner nay TẮT zero-gating (xem ROUTE-01 keep/drop), nên test
    của cơ chế này phải bật cờ tường minh."""

    import asyncio

    planner = RuleBasedQueryPlanner(allow_zero_modality=allow_zero)
    return asyncio.run(planner.plan(SearchRequest(query=query, task=TaskType.TEXTUAL_KIS)))


class CueDetectionTests(unittest.TestCase):
    def test_visual_query_has_no_cue(self) -> None:
        self.assertFalse(has_text_cue("cột nước phun lên từ lòng đất", []))
        self.assertFalse(has_speech_cue("cột nước phun lên từ lòng đất"))

    def test_text_cues_are_detected(self) -> None:
        for query in ["biển báo ghi gì", "dòng chữ trên màn hình", "tấm bảng màu vàng",
                      "phụ đề nói gì", "logo của đài"]:
            self.assertTrue(has_text_cue(query, []), query)

    def test_quoted_phrase_is_a_text_cue_even_without_keyword(self) -> None:
        self.assertTrue(has_text_cue("đoàn người đi qua", ["xin đừng quên nhau"]))

    def test_speech_cues_are_detected(self) -> None:
        for query in ["người dẫn nói gì", "nghe thấy tiếng còi", "buổi phỏng vấn",
                      "giọng của phóng viên"]:
            self.assertTrue(has_speech_cue(query), query)


class DefaultIsFloorsTests(unittest.TestCase):
    """ROUTE-01 đã đo và bị DROP làm mặc định — khoá lại quyết định đó."""

    def test_planner_keeps_the_floors_by_default(self) -> None:
        plan = _plan("cột nước phun lên từ lòng đất", allow_zero=False)
        self.assertAlmostEqual(plan.modality_weights[Modality.OCR], 0.35)
        self.assertAlmostEqual(plan.modality_weights[Modality.ASR], 0.25)

    def test_default_constructor_does_not_zero_gate(self) -> None:
        import asyncio

        plan = asyncio.run(
            RuleBasedQueryPlanner().plan(
                SearchRequest(query="cột nước phun lên từ lòng đất", task=TaskType.TEXTUAL_KIS)
            )
        )
        self.assertGreater(plan.modality_weights[Modality.OCR], 0.0)


class ZeroGatingTests(unittest.TestCase):
    def test_visual_query_zeroes_ocr_and_asr(self) -> None:
        weights = compute_modality_weights("cột nước phun lên từ lòng đất", [])
        self.assertEqual(weights[Modality.OCR], 0.0)
        self.assertEqual(weights[Modality.ASR], 0.0)
        # Nhánh thị giác/caption phải giữ nguyên.
        self.assertGreater(weights[Modality.VISUAL], 0.0)
        self.assertGreater(weights[Modality.CAPTION], 0.0)

    def test_text_cue_enables_ocr_only(self) -> None:
        weights = compute_modality_weights("biển báo ghi chữ gì", [])
        self.assertGreater(weights[Modality.OCR], 0.0)
        self.assertEqual(weights[Modality.ASR], 0.0)

    def test_speech_cue_enables_asr_only(self) -> None:
        weights = compute_modality_weights("người dẫn chương trình nói gì", [])
        self.assertGreater(weights[Modality.ASR], 0.0)
        self.assertEqual(weights[Modality.OCR], 0.0)

    def test_allow_zero_false_restores_the_old_floors(self) -> None:
        """Nhánh A của ablation phải tái lập được đúng hành vi cũ."""

        weights = compute_modality_weights("cột nước phun lên từ lòng đất", [], allow_zero=False)
        self.assertAlmostEqual(weights[Modality.OCR], 0.35)
        self.assertAlmostEqual(weights[Modality.ASR], 0.25)


class ExactMatchExemptionTests(unittest.TestCase):
    """`ocr_fuzzy` khớp gần-nguyên-chuỗi nên không tạo được false positive kiểu
    "trùng vài token phổ biến"; tắt nó theo modality OCR chỉ làm mất recall của
    truy vấn gõ thẳng chữ nhìn thấy trên màn hình (không hề chứa cue)."""

    def test_exact_match_branch_stays_alive_when_ocr_is_routed_off(self) -> None:
        plan = _plan("hen ngay gap lai")
        self.assertEqual(plan.modality_weights[Modality.OCR], 0.0)
        self.assertIn("ocr_fuzzy", plan.search_options.branches)
        self.assertTrue(plan.search_options.branches["ocr_fuzzy"].enabled)
        self.assertGreater(plan.search_options.branches["ocr_fuzzy"].weight, 0.0)

    def test_no_exemption_when_ocr_is_already_on(self) -> None:
        plan = _plan("biển báo ghi chữ gì")
        self.assertNotIn("ocr_fuzzy", plan.search_options.branches)

    def test_user_branch_config_is_never_overwritten(self) -> None:
        import asyncio

        from online.domain.search_config import BranchRuntimeOptions, SearchOptions

        options = SearchOptions(branches={"ocr_fuzzy": BranchRuntimeOptions(enabled=False)})
        planner = RuleBasedQueryPlanner(allow_zero_modality=True)
        plan = asyncio.run(
            planner.plan(
                SearchRequest(query="cột nước phun lên", task=TaskType.TEXTUAL_KIS,
                              search_options=options)
            )
        )
        self.assertFalse(plan.search_options.branches["ocr_fuzzy"].enabled)


if __name__ == "__main__":
    unittest.main()
