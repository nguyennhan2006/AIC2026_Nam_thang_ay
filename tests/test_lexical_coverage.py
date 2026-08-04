"""BM25-01: coverage phải phân biệt được khớp-đủ-ý với khớp-một-mảnh.

Bối cảnh đo được (docs/20_EXPERIMENT_LOG.md): scene ngẫu nhiên đã trùng sẵn
2.25 token OCR / 6.75 token ASR với query, nên cộng điểm theo số token khớp
sẽ thưởng cả candidate chỉ tình cờ dùng chung vài từ phổ biến.

Test ở đây khoá lại CẢ hai kết quả: cái chạy được (IDF/unique coverage) và
cái KHÔNG chạy được (concept group bằng luật hư từ) — để không ai giả định
nhầm rằng nhóm khái niệm đang hoạt động.
"""

from __future__ import annotations

import unittest

from online.services.lexical_coverage import (
    CoverageConfig,
    compute_coverage,
    concept_groups,
    content_tokens,
)

QUERY = "cột nước phun lên từ lòng đất"
GOLD_TEXT = "một cột nước cao vút phun lên từ giếng, bao quanh là cây cỏ"
NEGATIVE_TEXT = "tìm kiếm nạn nhân trong các trận lở đất ở Ấn Độ"


class ContentTokenTests(unittest.TestCase):
    def test_function_words_are_dropped(self) -> None:
        tokens = content_tokens(QUERY)
        self.assertNotIn("từ", tokens)
        for token in ("cột", "nước", "phun", "lòng", "đất"):
            self.assertIn(token, tokens)


class CoverageDiscriminationTests(unittest.TestCase):
    def test_partial_single_token_match_receives_penalty(self) -> None:
        """Candidate chỉ khớp "đất" phải bị chấm thấp hơn hẳn."""

        gold = compute_coverage(QUERY, GOLD_TEXT)
        negative = compute_coverage(QUERY, NEGATIVE_TEXT)
        self.assertGreater(gold.unique, negative.unique)
        self.assertGreater(gold.idf_weighted, negative.idf_weighted)
        self.assertEqual(negative.matched_terms, ("đất",))

    def test_multi_concept_match_beats_single_common_token_match(self) -> None:
        config = CoverageConfig(idf_weight=0.5, partial_penalty=0.3)
        gold = compute_coverage(QUERY, GOLD_TEXT).adjustment(config)
        negative = compute_coverage(QUERY, NEGATIVE_TEXT).adjustment(config)
        self.assertGreater(gold, negative)

    def test_exact_phrase_query_is_not_penalised(self) -> None:
        """Truy vấn gõ thẳng chữ trên màn hình phải khớp trọn, không bị phạt."""

        query = "hẹn ngày gặp lại"
        result = compute_coverage(query, "biển ghi HẸN NGÀY GẶP LẠI ở cuối chương trình")
        self.assertEqual(result.unique, 1.0)
        self.assertEqual(result.group, 1.0)
        self.assertGreater(result.phrase, 0.0)
        self.assertGreaterEqual(result.adjustment(CoverageConfig(idf_weight=0.5, partial_penalty=0.3)), 0.0)


class ConceptGroupLimitationTests(unittest.TestCase):
    """Khoá lại một GIỚI HẠN ĐÃ ĐO, không phải một tính năng.

    Tách nhóm bằng hư từ không hoạt động với tiếng Việt viết liền: "cột nước
    phun lên" không có hư từ ở giữa nên thành MỘT nhóm thay vì hai (chất +
    chuyển động). Hệ quả là gold và false positive "lở đất" nhận cùng điểm
    nhóm 0.5, tức cơ chế vô hiệu đúng trên case nó sinh ra để xử lý.

    Muốn nhóm khái niệm chạy thật thì cần từ điển cụm hoặc LLM decomposition
    (§5.2 của tài liệu Experiment Validation giả định concept_groups do LLM
    sinh) — không phải chỉnh thêm danh sách hư từ.
    """

    def test_rule_based_segmentation_undersplits_vietnamese(self) -> None:
        groups = concept_groups(QUERY)
        self.assertEqual(len(groups), 2)
        self.assertIn("phun", groups[0])
        self.assertIn("nước", groups[0])

    def test_group_coverage_cannot_separate_gold_from_the_known_negative(self) -> None:
        gold = compute_coverage(QUERY, GOLD_TEXT)
        negative = compute_coverage(QUERY, NEGATIVE_TEXT)
        self.assertEqual(gold.group, negative.group)


class NoopConfigTests(unittest.TestCase):
    def test_default_config_is_noop(self) -> None:
        self.assertTrue(CoverageConfig().is_noop)

    def test_noop_config_adds_nothing(self) -> None:
        result = compute_coverage(QUERY, NEGATIVE_TEXT)
        self.assertEqual(result.adjustment(CoverageConfig()), 0.0)


if __name__ == "__main__":
    unittest.main()
