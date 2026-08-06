"""DIAG-01 — `resolve_gold_text` phải phủ cả bốn task, và không im lặng.

Lỗi gốc: một công cụ audit chỉ đọc `query_vi` nên bỏ sót toàn bộ 36 truy vấn
QA. Nó không báo lỗi, chỉ đếm thiếu — nên kết luận "0 truy vấn bị ảnh hưởng"
trông vẫn hợp lệ. Test ở đây khoá đúng chỗ đó.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from online.competition.gold_text import resolve_gold_text, resolve_gold_text_detailed

GOLD = Path("examples/gold_all3.jsonl")


class ResolveGoldTextTests(unittest.TestCase):
    def test_kis_uses_query_vi(self) -> None:
        got = resolve_gold_text_detailed(
            {"query_id": "X_KIS_E01", "task": "KIS", "query_vi": "Tìm cảnh biển vàng",
             "query_en": "Find the yellow sign"}
        )
        self.assertEqual(got.text, "Tìm cảnh biển vàng")
        self.assertEqual(got.source_keys, ("query_vi",))

    def test_qa_joins_event_context_with_the_question(self) -> None:
        got = resolve_gold_text_detailed({
            "query_id": "X_VQA_M03", "task": "VQA",
            "event_description_vi": "Cháy rừng trên sườn đồi",
            "question_vi": "Phương tiện nào bay trên không?",
        })
        self.assertEqual(got.text, "Cháy rừng trên sườn đồi Phương tiện nào bay trên không?")
        self.assertEqual(got.source_keys, ("event_description_vi", "question_vi"))

    def test_qa_without_context_still_returns_the_question(self) -> None:
        got = resolve_gold_text_detailed(
            {"query_id": "X", "task": "VQA", "question_vi": "Có bao nhiêu xe?"}
        )
        self.assertEqual(got.text, "Có bao nhiêu xe?")

    def test_falls_back_to_english_then_raw_query(self) -> None:
        self.assertEqual(
            resolve_gold_text({"query_id": "X", "query_en": "yellow sign"}), "yellow sign"
        )
        self.assertEqual(
            resolve_gold_text({"query_id": "X", "raw_query": "biển vàng"}), "biển vàng"
        )

    def test_missing_every_key_raises_instead_of_returning_empty(self) -> None:
        """Trả chuỗi rỗng = truy vấn biến mất khỏi tập đo mà không ai biết."""

        with self.assertRaises(KeyError) as ctx:
            resolve_gold_text({"query_id": "X_AVS_H01", "task": "AVS"})
        self.assertIn("X_AVS_H01", str(ctx.exception))

    def test_blank_string_counts_as_missing(self) -> None:
        with self.assertRaises(KeyError):
            resolve_gold_text({"query_id": "X", "query_vi": "   "})


@unittest.skipUnless(GOLD.exists(), "cần examples/gold_all3.jsonl")
class RealGoldCoverageTests(unittest.TestCase):
    """Regression trên chính bộ gold thật — nơi lỗi đã xảy ra."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = [
            json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_every_gold_record_resolves(self) -> None:
        for record in self.records:
            with self.subTest(query_id=record["query_id"]):
                self.assertTrue(resolve_gold_text(record).strip())

    def test_all_four_tasks_are_covered(self) -> None:
        tasks = {record["task"] for record in self.records}
        self.assertEqual(tasks, {"KIS", "VQA", "TRAKE", "AVS"})
        for task in sorted(tasks):
            rows = [r for r in self.records if r["task"] == task]
            with self.subTest(task=task):
                self.assertTrue(all(resolve_gold_text(r).strip() for r in rows))

    def test_v001_vqa_m03_is_not_silently_skipped(self) -> None:
        """Đúng truy vấn mà công cụ chẩn đoán đã bỏ sót.

        Nó không có `query_vi`, chỉ có `question_vi`, và nó mang mệnh đề
        "bay trên không và mang túi nước" — constraint dương-tính-giả nguy hiểm
        nhất trong bộ gold.
        """

        record = next(r for r in self.records if r["query_id"] == "V001_VQA_M03")
        self.assertNotIn("query_vi", record)
        text = resolve_gold_text(record)
        self.assertIn("trên không", text)

    def test_qa_records_have_no_query_vi_so_the_fallback_is_load_bearing(self) -> None:
        qa = [r for r in self.records if r["task"] == "VQA"]
        self.assertEqual(len(qa), 36)
        self.assertFalse([r for r in qa if r.get("query_vi")])


if __name__ == "__main__":
    unittest.main()
