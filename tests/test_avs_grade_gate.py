"""AVS-GRADE-01: `grade()` là cổng TỪ VỰNG đứng chắn sau retrieval ngữ nghĩa.

`AvsCriteria.grade()` chấm 0–3 bằng khớp token nguyên văn trên các nhóm
inclusion, và `min_grade = 1` loại thẳng mọi candidate không khớp chữ nào.
Toàn bộ CLIP + dense + rerank phía trên bị một bộ lọc từ vựng ở bước cuối vứt
đi. Đo được trên bộ gold: **3/8 truy vấn trả về đúng 0 kết quả** dù top-100
được phép.

Cùng loại lỗi với lexicon VI→EN: một tầng rule-based viết theo giả định cũ,
đứng trước tầng ngữ nghĩa.
"""

from __future__ import annotations

import unittest

from online.domain.evidence import EvidencePack
from online.services.avs import AvsConfig, AvsProcessor, extract_criteria


def _pack(candidate_id: str, caption: str) -> EvidencePack:
    return EvidencePack(
        candidate_id=candidate_id,
        video_id="L21_V001",
        scene_id=candidate_id,
        start_frame=0,
        end_frame_exclusive=100,
        start_sec=0.0,
        end_sec=4.0,
        best_frame_idx=10,
        caption_text=caption,
    )


# Truy vấn và caption CÙNG NGHĨA nhưng gần như không chung token: "phương
# tiện cứu hộ" vs "xe cứu thương", "tiến vào hiện trường" vs "chạy trên đường".
# Đây đúng là loại cặp mà retrieval ngữ nghĩa bắt được còn khớp token thì không.
QUERY = "phương tiện cứu hộ tiến vào hiện trường"
SEMANTIC_MATCH = _pack("S_amb", "xe cứu thương đang chạy trên đường")
LEXICAL_MATCH = _pack("S_lex", "phương tiện cứu hộ tiến vào hiện trường vụ tai nạn")

# Điểm ngữ nghĩa cao cho candidate đúng-nghĩa, thấp cho candidate lạc đề.
SCORES = {"S_amb": 0.71, "S_lex": 0.68, "S_junk": 0.02}
JUNK = _pack("S_junk", "bản tin thể thao buổi sáng")


class HardGateDropsSemanticMatchTests(unittest.TestCase):
    def _run(self, mode: str, packs: list[EvidencePack]) -> tuple[list, dict]:
        processor = AvsProcessor(AvsConfig(grade_mode=mode, max_per_video=10))
        diagnostics: dict = {}
        results = processor.rank(
            QUERY, packs, retrieval_scores=SCORES, diagnostics=diagnostics
        )
        return results, diagnostics

    def test_hard_gate_no_longer_drops_the_semantically_correct_candidate(self) -> None:
        """AVS-CRITERIA-01 đã cứu đúng cặp này, và cổng vẫn loại được rác.

        Bản đầu của test khoá hành vi NGƯỢC LẠI: `xe cứu thương đang chạy trên
        đường` mô tả đúng cảnh truy vấn hỏi và có điểm ngữ nghĩa cao nhất
        (0.71), nhưng khớp chuỗi con cho `grade = 0` nên cổng cứng loại thẳng.
        Nó tồn tại để chứng minh vấn đề, không phải để bảo vệ hành vi đó.

        Nay `grade()` chấm bằng độ phủ token có trọng số IDF: `cứu` chung giữa
        `cứu hộ` và `cứu thương` đủ cho grade 1. Điều đáng kiểm là cổng KHÔNG
        vì thế mà mở toang — `S_junk` vẫn phải bị loại.
        """

        results, diagnostics = self._run("hard_gate", [SEMANTIC_MATCH, JUNK])
        kept = {item.segment_id for item in results}
        self.assertIn("S_amb", kept, "candidate đúng nghĩa lẽ ra phải sống sót")
        self.assertNotIn("S_junk", kept, "cổng vẫn phải loại được candidate lạc đề")
        dropped = {row["candidate_id"]: row for row in diagnostics["dropped"]}
        self.assertEqual(dropped["S_junk"]["drop_reason"], "min_grade")
        self.assertEqual(dropped["S_junk"]["lexical_grade"], 0)

    def test_lexical_match_still_outgrades_the_paraphrase(self) -> None:
        """Cứu được bản diễn đạt khác không có nghĩa xoá mất thứ tự ưu tiên."""

        criteria = extract_criteria(QUERY)
        self.assertGreater(
            criteria.grade(LEXICAL_MATCH.caption_text or ""),
            criteria.grade(SEMANTIC_MATCH.caption_text or ""),
        )

    def test_soft_grade_keeps_it(self) -> None:
        results, diagnostics = self._run("soft", [SEMANTIC_MATCH, JUNK])
        self.assertIn("S_amb", {item.segment_id for item in results})
        self.assertEqual(diagnostics["dropped"], [])

    def test_semantic_or_lexical_keeps_it_but_still_drops_junk(self) -> None:
        """Đường cứu hộ: điểm ngữ nghĩa đủ cao thì tự cứu, còn rác vẫn bị loại.

        Khác `no_gate`/`soft` ở chỗ nó KHÔNG mở toang — candidate vừa không
        khớp chữ vừa không đủ điểm ngữ nghĩa vẫn bị vứt.
        """

        results, diagnostics = self._run(
            "semantic_or_lexical", [SEMANTIC_MATCH, JUNK]
        )
        kept = {item.segment_id for item in results}
        self.assertIn("S_amb", kept)
        self.assertNotIn("S_junk", kept)
        reasons = {row["candidate_id"]: row["drop_reason"] for row in diagnostics["dropped"]}
        self.assertEqual(reasons.get("S_junk"), "below_semantic_tau_and_min_grade")

    def test_no_gate_keeps_everything_including_junk(self) -> None:
        """`no_gate` là biến thể CHẨN ĐOÁN, không phải để dùng thật.

        Nó xác nhận cổng là thủ phạm, nhưng đổi lại là nhồi danh sách bằng rác —
        nên không được promote chỉ vì `zero_result_rate` biến mất.
        """

        results, diagnostics = self._run("no_gate", [SEMANTIC_MATCH, JUNK])
        self.assertEqual(diagnostics["dropped"], [])
        self.assertIn("S_junk", {item.segment_id for item in results})

    def test_diagnostics_report_counts_on_both_sides_of_the_gate(self) -> None:
        _results, diagnostics = self._run("hard_gate", [SEMANTIC_MATCH, LEXICAL_MATCH, JUNK])
        self.assertEqual(diagnostics["pre_grade_candidate_count"], 3)
        self.assertEqual(
            diagnostics["post_grade_candidate_count"],
            3 - len(diagnostics["dropped"]),
        )
        self.assertEqual(diagnostics["grade_mode"], "hard_gate")

    def test_soft_mode_ranks_by_semantics_not_by_token_overlap(self) -> None:
        """Công thức cũ để grade CHIẾM ƯU THẾ: `0.7*(grade/3) + 0.3*semantic`.

        Nghĩa là khớp token quyết định thứ hạng còn điểm ngữ nghĩa chỉ phụ.
        Mode `soft` đảo lại.
        """

        weak_lexical = _pack("S_weak", "phương tiện cứu hộ")  # khớp chữ, cảnh sai
        scores = {"S_amb": 0.95, "S_weak": 0.05}
        processor = AvsProcessor(AvsConfig(grade_mode="soft", max_per_video=10))
        results = processor.rank(QUERY, [weak_lexical, SEMANTIC_MATCH], retrieval_scores=scores)
        self.assertEqual(results[0].segment_id, "S_amb")


if __name__ == "__main__":
    unittest.main()
