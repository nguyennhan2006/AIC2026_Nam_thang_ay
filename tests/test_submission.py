"""PR-08: tầng nộp bài — build/validate/scorer, tách khỏi SearchService.

Ba rủi ro cụ thể các test này chặn:

1. CSV sai format BTC (có header, sai cột, sai kiểu ngoặc kép).
2. Validator không bắt được "frame_idx vượt quá độ dài video" — dấu hiệu
   kinh điển của việc lỡ nộp thứ tự keyframe thay vì true frame index.
3. Scorer tính sai luật (đặc biệt TRAKE: sai video phải là 0, không phải
   điểm trung bình một phần).
"""

from __future__ import annotations

import asyncio
import unittest

from online.competition.ranking_planner import annotate_zones, zone_summary
from online.competition.rules import MAX_SUBMISSION_ITEMS, zone_for_rank
from online.competition.scorer import (
    GoldInterval,
    KisGold,
    QaGold,
    TrakeGold,
    score_kis,
    score_qa,
    score_trake,
)
from online.competition.submission_builder import (
    build_kis_submission,
    build_qa_submission,
    build_trake_submission,
    kis_to_csv,
    qa_to_csv,
    trake_to_csv,
)
from online.competition.submission_validator import (
    has_errors,
    validate_kis,
    validate_qa,
    validate_trake,
)
from online.domain.submission import KisSubmissionItem, QaSubmissionItem, TrakeSubmissionItem
from online.domain.task_results import KisResultItem, QaResultItem, TrakeResultItem, TrakeStep


def run(coro):
    return asyncio.run(coro)


def kis_result(video: str, frame: int, rank: int, score: float = 1.0) -> KisResultItem:
    return KisResultItem(rank=rank, video_id=video, frame_idx=frame, score=score)


def qa_result(video: str, frame: int, answer: str, rank: int) -> QaResultItem:
    return QaResultItem(
        rank=rank, video_id=video, frame_idx=frame, answer=answer,
        canonical_answer=answer, joint_score=0.5, verifier_status="SUPPORTED",
    )


def trake_result(video: str, frame_ids: list[int], rank: int) -> TrakeResultItem:
    return TrakeResultItem(
        rank=rank, video_id=video, frame_ids=frame_ids, sequence_score=0.5,
        steps=[TrakeStep(step=i + 1, frame_idx=f, confidence=0.5) for i, f in enumerate(frame_ids)],
    )


class BuilderTests(unittest.TestCase):
    def test_kis_csv_has_no_header_and_two_columns(self) -> None:
        items = build_kis_submission([kis_result("L21_V001", 100, 1)])
        csv_text = kis_to_csv(items)
        self.assertEqual(csv_text.strip(), "L21_V001,100")

    def test_qa_csv_includes_answer_as_third_column(self) -> None:
        items = build_qa_submission([qa_result("L21_V001", 100, "5", 1)])
        csv_text = qa_to_csv(items)
        self.assertEqual(csv_text.strip(), "L21_V001,100,5")

    def test_qa_answer_containing_a_comma_is_quoted(self) -> None:
        items = build_qa_submission([qa_result("L21_V001", 100, "5, có thể 6", 1)])
        csv_text = qa_to_csv(items)
        self.assertIn('"5, có thể 6"', csv_text)

    def test_trake_csv_lists_every_frame_id_on_one_row(self) -> None:
        items = build_trake_submission([trake_result("L21_V001", [100, 200, 300], 1)])
        csv_text = trake_to_csv(items)
        self.assertEqual(csv_text.strip(), "L21_V001,100,200,300")

    def test_output_is_truncated_to_the_max_and_kept_in_rank_order(self) -> None:
        results = [kis_result("L21_V001", i, rank=i + 1) for i in range(150)]
        items = build_kis_submission(results)
        self.assertEqual(len(items), MAX_SUBMISSION_ITEMS)
        self.assertEqual(items[0].frame_idx, 0)
        self.assertEqual(items[-1].frame_idx, MAX_SUBMISSION_ITEMS - 1)

    def test_builder_sorts_by_rank_even_if_input_order_differs(self) -> None:
        results = [kis_result("L21_V001", 999, rank=2), kis_result("L21_V001", 111, rank=1)]
        items = build_kis_submission(results)
        self.assertEqual(items[0].frame_idx, 111)


class ValidatorTests(unittest.TestCase):
    async def _lookup(self, counts: dict[str, int]):
        async def lookup(video_id: str) -> int | None:
            return counts.get(video_id)
        return lookup

    def test_empty_submission_is_an_error(self) -> None:
        issues = run(validate_kis([]))
        self.assertTrue(has_errors(issues))
        self.assertEqual(issues[0].code, "empty_submission")

    def test_over_the_row_limit_is_an_error(self) -> None:
        items = [KisSubmissionItem(video_id="L21_V001", frame_idx=i) for i in range(101)]
        issues = run(validate_kis(items))
        self.assertTrue(any(item.code == "too_many_rows" for item in issues))

    def test_negative_frame_is_an_error(self) -> None:
        issues = run(validate_kis([KisSubmissionItem(video_id="L21_V001", frame_idx=-1)]))
        self.assertTrue(any(item.code == "negative_frame" for item in issues))

    def test_frame_beyond_video_length_is_an_error(self) -> None:
        lookup = run(self._lookup({"L21_V001": 500}))
        issues = run(validate_kis(
            [KisSubmissionItem(video_id="L21_V001", frame_idx=600)], frame_count=lookup
        ))
        self.assertTrue(any(item.code == "frame_out_of_bounds" for item in issues))

    def test_frame_within_video_length_passes(self) -> None:
        lookup = run(self._lookup({"L21_V001": 500}))
        issues = run(validate_kis(
            [KisSubmissionItem(video_id="L21_V001", frame_idx=499)], frame_count=lookup
        ))
        self.assertFalse(has_errors(issues))

    def test_unknown_video_is_a_warning_not_an_error(self) -> None:
        lookup = run(self._lookup({}))
        issues = run(validate_kis(
            [KisSubmissionItem(video_id="L21_V999", frame_idx=100)], frame_count=lookup
        ))
        self.assertTrue(any(item.code == "unknown_video" and item.severity == "warning" for item in issues))
        self.assertFalse(has_errors(issues))

    def test_empty_qa_answer_is_an_error(self) -> None:
        issues = run(validate_qa([QaSubmissionItem(video_id="L21_V001", frame_idx=100, answer="")]))
        self.assertTrue(any(item.code == "empty_answer" for item in issues))

    def test_qa_answer_over_100_chars_is_a_warning(self) -> None:
        long_answer = "x" * 150
        issues = run(validate_qa(
            [QaSubmissionItem(video_id="L21_V001", frame_idx=100, answer=long_answer)]
        ))
        self.assertTrue(any(item.code == "answer_too_long" for item in issues))

    def test_trake_frames_not_strictly_increasing_is_an_error(self) -> None:
        issues = run(validate_trake(
            [TrakeSubmissionItem(video_id="L21_V001", frame_ids=[300, 200, 400])]
        ))
        self.assertTrue(any(item.code == "frames_not_increasing" for item in issues))

    def test_trake_wrong_step_count_is_an_error(self) -> None:
        issues = run(validate_trake(
            [TrakeSubmissionItem(video_id="L21_V001", frame_ids=[100, 200])], expected_steps=4,
        ))
        self.assertTrue(any(item.code == "wrong_step_count" for item in issues))

    def test_duplicate_rows_are_flagged(self) -> None:
        items = [
            KisSubmissionItem(video_id="L21_V001", frame_idx=100),
            KisSubmissionItem(video_id="L21_V001", frame_idx=100),
        ]
        issues = run(validate_kis(items))
        self.assertTrue(any(item.code == "duplicate_row" for item in issues))

    def test_valid_submission_has_no_errors(self) -> None:
        lookup = run(self._lookup({"L21_V001": 1000}))
        items = [KisSubmissionItem(video_id="L21_V001", frame_idx=i * 10) for i in range(5)]
        issues = run(validate_kis(items, frame_count=lookup))
        self.assertFalse(has_errors(issues))


class ScorerTests(unittest.TestCase):
    def test_kis_scores_one_when_correct_video_and_frame_in_interval(self) -> None:
        gold = KisGold("L21_V001", (GoldInterval(2175, 2355),))
        items = [KisSubmissionItem(video_id="L21_V001", frame_idx=2250)]
        result = score_kis(items, gold)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.best_rank, 1)

    def test_kis_scores_zero_on_wrong_video(self) -> None:
        gold = KisGold("L21_V001", (GoldInterval(2175, 2355),))
        items = [KisSubmissionItem(video_id="L21_V002", frame_idx=2250)]
        self.assertEqual(score_kis(items, gold).score, 0.0)

    def test_kis_finds_the_first_matching_row_beyond_rank_one(self) -> None:
        gold = KisGold("L21_V001", (GoldInterval(2175, 2355),))
        items = [
            KisSubmissionItem(video_id="L21_V001", frame_idx=100),
            KisSubmissionItem(video_id="L21_V001", frame_idx=2300),
        ]
        result = score_kis(items, gold)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.best_rank, 2)

    def test_qa_requires_frame_and_answer_both_correct(self) -> None:
        gold = QaGold("L21_V001", (GoldInterval(100, 200),), ("5", "năm"))
        right_frame_wrong_answer = [QaSubmissionItem(video_id="L21_V001", frame_idx=150, answer="9")]
        self.assertEqual(score_qa(right_frame_wrong_answer, gold).score, 0.0)
        right_both = [QaSubmissionItem(video_id="L21_V001", frame_idx=150, answer="5")]
        self.assertEqual(score_qa(right_both, gold).score, 1.0)

    def test_qa_accepts_any_alias_in_the_accepted_list(self) -> None:
        gold = QaGold("L21_V001", (GoldInterval(100, 200),), ("5", "năm", "five"))
        items = [QaSubmissionItem(video_id="L21_V001", frame_idx=150, answer="năm")]
        self.assertEqual(score_qa(items, gold).score, 1.0)

    def test_trake_wrong_video_scores_zero_not_a_partial_average(self) -> None:
        gold = TrakeGold("L21_V001", (GoldInterval(96, 104), GoldInterval(196, 204)))
        item = TrakeSubmissionItem(video_id="L21_V002", frame_ids=[100, 200])
        result = score_trake(item, gold)
        self.assertEqual(result.score, 0.0)
        self.assertIn("sai video", result.detail)

    def test_trake_correct_video_scores_mean_step_hit_ratio(self) -> None:
        gold = TrakeGold("L21_V001", (GoldInterval(96, 104), GoldInterval(196, 204)))
        # step 1 đúng cửa sổ, step 2 lệch hẳn ra ngoài.
        item = TrakeSubmissionItem(video_id="L21_V001", frame_ids=[100, 500])
        result = score_trake(item, gold)
        self.assertEqual(result.score, 0.5)

    def test_trake_all_steps_correct_scores_one(self) -> None:
        gold = TrakeGold("L21_V001", (GoldInterval(96, 104), GoldInterval(196, 204)))
        item = TrakeSubmissionItem(video_id="L21_V001", frame_ids=[100, 200])
        self.assertEqual(score_trake(item, gold).score, 1.0)


class RankingZoneTests(unittest.TestCase):
    def test_zone_boundaries_match_the_official_cutoffs(self) -> None:
        self.assertEqual(zone_for_rank(1), "rank_1")
        self.assertEqual(zone_for_rank(2), "ranks_2_5")
        self.assertEqual(zone_for_rank(5), "ranks_2_5")
        self.assertEqual(zone_for_rank(6), "ranks_6_20")
        self.assertEqual(zone_for_rank(20), "ranks_6_20")
        self.assertEqual(zone_for_rank(21), "ranks_21_50")
        self.assertEqual(zone_for_rank(50), "ranks_21_50")
        self.assertEqual(zone_for_rank(51), "ranks_51_100")
        self.assertEqual(zone_for_rank(100), "ranks_51_100")
        self.assertEqual(zone_for_rank(101), "beyond_100")

    def test_annotate_zones_does_not_reorder_items(self) -> None:
        items = [kis_result("L21_V001", 100, rank=3), kis_result("L21_V001", 200, rank=1)]
        zoned = annotate_zones(items)
        self.assertEqual([item.item.frame_idx for item in zoned], [100, 200])
        self.assertEqual(zoned[0].zone, "ranks_2_5")
        self.assertEqual(zoned[1].zone, "rank_1")

    def test_zone_summary_counts_per_zone(self) -> None:
        items = [kis_result("L21_V001", i, rank=i + 1) for i in range(7)]
        summary = zone_summary(items)
        self.assertEqual(summary["rank_1"], 1)
        self.assertEqual(summary["ranks_2_5"], 4)
        self.assertEqual(summary["ranks_6_20"], 2)


if __name__ == "__main__":
    unittest.main()
