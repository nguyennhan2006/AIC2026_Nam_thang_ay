"""PR-02: harness chấm 4 task phải chấm ở mức FRAME, đúng luật thi.

Chấm ở mức scene (cách `scripts/eval_kis.py` làm) cho điểm cao giả tạo: một
scene dài 10 giây chồng lấn interval GT vẫn tính là hit kể cả khi frame được
nộp nằm ngoài interval. Các test dưới đây khóa lại ngữ nghĩa đúng.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from offline.indexing import frame_rows
from scripts.eval_tasks import (
    GoldQuery,
    Interval,
    _frame_hit,
    answer_matches,
    load_gold,
    ndcg_at_k,
    normalize_answer,
)
from online.domain.tasks import TaskType

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples" / "AIC2026_L21_V001_queries_4tasks.jsonl"


class _Hit:
    """SearchHit tối giản — harness chỉ cần video_id + best_frame_idx."""

    def __init__(self, video_id: str, frame_idx: int) -> None:
        self.video_id = video_id
        self.best_frame_idx = frame_idx


class FrameLevelScoringTests(unittest.TestCase):
    def _gold(self) -> GoldQuery:
        return GoldQuery(
            query_id="q1",
            task=TaskType.TEXTUAL_KIS,
            query="x",
            video_id="L21_V001",
            intervals=(Interval(start_frame=2175, end_frame=2355),),
        )

    def test_frame_inside_interval_is_a_hit(self) -> None:
        self.assertTrue(_frame_hit(_Hit("L21_V001", 2250), self._gold()))
        self.assertTrue(_frame_hit(_Hit("L21_V001", 2175), self._gold()))
        self.assertTrue(_frame_hit(_Hit("L21_V001", 2355), self._gold()))

    def test_frame_outside_interval_is_not_a_hit(self) -> None:
        self.assertFalse(_frame_hit(_Hit("L21_V001", 2174), self._gold()))
        self.assertFalse(_frame_hit(_Hit("L21_V001", 2356), self._gold()))

    def test_right_frame_in_the_wrong_video_is_not_a_hit(self) -> None:
        self.assertFalse(_frame_hit(_Hit("L21_V002", 2250), self._gold()))


class AnswerMatchingTests(unittest.TestCase):
    def test_normalization_strips_vietnamese_diacritics(self) -> None:
        self.assertEqual(normalize_answer("Hơn 14,5 tỷ đồng"), "hon 14 5 ty dong")

    def test_accepted_variants_all_match(self) -> None:
        accepted = ("5", "năm", "five")
        for predicted in ("5", "Năm", "five motorcycles", "có 5 xe máy"):
            with self.subTest(predicted=predicted):
                self.assertTrue(answer_matches(predicted, accepted))

    def test_empty_prediction_never_matches(self) -> None:
        self.assertFalse(answer_matches("", ("5",)))


class NdcgTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([3, 3, 2], [3, 3, 2], 3), 1.0)

    def test_relevant_items_ranked_low_score_less(self) -> None:
        good = ndcg_at_k([3, 0, 0], [3, 3], 3)
        bad = ndcg_at_k([0, 0, 3], [3, 3], 3)
        self.assertGreater(good, bad)


class GoldLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        if not GOLD.exists():
            self.skipTest(f"gold benchmark chưa có: {GOLD}")

    def test_loads_all_forty_queries_with_canonical_task_names(self) -> None:
        gold = load_gold(GOLD)
        self.assertEqual(len(gold), 40)
        counts: dict[TaskType, int] = {}
        for item in gold:
            counts[item.task] = counts.get(item.task, 0) + 1
        self.assertEqual(counts[TaskType.TEXTUAL_KIS], 12)
        self.assertEqual(counts[TaskType.QA], 12)  # gold ghi "VQA"
        self.assertEqual(counts[TaskType.TRAKE], 8)
        self.assertEqual(counts[TaskType.AVS], 8)

    def test_every_query_has_frame_level_ground_truth(self) -> None:
        for item in load_gold(GOLD):
            with self.subTest(query=item.query_id):
                self.assertTrue(item.query)
                if item.task == TaskType.TRAKE:
                    self.assertTrue(item.steps)
                    for step in item.steps:
                        self.assertLessEqual(step.start_frame, step.end_frame)
                else:
                    self.assertTrue(item.intervals)

    def test_trake_windows_are_nine_frames_as_the_schema_states(self) -> None:
        for item in load_gold(GOLD, {TaskType.TRAKE}):
            for step in item.steps:
                with self.subTest(query=item.query_id, step=step.event_id):
                    self.assertEqual(step.end_frame - step.start_frame + 1, 9)

    def test_qa_queries_carry_accepted_answers(self) -> None:
        for item in load_gold(GOLD, {TaskType.QA}):
            with self.subTest(query=item.query_id):
                self.assertTrue(item.accepted_answers)


class FrameIndexTests(unittest.TestCase):
    def test_frame_rows_carry_submission_coordinates(self) -> None:
        exports = ROOT / "storage" / "exports"
        if not (exports / "keyframes.jsonl").exists():
            self.skipTest("chưa có export; chạy scripts/seed_demo.py trước")
        rows, degraded = frame_rows(exports, ROOT / "storage")
        self.assertTrue(rows)
        for row in rows:
            payload = row["payload"]
            with self.subTest(keyframe=payload["keyframe_id"]):
                self.assertIsInstance(payload["frame_idx"], int)
                self.assertEqual(payload["entity_type"], "keyframe")
                self.assertTrue(payload["video_id"])
                self.assertTrue(payload["scene_id"])
                self.assertTrue(
                    payload["start_frame"] <= payload["frame_idx"] <= payload["end_frame"]
                )
        # Export demo chưa có embedding thật -> phải báo degraded, không im lặng.
        self.assertTrue(degraded)

    def test_frame_rows_prefer_real_embeddings_over_hashing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exports = root / "exports"
            exports.mkdir()
            (root / "vectors").mkdir()
            (root / "vectors" / "f.json").write_text("[0.1, 0.2, 0.3]", encoding="utf-8")
            (exports / "scenes.jsonl").write_text(
                json.dumps({
                    "scene_id": "L21_V001_S0000", "video_id": "L21_V001",
                    "start_frame": 0, "end_frame_exclusive": 100,
                    "start_sec": 0.0, "end_sec": 4.0, "asr_segments": [],
                }) + "\n",
                encoding="utf-8",
            )
            (exports / "keyframes.jsonl").write_text(
                json.dumps({
                    "keyframe_id": "L21_V001_S0000_F000050", "scene_id": "L21_V001_S0000",
                    "video_id": "L21_V001", "frame_idx": 50, "timestamp_sec": 2.0,
                    "image_path": "processed/keyframes/L21_V001/frame_000050.jpg",
                    "ocr_instances": [],
                    "embedding_refs": [{
                        "embedding_name": "openclip_l14",
                        "storage_locations": [{"backend": "file", "vector_uri": "vectors/f.json"}],
                    }],
                }) + "\n",
                encoding="utf-8",
            )
            rows, degraded = frame_rows(exports, root)
            self.assertFalse(degraded)
            self.assertEqual(rows[0]["vector"], [0.1, 0.2, 0.3])
            self.assertEqual(rows[0]["vector_name"], "openclip_l14")


if __name__ == "__main__":
    unittest.main()
