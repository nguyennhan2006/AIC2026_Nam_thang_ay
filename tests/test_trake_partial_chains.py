"""Chuỗi THIẾU step vẫn phải được trả về (TRAKE-PARTIAL-01).

Lý do bằng số, đo trên corpus 873 video: chỉ **14/24** truy vấn có đủ candidate
cho MỌI step. Đòi chuỗi đầy đủ nghĩa là vứt 10/24 truy vấn về không — kể cả ca
`V003_TRAKE_H02`, nơi video đúng nằm sẵn trong pool ở **4/5 step** (có step ở
hạng 7) nhưng bị loại sạch chỉ vì thiếu step còn lại.

Luật chấm cho điểm theo TỈ LỆ step đúng, nên bắt được 2/5 step vẫn hơn hẳn
không nộp gì. Bộ test này khoá lại điều đó ở cả ba tầng: dựng chuỗi, gán step,
và chấm điểm.
"""

from __future__ import annotations

import unittest

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument, SearchHit
from online.services.temporal import link_event_hits
from online.services.trake.from_sequences import to_trake_results


def _hit(video: str, scene: int, score: float = 0.04) -> SearchHit:
    scene_id = f"{video}_S{scene:04d}"
    return SearchHit(
        candidate_id=scene_id, scene_id=scene_id, video_id=video, scene_idx=scene,
        start_frame=scene * 150, end_frame_exclusive=scene * 150 + 120,
        start_sec=scene * 5.0, end_sec=scene * 5.0 + 4.0,
        best_frame_idx=scene * 150 + 60, best_timestamp_sec=scene * 5.0 + 2.0,
        score=score,
    )


class PartialChainsAreReturnedTests(unittest.TestCase):
    def test_a_video_missing_the_middle_step_is_dropped_by_default(self) -> None:
        """Hành vi CŨ, giữ nguyên khi không bật cờ — để đổi là một lựa chọn."""

        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 9)]]
        self.assertEqual(link_event_hits(steps), [])

    def test_the_same_video_survives_when_holes_are_allowed(self) -> None:
        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 9)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        gold = [s for s in sequences if s.video_id == "L01_V001"]
        self.assertTrue(gold)
        self.assertEqual(gold[0].covered_steps, [1, 3])
        self.assertEqual(gold[0].missing_steps, [2])
        self.assertEqual(gold[0].total_steps, 3)

    def test_a_step_with_no_candidate_at_all_no_longer_kills_the_query(self) -> None:
        """Một step rỗng hoàn toàn trước đây làm CẢ truy vấn trả rỗng."""

        steps = [[_hit("L01_V001", 1)], [], [_hit("L01_V001", 9)]]
        self.assertEqual(link_event_hits(steps), [])
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        self.assertTrue(sequences)
        self.assertEqual(sequences[0].covered_steps, [1, 3])

    def test_two_of_five_steps_is_enough_when_that_is_all_there_is(self) -> None:
        steps = [
            [_hit("L01_V001", 1)], [], [], [_hit("L01_V001", 12)], [],
        ]
        [sequence] = link_event_hits(
            steps, allow_missing_steps=True, min_covered_steps=2
        )
        self.assertEqual(sequence.covered_steps, [1, 4])
        self.assertEqual(sequence.missing_steps, [2, 3, 5])

    def test_min_covered_steps_is_enforced(self) -> None:
        steps = [[_hit("L01_V001", 1)], [], [], [], []]
        self.assertEqual(
            link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2), []
        )
        self.assertTrue(
            link_event_hits(steps, allow_missing_steps=True, min_covered_steps=1)
        )

    def test_a_complete_chain_still_outranks_a_holed_one(self) -> None:
        """Phạt phải đủ để xếp chuỗi đủ lên trước — nhưng chỉ để xếp hạng."""

        steps = [
            [_hit("L01_FULL", 1), _hit("L02_HOLE", 1)],
            [_hit("L01_FULL", 5)],
            [_hit("L01_FULL", 9), _hit("L02_HOLE", 9)],
        ]
        sequences = link_event_hits(
            steps, allow_missing_steps=True, min_covered_steps=2
        )
        self.assertEqual(sequences[0].video_id, "L01_FULL")
        self.assertIn("L02_HOLE", {s.video_id for s in sequences})


class StepNumbersSurviveTheHoleTests(unittest.TestCase):
    def test_a_middle_hole_does_not_shift_later_steps(self) -> None:
        """Lỗ ở GIỮA từng bị suy ra thành lỗ ở ĐUÔI, khiến step 3 bị gán nhãn
        step 2 — sai lặng lẽ và không có gì báo."""

        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 9)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        gold = next(s for s in sequences if s.video_id == "L01_V001")
        [item] = to_trake_results([gold], expected_steps=3)
        self.assertEqual([step.step for step in item.steps], [1, 3])
        self.assertEqual(item.missing_steps, [2])
        self.assertAlmostEqual(item.step_coverage, 2 / 3)
        self.assertTrue(item.degraded)

    def test_a_complete_chain_reports_no_holes(self) -> None:
        """Bật lỗ thủng thì mỗi video còn sinh thêm các biến thể thiếu step.
        Chuỗi ĐẦY ĐỦ phải đứng đầu — phạt tồn tại đúng để bảo đảm điều đó."""

        steps = [[_hit("L01_V001", 1)], [_hit("L01_V001", 5)], [_hit("L01_V001", 9)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        [item] = to_trake_results(sequences[:1], expected_steps=3)
        self.assertEqual([step.step for step in item.steps], [1, 2, 3])
        self.assertEqual(item.missing_steps, [])
        self.assertFalse(item.degraded)

    def test_one_row_per_video_collapses_the_subset_variants(self) -> None:
        """Cho phép lỗ thủng sinh ra tổ hợp con của cùng một video. Với danh
        sách để người duyệt, hạn ngạch đầu ra gộp chúng lại thành một dòng —
        dòng tốt nhất của video đó."""

        steps = [[_hit("L01_V001", 1)], [_hit("L01_V001", 5)], [_hit("L01_V001", 9)]]
        many = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        self.assertGreater(len(many), 1)
        one = link_event_hits(
            steps, allow_missing_steps=True, min_covered_steps=2,
            max_chains_per_video=1,
        )
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0].missing_steps, [])


if __name__ == "__main__":
    unittest.main()


class HoleFillingTests(unittest.TestCase):
    """Lấp step thiếu bằng keyframe THẬT, đặt đúng vị trí step.

    Không lấp thì chuỗi thủng vô dụng cho việc nộp bài: validator đòi đúng
    `expected_steps` frame mỗi dòng, nên một dòng 2/5 frame bị từ chối thẳng và
    2 step tìm đúng cũng thành 0 điểm.
    """

    def _documents(self, video: str, frames: list[int]) -> dict[str, SceneDocument]:
        docs: dict[str, SceneDocument] = {}
        for frame in frames:
            scene = frame // 150
            scene_id = f"{video}_S{scene:04d}"
            docs[scene_id] = SceneDocument(
                scene_id=scene_id, video_id=video, scene_idx=scene,
                start_frame=scene * 150, end_frame_exclusive=scene * 150 + 150,
                start_sec=scene * 5.0, end_sec=scene * 5.0 + 5.0,
                keyframes=[FrameEvidence(
                    keyframe_id=f"{scene_id}_F{frame:06d}", video_id=video,
                    scene_id=scene_id, frame_idx=frame,
                    timestamp_sec=frame / 30.0, image_path=f"kf/{frame}.jpg",
                )],
            )
        return docs

    def test_a_hole_is_filled_with_a_real_keyframe_at_the_right_position(self) -> None:
        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 9)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        gold = next(s for s in sequences if s.video_id == "L01_V001")
        documents = self._documents("L01_V001", [210, 810, 1410])
        [item] = to_trake_results([gold], expected_steps=3, documents=documents)

        self.assertEqual([step.step for step in item.steps], [1, 2, 3])
        self.assertEqual(len(item.frame_ids), 3)
        self.assertEqual(item.missing_steps, [])
        filled = item.steps[1]
        self.assertEqual(filled.refinement, "interpolated")
        # Lấp xong vẫn phải cảnh báo: một frame ở đây là nội suy, không phải
        # bằng chứng. Không gắn cờ thì dòng này trông y hệt dòng lành.
        self.assertTrue(item.degraded)
        # Frame lấp phải là keyframe CÓ THẬT, không phải số nội suy.
        self.assertIn(filled.frame_idx, {210, 810, 1410})
        self.assertLess(item.steps[0].frame_idx, filled.frame_idx)
        self.assertLess(filled.frame_idx, item.steps[2].frame_idx)

    def test_no_keyframe_in_the_gap_leaves_the_hole_rather_than_inventing_one(self) -> None:
        """Frame bịa vừa chắc chắn mất điểm vừa làm người dùng tin nhầm."""

        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 2)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        gold = next(s for s in sequences if s.video_id == "L01_V001")
        # Không có keyframe nào nằm giữa frame 210 và 360.
        documents = self._documents("L01_V001", [210, 360])
        [item] = to_trake_results([gold], expected_steps=3, documents=documents)
        self.assertEqual(item.missing_steps, [2])
        self.assertTrue(item.degraded)

    def test_filling_is_off_without_documents(self) -> None:
        steps = [[_hit("L01_V001", 1)], [_hit("L02_V009", 5)], [_hit("L01_V001", 9)]]
        sequences = link_event_hits(steps, allow_missing_steps=True, min_covered_steps=2)
        gold = next(s for s in sequences if s.video_id == "L01_V001")
        [item] = to_trake_results([gold], expected_steps=3)
        self.assertEqual(item.missing_steps, [2])
