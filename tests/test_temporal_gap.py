"""Phạt khoảng cách thời gian: dead-zone + tuyến tính + trần.

Ràng buộc thời gian ở TRAKE là **tín hiệu mềm**, không phải luật. Luật chỉ đòi
đúng video, đúng thứ tự, đúng ngữ nghĩa — không điều khoản nào thưởng cho chuỗi
gọn về thời gian. Bộ test này khoá ba tính chất khiến nó ở đúng vai trò đó:

1. Chuỗi có nhịp GIỐNG GOLD (gap p50=10s, p100=36s) không mất một điểm nào.
2. Phạt có TRẦN — nhưng trần ĐỦ CAO. GAP-RELAX-01 đo được trần thấp (0.01, tức
   1/4 điểm một scene) làm `video_recall@1` rơi 1.000 -> 0.958, tệ hơn cả tắt
   hẳn phạt: ở đây phạt khoảng cách không phải cái phá hoà, nó là thứ dìm chuỗi
   trải rộng ở video SAI xuống. Test khoá lại thẩm quyền đó.
3. beam và dp tính CÙNG một hàm mục tiêu — nếu không, mọi so sánh giữa hai
   solver đều là so hai bài toán khác nhau.
"""

from __future__ import annotations

import unittest

from online.domain.models import SearchHit
from online.services.temporal import link_event_hits
from online.services.temporal_dp import link_event_hits_dp
from online.services.temporal_gap import (
    DEFAULT_FREE_GAP_SEC,
    DEFAULT_MAX_GAP_PENALTY,
    gap_penalty_ceiling_sec,
    gap_penalty_value,
)


def _hit(
    scene: int,
    score: float,
    *,
    video: str = "L01_V001",
    start_sec: float | None = None,
    frame: int | None = None,
) -> SearchHit:
    """Mỗi scene dài 4s, cách nhau 5s trừ khi `start_sec` nói khác."""

    begin = scene * 5.0 if start_sec is None else start_sec
    best_frame = scene * 150 + 60 if frame is None else frame
    scene_id = f"{video}_S{scene:04d}_F{best_frame:06d}"
    return SearchHit(
        candidate_id=scene_id,
        scene_id=scene_id,
        video_id=video,
        scene_idx=scene,
        start_frame=int(begin * 30),
        end_frame_exclusive=int(begin * 30) + 120,
        start_sec=begin,
        end_sec=begin + 4.0,
        best_frame_idx=best_frame,
        best_timestamp_sec=begin + 2.0,
        score=score,
    )


class GapPenaltyShapeTests(unittest.TestCase):
    def test_gap_inside_the_free_window_costs_nothing(self) -> None:
        # p10=5s, p50=10s, p90=21s, max=36s trên gold TRAKE của corpus này.
        for gap in (0.0, 5.0, 10.0, 21.0, 36.0, DEFAULT_FREE_GAP_SEC):
            self.assertEqual(gap_penalty_value(gap), 0.0, f"gap={gap}")

    def test_negative_gap_is_not_a_bonus(self) -> None:
        """Hai bước chồng lấn (cùng scene) không được ĂN ĐIỂM vì gap âm."""

        self.assertEqual(gap_penalty_value(-30.0), 0.0)

    def test_penalty_is_capped_no_matter_how_far(self) -> None:
        ceiling = gap_penalty_ceiling_sec()
        self.assertAlmostEqual(gap_penalty_value(ceiling), DEFAULT_MAX_GAP_PENALTY)
        for gap in (ceiling, ceiling * 2, 36000.0):
            self.assertLessEqual(gap_penalty_value(gap), DEFAULT_MAX_GAP_PENALTY)
        self.assertEqual(gap_penalty_value(36000.0), gap_penalty_value(ceiling))

    def test_cap_is_high_enough_to_reject_a_sprawling_chain(self) -> None:
        """Ngưỡng gãy đo được nằm giữa cap=0.3 (recall@1 0.958) và cap=0.5
        (1.000). Trần phải nằm TRÊN 0.5 với biên, nếu không ta lặp lại đúng
        hồi quy của bản đầu — nơi cap=0.01 tệ hơn cả không phạt gì."""

        self.assertGreaterEqual(DEFAULT_MAX_GAP_PENALTY, 0.5)

    def test_zero_lambda_disables_the_penalty_entirely(self) -> None:
        self.assertEqual(gap_penalty_value(99999.0, penalty_per_sec=0.0), 0.0)


class HardConstraintTests(unittest.TestCase):
    def test_cross_video_chains_are_still_rejected(self) -> None:
        first = [_hit(1, 1.0, video="L01_V001")]
        second = [_hit(2, 1.0, video="L01_V002")]
        self.assertEqual(link_event_hits([first, second]), [])
        self.assertEqual(link_event_hits_dp([first, second]), [])

    def test_out_of_order_chains_are_still_rejected(self) -> None:
        self.assertEqual(link_event_hits([[_hit(2, 1.0)], [_hit(1, 1.0)]]), [])
        self.assertEqual(link_event_hits_dp([[_hit(2, 1.0)], [_hit(1, 1.0)]]), [])

    def test_two_steps_inside_one_scene_are_allowed_when_frames_advance(self) -> None:
        """Bản trước đòi `scene_idx` tăng, tức vứt cả chuỗi này. Nhưng scene p50
        chỉ dài ~4s và người ra đề không cắt scene theo bước của họ."""

        early = _hit(3, 1.0, frame=450)
        late = _hit(3, 0.9, frame=470)
        [sequence] = link_event_hits([[early], [late]])
        self.assertEqual([scene.best_frame_idx for scene in sequence.scenes], [450, 470])

    def test_same_scene_same_frame_is_rejected(self) -> None:
        """Thứ tự phải tăng NGHIÊM NGẶT: hai bước trùng frame thì submission
        không phân biệt được chúng."""

        twin = _hit(3, 1.0, frame=450)
        self.assertEqual(link_event_hits([[twin], [twin]]), [])


class PenaltyDoesNotOutrankRelevanceTests(unittest.TestCase):
    def test_gold_paced_chain_keeps_its_full_score(self) -> None:
        chain = [[_hit(0, 0.04, start_sec=0.0)], [_hit(1, 0.04, start_sec=14.0)]]
        [sequence] = link_event_hits(chain)
        self.assertAlmostEqual(sequence.score, 0.08)

    def test_a_step_just_outside_the_window_is_not_thrown_away(self) -> None:
        """Đây mới là chỗ dead-zone làm việc. Bước cách 45s (trong vùng miễn
        phạt) không bị trừ gì, nên một ứng viên khớp hơn 0.02 điểm ở đó thắng
        được ứng viên sát nhau nhưng khớp kém. Bản cũ trừ nó 0.09 và nó thua."""

        anchor = _hit(0, 0.04, start_sec=0.0)
        tight_but_weak = _hit(1, 0.02, start_sec=10.0)
        spaced_but_strong = _hit(2, 0.04, start_sec=45.0)
        [best, *_rest] = link_event_hits([[anchor], [tight_but_weak, spaced_but_strong]])
        self.assertEqual(best.scenes[1].start_sec, 45.0)

    def test_a_chain_sprawling_across_half_an_hour_is_crushed(self) -> None:
        """Mặt còn lại của cùng một chính sách, và là mặt ĐO ĐƯỢC là quan trọng:
        hai bước cách nhau 30 phút gần như chắc chắn không cùng một diễn biến,
        nên chuỗi đó phải thua kể cả khi step của nó khớp hơn hẳn."""

        anchor = _hit(0, 0.04, start_sec=0.0)
        near_but_weak = _hit(1, 0.01, start_sec=20.0)
        far_but_strong = _hit(2, 0.04, start_sec=1800.0)
        [best, *_rest] = link_event_hits([[anchor], [near_but_weak, far_but_strong]])
        self.assertEqual(best.scenes[1].start_sec, 20.0)

    def test_a_far_chain_still_loses_to_an_equally_good_near_chain(self) -> None:
        """Phạt nhỏ nhưng không phải không có: khi độ liên quan HOÀ, nhịp thời
        gian mới được quyền phá hoà."""

        anchor = _hit(0, 0.04, start_sec=0.0)
        near = _hit(1, 0.04, start_sec=10.0)
        far = _hit(2, 0.04, start_sec=1800.0)
        [best, *_rest] = link_event_hits([[anchor], [near, far]])
        self.assertEqual(best.scenes[1].start_sec, 10.0)


class BeamAndDpShareOneObjectiveTests(unittest.TestCase):
    def test_both_solvers_score_the_same_chain_identically(self) -> None:
        steps = [
            [_hit(0, 0.05, start_sec=0.0)],
            [_hit(1, 0.04, start_sec=12.0)],
            [_hit(2, 0.03, start_sec=900.0)],
        ]
        [beam] = link_event_hits(steps)
        [dp] = link_event_hits_dp(steps)
        self.assertEqual(
            [scene.scene_id for scene in beam.scenes],
            [scene.scene_id for scene in dp.scenes],
        )
        self.assertAlmostEqual(beam.score, dp.score)

    def test_dp_defaults_are_no_longer_penalty_free(self) -> None:
        """Trước đây `search.py` ép `gap_penalty=0.0` cho nhánh dp, nên hai
        solver âm thầm giải hai bài toán khác nhau."""

        steps = [
            [_hit(0, 0.04, start_sec=0.0)],
            [_hit(1, 0.04, start_sec=10.0), _hit(2, 0.04, start_sec=1800.0)],
        ]
        [best, *_rest] = link_event_hits_dp(steps)
        self.assertEqual(best.scenes[1].start_sec, 10.0)


if __name__ == "__main__":
    unittest.main()
