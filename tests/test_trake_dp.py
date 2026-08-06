"""Quy hoạch động ghép chuỗi TRAKE so với beam search.

Mục đích của DP không phải "tốt hơn beam" mà là **tách bạch hai câu hỏi** vẫn
hay bị gộp:

1. Beam có bỏ sót chuỗi tốt hơn không?  -> giữ nguyên `s_i`, đổi cách tìm.
2. `s_i` có phải tín hiệu sai không?    -> giữ nguyên cách tìm, đổi `s_i`.

Không tách thì sửa cả hai cùng lúc rồi không biết cái nào có tác dụng — đúng
lỗi đã mắc ở CAPTION-ENRICH-01.
"""

from __future__ import annotations

import random
import unittest

from online.domain.models import SearchHit
from online.services.trake.sequence_search import (
    SequenceConfig,
    search_sequences,
    search_sequences_dp,
)


def _hit(frame: int, score: float, video: str = "V1") -> SearchHit:
    return SearchHit(
        candidate_id=f"{video}_S{frame:04d}",
        scene_id=f"{video}_S{frame:04d}",
        video_id=video,
        scene_idx=frame // 100,
        score=score,
        start_frame=frame,
        end_frame_exclusive=frame + 1,
        start_sec=frame / 30.0,
        end_sec=(frame + 1) / 30.0,
        best_frame_idx=frame,
        best_timestamp_sec=frame / 30.0,
    )


NO_MISS = SequenceConfig(allow_missing_steps=False, gap_penalty_per_sec=0.0)


class DpMatchesOrBeatsBeamTests(unittest.TestCase):
    def test_dp_recovers_the_chain_beam_prunes_away(self) -> None:
        """Trường hợp beam CẮT NHẦM.

        Cần ÍT NHẤT ba bước: beam khởi tạo không bị cắt, việc cắt chỉ xảy ra
        sau mỗi lần mở rộng. Với hai bước thì beam luôn tối ưu, nên bản đầu của
        test này có tiền đề sai.

        Beam rộng 1, sau bước 2 nó giữ [10, 20] (điểm 2.0) và vứt [1, 5]
        (điểm 1.5). Nhưng frame 20 chặn mất bước 3 tốt nhất (frame 7), nên
        chuỗi tối ưu thật là 1 -> 5 -> 7 với điểm 2.5.
        """

        step_hits = [
            [_hit(1, 0.5), _hit(10, 1.0)],
            [_hit(5, 1.0), _hit(20, 1.0)],
            [_hit(7, 1.0), _hit(30, 0.1)],
        ]
        narrow = SequenceConfig(
            beam_size=1, allow_missing_steps=False, gap_penalty_per_sec=0.0
        )
        beam = search_sequences("V1", step_hits, narrow)
        dp = search_sequences_dp("V1", step_hits, narrow)

        self.assertEqual(beam[0].frame_ids, [10, 20, 30], "tiền đề của test đã đổi")
        self.assertEqual(dp[0].frame_ids, [1, 5, 7], "DP phải tìm ra chuỗi beam đã cắt")
        self.assertGreater(dp[0].score, beam[0].score)

    def test_dp_is_never_worse_than_beam_on_random_inputs(self) -> None:
        """Tính chất phải luôn đúng: DP tối ưu ĐÚNG hàm mục tiêu mà beam xấp xỉ.

        Chú ý: điều này chỉ nói về HÀM MỤC TIÊU `Σ s_i`, KHÔNG nói gì về
        R-score thật. Nếu `s_i` là proxy tồi thì tối ưu nó mạnh hơn hoàn toàn
        có thể làm điểm thật tệ đi — đó là lý do phải đo, không suy ra.
        """

        rng = random.Random(20260805)
        for _ in range(60):
            steps = rng.randint(2, 4)
            step_hits = [
                [
                    _hit(frame, round(rng.uniform(0.1, 1.0), 3))
                    for frame in sorted(rng.sample(range(1, 200), rng.randint(2, 8)))
                ]
                for _ in range(steps)
            ]
            beam = search_sequences("V1", step_hits, NO_MISS)
            dp = search_sequences_dp("V1", step_hits, NO_MISS)
            if not dp:
                self.assertFalse(beam, "beam tìm ra chuỗi mà DP bỏ sót")
                continue
            self.assertTrue(beam, "DP tìm ra chuỗi mà beam bỏ sót")
            self.assertGreaterEqual(dp[0].score + 1e-9, beam[0].score)

    def test_dp_keeps_the_strict_ordering_gate(self) -> None:
        """Thứ tự là cổng cứng, không phải trọng số mềm."""

        step_hits = [[_hit(50, 1.0)], [_hit(10, 1.0)]]
        self.assertEqual(search_sequences_dp("V1", step_hits, NO_MISS), [])

    def test_dp_never_puts_two_steps_on_the_same_frame(self) -> None:
        step_hits = [[_hit(7, 1.0)], [_hit(7, 1.0)]]
        self.assertEqual(search_sequences_dp("V1", step_hits, NO_MISS), [])

    def test_dp_can_skip_a_step_when_that_scores_better(self) -> None:
        """Bỏ 1/4 bước vẫn được 0.75 điểm; không có chuỗi nào thì được 0."""

        config = SequenceConfig(
            allow_missing_steps=True, missing_step_penalty=0.1, gap_penalty_per_sec=0.0
        )
        # Điểm phải khác nhau, nếu không hai phương án cùng 1.9 và kết quả
        # không xác định — test hoà điểm thì không khoá được gì.
        step_hits = [[_hit(10, 1.0)], [_hit(5, 0.5)], [_hit(90, 1.0)]]
        [result] = search_sequences_dp("V1", step_hits, config)
        # Bước 2 chỉ có ứng viên ở frame 5, đứng TRƯỚC bước 1 -> phải bỏ qua.
        self.assertEqual(result.frame_ids, [10, 90])
        self.assertEqual(result.hits[1], None)

    def test_dp_ignores_only_hits_from_other_videos(self) -> None:
        step_hits = [
            [_hit(10, 1.0, video="V2"), _hit(20, 0.5)],
            [_hit(30, 1.0)],
        ]
        [result] = search_sequences_dp("V1", step_hits, NO_MISS)
        self.assertEqual(result.frame_ids, [20, 30])


if __name__ == "__main__":
    unittest.main()
