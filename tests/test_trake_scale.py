"""Chế độ hỏng chỉ xuất hiện khi corpus đủ lớn (TRAKE-SCALE-01).

Benchmark 3 video không thể hiện được lỗi này: ở đó mọi step đều có candidate
trong cả 3 video nên giao luôn đầy. Bộ test này dựng một corpus 873 video ngay
trong bộ nhớ — không cần dữ liệu thật — để khoá lại hai tính chất mà mọi phép
đo trước đây đều mù.
"""

from __future__ import annotations

import collections
import random
import unittest

from online.domain.models import SearchHit
from online.services.temporal import link_event_hits
from online.services.trake.video_retriever import VideoRetrieverConfig, rank_videos


def _hit(video: str, scene: int, frame: int, score: float) -> SearchHit:
    return SearchHit(
        candidate_id=f"{video}_S{scene:04d}_{frame}",
        scene_id=f"{video}_S{scene:04d}",
        video_id=video,
        scene_idx=scene,
        start_frame=scene * 150,
        end_frame_exclusive=scene * 150 + 120,
        start_sec=scene * 5.0,
        end_sec=scene * 5.0 + 4.0,
        best_frame_idx=frame,
        best_timestamp_sec=scene * 5.0 + 2.0,
        score=score,
    )


def _corpus(*, videos: int, steps: int, candidate_limit: int, seed: int = 0):
    """Mỗi step lấy top-`candidate_limit` trên TOÀN corpus — đúng như search.py."""

    rng = random.Random(seed)
    pool = [
        (f"L{v // 100:02d}_V{v % 100:03d}", s)
        for v in range(videos)
        for s in range(40)
    ]
    lists = []
    for step in range(steps):
        scored = sorted(
            ((rng.random(), video, scene) for video, scene in pool), reverse=True
        )[:candidate_limit]
        lists.append([
            # Frame khác nhau theo step: `_hydrate` chọn frame theo văn bản của
            # từng step, nên cùng một scene cho ra frame khác nhau ở step khác.
            _hit(video, scene, scene * 150 + 40 + step * 7, 1.0 / (60 + rank))
            for rank, (_base, video, scene) in enumerate(scored)
        ])
    return lists


class CandidateLimitDrivesVideoDiversityTests(unittest.TestCase):
    """K=100 trên 873 video là mức mà TRAKE sụp về 0–1 video."""

    def test_narrow_limit_collapses_to_almost_nothing(self) -> None:
        lists = _corpus(videos=873, steps=3, candidate_limit=100)
        videos = {sequence.video_id for sequence in link_event_hits(lists, limit=20)}
        # Đây là HỎNG, không phải hành vi mong muốn: test ghi lại nó để nếu ai
        # hạ candidate_limit về 100 thì thấy ngay hậu quả, chứ không phải để
        # bảo vệ nó.
        self.assertLessEqual(len(videos), 1)

    def test_wide_limit_restores_diversity(self) -> None:
        lists = _corpus(videos=873, steps=3, candidate_limit=500)
        videos = {sequence.video_id for sequence in link_event_hits(lists, limit=20)}
        self.assertGreaterEqual(len(videos), 5)

    def test_more_steps_shrink_the_intersection_further(self) -> None:
        """Mỗi step thêm vào là một phép nhân xác suất nữa. TRAKE 6 step khó
        hơn hẳn 3 step ở CÙNG một K — lý do vì sao nâng K chỉ hoãn vấn đề."""

        def intersection(steps: int) -> int:
            lists = _corpus(videos=873, steps=steps, candidate_limit=500)
            return len(set.intersection(*[{h.video_id for h in x} for x in lists]))

        self.assertGreater(intersection(3), intersection(6))


class VideoFirstToleratesMissingStepsTests(unittest.TestCase):
    """`rank_videos` (Stage A) KHÔNG đòi video phải có mặt ở mọi step.

    Đây chính là tính chất mà `link_event_hits` thiếu, và là lý do Stage A đáng
    được đo lại trên corpus lớn thay vì bị loại dựa trên benchmark 3 video.
    """

    def test_gold_video_survives_one_missing_step(self) -> None:
        gold = "L21_V007"
        # Gold có bằng chứng ở step 1, 2, 4 — step 3 rơi ngoài top-K toàn corpus.
        step_hits = [
            [_hit(gold, 2, 340, 0.030), _hit("L01_V001", 5, 800, 0.040)],
            [_hit(gold, 6, 940, 0.020), _hit("L01_V001", 9, 1400, 0.045)],
            [_hit("L01_V001", 12, 1900, 0.050)],
            [_hit(gold, 15, 2300, 0.035), _hit("L01_V001", 20, 3100, 0.041)],
        ]
        shortlist = [item.video_id for item in rank_videos(step_hits)]
        self.assertIn(gold, shortlist)

        # Còn linker theo giao thì loại thẳng gold: nó không có step 3.
        chained = {s.video_id for s in link_event_hits(step_hits, limit=20)}
        self.assertNotIn(gold, chained)

    def test_shortlist_size_is_configurable(self) -> None:
        step_hits = [
            [_hit(f"L01_V{v:03d}", 1, 200, 0.04 - v * 0.001) for v in range(12)],
            [_hit(f"L01_V{v:03d}", 5, 800, 0.04 - v * 0.001) for v in range(12)],
        ]
        self.assertEqual(len(rank_videos(step_hits)), 5)  # top_videos mặc định
        wide = rank_videos(step_hits, VideoRetrieverConfig(top_videos=10))
        self.assertEqual(len(wide), 10)


if __name__ == "__main__":
    unittest.main()


class PerVideoQuotaTests(unittest.TestCase):
    """Hạn ngạch video: đổi bản-sao-gần-trùng lấy độ phủ.

    Đo trên 873 video (GAP-RELAX/TRAKE-SCALE): 20 dòng output chỉ chứa 1.58
    video khác nhau, và có query mà gold có mặt ở CẢ 5 step vẫn không lọt được
    vào output vì một video "nam châm" chiếm sạch beam. Với người dùng tự xem
    video để chốt đáp án, `video_recall@20` gần bằng `video_recall@1` nghĩa là
    danh sách dài không cho họ thêm gì.
    """

    def _magnet_corpus(self):
        """Một video mạnh đều ở mọi step + vài video yếu hơn nhưng hợp lệ."""

        magnet = [
            [_hit("L01_MAG", scene, scene * 150 + 40, 0.050 - scene * 0.0001)
             for scene in range(0, 30, 3)]
            for _step in range(3)
        ]
        others = [
            [_hit(f"L02_V{v:03d}", scene, scene * 150 + 40, 0.030)
             for v in range(6) for scene in (1, 9, 17)]
            for _step in range(3)
        ]
        return [m + o for m, o in zip(magnet, others, strict=True)]

    def test_without_quota_one_video_takes_the_whole_list(self) -> None:
        sequences = link_event_hits(self._magnet_corpus(), limit=20)
        self.assertEqual(len({s.video_id for s in sequences}), 1)

    def test_quota_trades_near_duplicates_for_coverage(self) -> None:
        sequences = link_event_hits(
            self._magnet_corpus(), limit=20, per_video_beam=3, max_chains_per_video=2
        )
        videos = {s.video_id for s in sequences}
        self.assertGreaterEqual(len(videos), 5)
        counts = collections.Counter(s.video_id for s in sequences)
        self.assertLessEqual(max(counts.values()), 2)

    def test_the_strongest_video_still_ranks_first(self) -> None:
        """Độ phủ KHÔNG được đánh đổi bằng việc bỏ tụt ứng viên mạnh nhất."""

        sequences = link_event_hits(
            self._magnet_corpus(), limit=20, per_video_beam=3, max_chains_per_video=2
        )
        self.assertEqual(sequences[0].video_id, "L01_MAG")

    def test_output_quota_alone_is_not_enough(self) -> None:
        """Chỉ khử trùng lặp lúc XUẤT thì vô ích: các video khác đã bị đẩy khỏi
        beam từ mấy bước trước. Đây là lý do hạn ngạch phải áp ở cả hai chỗ."""

        only_output = link_event_hits(
            self._magnet_corpus(), limit=20, max_chains_per_video=2
        )
        both = link_event_hits(
            self._magnet_corpus(), limit=20, per_video_beam=3, max_chains_per_video=2
        )
        self.assertLess(
            len({s.video_id for s in only_output}), len({s.video_id for s in both})
        )

    def test_one_row_per_video_is_expressible(self) -> None:
        """`max_chains_per_video=1` = danh sách để NGƯỜI duyệt: mỗi video đúng
        một dòng, không có bản sao gần trùng nào chiếm chỗ.

        Đánh đổi có chủ ý: bỏ các biến thể frame của cùng một video sẽ làm điểm
        frame tự động kém đi, nhưng người dùng tự chỉnh frame nên thứ họ cần là
        ĐỘ PHỦ video, không phải 20 phương án frame của một video.
        """

        sequences = link_event_hits(
            self._magnet_corpus(), limit=20, per_video_beam=5, max_chains_per_video=1
        )
        videos = [s.video_id for s in sequences]
        self.assertEqual(len(videos), len(set(videos)))
        self.assertGreaterEqual(len(videos), 5)

    def test_deeper_scan_actually_reveals_new_videos(self) -> None:
        """Tính chất mà bản hiện tại KHÔNG có: trên 873 video đo được
        `video_recall@1 = 0.750` còn `video_recall@20 = 0.792` — quét sâu hơn
        gần như không lộ ra video nào mới. Với một video một dòng thì mỗi dòng
        thêm vào là một video thật sự mới."""

        sequences = link_event_hits(
            self._magnet_corpus(), limit=20, per_video_beam=5, max_chains_per_video=1
        )
        for depth in (1, 3, 5):
            self.assertEqual(
                len({s.video_id for s in sequences[:depth]}),
                min(depth, len(sequences)),
            )
