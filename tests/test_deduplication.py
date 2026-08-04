"""PR-05: dedup thật, có chính sách riêng cho từng task.

Trước PR-05 việc gom chỉ là hệ quả phụ của `fuse_candidates` gom theo
`scene_id`, nên top-100 chứa hàng chục scene liền kề của cùng một sự kiện —
chiếm hết các mốc chấm 1/5/20 mà không thêm thông tin nào.
"""

from __future__ import annotations

import unittest

from online.domain.models import Candidate, Modality
from online.domain.tasks import TaskType
from online.services.deduplication import deduplicate, deduplicate_for_task


def candidate(
    scene: str,
    *,
    video: str = "L21_V001",
    event: str | None = None,
    frame: int | None = None,
    timestamp: float | None = None,
    score: float = 1.0,
    rank: int = 1,
) -> Candidate:
    return Candidate(
        candidate_id=f"{video}_{scene}",
        scene_id=f"{video}_{scene}",
        video_id=video,
        event_id=event,
        frame_idx=frame,
        timestamp_sec=timestamp,
        source="fusion_rrf",
        modality=Modality.VISUAL,
        raw_score=score,
        rank=rank,
    )


class DedupScopeTests(unittest.TestCase):
    def test_event_scope_collapses_scenes_of_one_event(self) -> None:
        items = [
            candidate("S0000", event="E0", score=0.9, rank=1),
            candidate("S0001", event="E0", score=0.8, rank=2),
            candidate("S0002", event="E1", score=0.7, rank=3),
        ]
        result = deduplicate(items, scope="event")
        self.assertEqual([item.scene_id for item in result], ["L21_V001_S0000", "L21_V001_S0002"])
        # Cái bị gộp phải để lại dấu vết, không biến mất không dấu tích.
        self.assertEqual(result[0].payload["absorbed_candidates"], ["L21_V001_S0001"])

    def test_candidates_without_event_are_not_lumped_together(self) -> None:
        # Lỗi kinh điển: mọi candidate thiếu event_id bị gom vào khóa None và
        # chỉ còn lại đúng một kết quả.
        items = [candidate(f"S{index:04d}", score=1.0 - index / 10, rank=index + 1) for index in range(4)]
        result = deduplicate(items, scope="event")
        self.assertEqual(len(result), 4)

    def test_frame_scope_keeps_one_candidate_per_frame(self) -> None:
        items = [
            candidate("S0000", frame=100, score=0.9, rank=1),
            candidate("S0001", frame=100, score=0.8, rank=2),
            candidate("S0002", frame=200, score=0.7, rank=3),
        ]
        result = deduplicate(items, scope="frame")
        self.assertEqual([item.frame_idx for item in result], [100, 200])

    def test_video_window_merges_nearby_moments(self) -> None:
        items = [
            candidate("S0000", timestamp=10.0, score=0.9, rank=1),
            candidate("S0001", timestamp=12.0, score=0.8, rank=2),
            candidate("S0002", timestamp=40.0, score=0.7, rank=3),
        ]
        result = deduplicate(items, scope="video_window", window_sec=15.0)
        self.assertEqual(len(result), 2)

    def test_scope_none_is_a_passthrough(self) -> None:
        items = [candidate("S0000", event="E0"), candidate("S0001", event="E0")]
        self.assertEqual(deduplicate(items, scope="none"), items)

    def test_max_per_video_keeps_alternatives_from_other_videos(self) -> None:
        items = [
            candidate("S0000", video="L21_V001", score=0.9, rank=1),
            candidate("S0001", video="L21_V001", score=0.8, rank=2),
            candidate("S0002", video="L21_V001", score=0.7, rank=3),
            candidate("S0000", video="L21_V002", score=0.6, rank=4),
        ]
        result = deduplicate(items, scope="scene", max_per_video=2)
        self.assertEqual(len(result), 3)
        self.assertEqual([item.video_id for item in result].count("L21_V001"), 2)
        self.assertIn("L21_V002", [item.video_id for item in result])

    def test_ranks_are_renumbered_contiguously(self) -> None:
        items = [
            candidate("S0000", event="E0", rank=1),
            candidate("S0001", event="E0", rank=2),
            candidate("S0002", event="E1", rank=3),
        ]
        result = deduplicate(items, scope="event")
        self.assertEqual([item.rank for item in result], [1, 2])

    def test_input_order_is_preserved(self) -> None:
        # Dedup không được sắp lại: thứ tự do fusion/rerank tạo ra là kết quả
        # của cả pipeline phía trước.
        items = [
            candidate("S0002", event="E2", score=0.9, rank=1),
            candidate("S0000", event="E0", score=0.5, rank=2),
        ]
        result = deduplicate(items, scope="event")
        self.assertEqual([item.scene_id for item in result], ["L21_V001_S0002", "L21_V001_S0000"])


class TaskPolicyTests(unittest.TestCase):
    def _event_heavy(self) -> list[Candidate]:
        return [
            candidate(f"S{index:04d}", event="E0", score=1.0 - index / 10, rank=index + 1)
            for index in range(5)
        ]

    def test_kis_collapses_an_event_to_one_result(self) -> None:
        result = deduplicate_for_task(self._event_heavy(), TaskType.TEXTUAL_KIS)
        self.assertEqual(len(result), 1)

    def test_qa_keeps_several_evidence_frames_of_one_event(self) -> None:
        # QA cần nhiều bằng chứng trong cùng event; siết như KIS sẽ vứt mất
        # frame chứa câu trả lời.
        result = deduplicate_for_task(self._event_heavy(), TaskType.QA)
        self.assertEqual(len(result), 3)

    def test_trake_does_not_dedup_at_all(self) -> None:
        result = deduplicate_for_task(self._event_heavy(), TaskType.TRAKE)
        self.assertEqual(len(result), 5)

    def test_avs_caps_results_per_video(self) -> None:
        items = [
            candidate(f"S{index:04d}", video=f"L21_V{index // 4 + 1:03d}",
                      event=f"E{index}", score=1.0 - index / 20, rank=index + 1)
            for index in range(8)
        ]
        result = deduplicate_for_task(items, TaskType.AVS)
        counts: dict[str, int] = {}
        for item in result:
            counts[item.video_id] = counts.get(item.video_id, 0) + 1
        self.assertTrue(all(value <= 3 for value in counts.values()))

    def test_request_can_override_the_task_policy(self) -> None:
        items = self._event_heavy()
        self.assertEqual(len(deduplicate_for_task(items, TaskType.TEXTUAL_KIS)), 1)
        self.assertEqual(
            len(deduplicate_for_task(items, TaskType.TEXTUAL_KIS, scope_override="none")), 5
        )


if __name__ == "__main__":
    unittest.main()
