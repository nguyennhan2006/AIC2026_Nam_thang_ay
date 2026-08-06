"""EVAL-01: mẫu số chuẩn hoá không được phụ thuộc lát cắt đang hiển thị.

Trước khi có `ScoreNormalizers`, cả ba processor lấy `max(...)` trên chính
danh sách được truyền vào — mà danh sách đó nằm SAU `deduplicate_for_task`,
nên nó phụ thuộc `fusion.max_results_per_video`. Hệ quả: nới cap làm đổi điểm
của những candidate vốn không liên quan gì tới phần mới thêm.
"""

from __future__ import annotations

import unittest

from online.domain.candidate import Candidate
from online.domain.evidence import EvidencePack
from online.services.avs import AvsProcessor
from online.services.normalizers import ScoreNormalizers
from online.services.qa import QaProcessor


def _candidate(candidate_id: str, score: float, branches: int) -> Candidate:
    from online.domain.scores import BranchScore

    return Candidate(
        candidate_id=candidate_id,
        video_id="L21_V001",
        scene_id=candidate_id,
        source="bm25_caption.raw",
        modality="caption",
        raw_score=score,
        rank=1,
        branch_scores={
            f"b{index}": BranchScore(raw_score=score, score_kind="bm25")
            for index in range(branches)
        },
    )


def _fused_candidate(candidate_id: str, score: float, branches: int) -> Candidate:
    """Candidate ĐÚNG HÌNH DẠNG mà `fuse_candidates` sinh ra.

    Khác `_candidate` ở chỗ nó KHÔNG có `branch_scores` — fusion ghi thông tin
    nhánh vào `payload["matched_branches"]`. Fixture cũ giàu hơn dữ liệu thật,
    và chính khoảng cách đó đã giấu lỗi `branch_ceiling` luôn bằng 1 ở đường
    production suốt nhiều đợt đo.
    """

    return Candidate(
        candidate_id=candidate_id,
        video_id="L21_V001",
        scene_id=candidate_id,
        source="bm25_caption.raw",
        modality="caption",
        raw_score=score,
        rank=1,
        payload={"matched_branches": [f"b{index}" for index in range(branches)]},
    )


def _pack(candidate_id: str) -> EvidencePack:
    from online.domain.candidate import FrameEvidence

    return EvidencePack(
        candidate_id=candidate_id, video_id="L21_V001", scene_id=candidate_id,
        start_frame=0, end_frame_exclusive=100, start_sec=0.0, end_sec=4.0,
        best_frame_idx=10,
        keyframes=[FrameEvidence(
            keyframe_id=f"{candidate_id}_F10", video_id="L21_V001", scene_id=candidate_id,
            frame_idx=10, timestamp_sec=1.0, image_path="f.jpg",
        )],
        caption_text="một cảnh quay ngoài trời",
    )


class ScoreNormalizersTests(unittest.TestCase):
    def test_taken_from_full_pool_not_the_slice(self) -> None:
        pool = [_candidate("a", 0.9, 3), _candidate("b", 0.4, 1), _candidate("c", 0.2, 2)]
        self.assertEqual(ScoreNormalizers.from_pool(pool).best_retrieval_score, 0.9)
        self.assertEqual(ScoreNormalizers.from_pool(pool).branch_ceiling, 3)

    def test_ceiling_works_on_candidates_shaped_like_fusion_output(self) -> None:
        """Hình dạng THẬT của production: chỉ có `payload`, không `branch_scores`.

        Bản trước chỉ đọc `branch_scores` nên trả `branch_ceiling = 1` cho mọi
        pool thật, biến `agreement` ở kis.py từ thang 0–1 thành 4.0–8.0 và làm
        nó thành số hạng lớn nhất của công thức chấm KIS.
        """

        pool = [_fused_candidate("a", 0.9, 8), _fused_candidate("b", 0.4, 3)]
        self.assertEqual(ScoreNormalizers.from_pool(pool).branch_ceiling, 8)

    def test_truncating_the_pool_changes_the_normalizer(self) -> None:
        """Chính là cái bẫy: cắt bớt pool làm mẫu số đổi.

        Test này KHOÁ lý do phải tính normalizer trước dedup — nếu ai đó tính
        lại sau khi cắt, giá trị sẽ khác và thứ hạng đã có sẽ bị xáo trộn.
        """

        pool = [_candidate("a", 0.9, 3), _candidate("b", 0.4, 1)]
        self.assertNotEqual(
            ScoreNormalizers.from_pool(pool).best_retrieval_score,
            ScoreNormalizers.from_pool(pool[1:]).best_retrieval_score,
        )

    def test_empty_pool_is_safe(self) -> None:
        normalizers = ScoreNormalizers.from_pool([])
        self.assertEqual(normalizers.best_retrieval_score, 1.0)
        self.assertEqual(normalizers.branch_ceiling, 1)


class ProcessorsHonourNormalizersTests(unittest.TestCase):
    def test_qa_uses_supplied_denominator(self) -> None:
        """Cùng pack + cùng frame_scores, đổi mẫu số phải đổi joint_score."""

        packs = [_pack("p1")]
        scores = {"p1": 0.5}
        low = QaProcessor().answer(
            "cảnh gì?", packs, frame_scores=scores,
            normalizers=ScoreNormalizers(best_retrieval_score=0.5),
        )
        high = QaProcessor().answer(
            "cảnh gì?", packs, frame_scores=scores,
            normalizers=ScoreNormalizers(best_retrieval_score=5.0),
        )
        self.assertTrue(low and high)
        self.assertGreater(low[0].joint_score, high[0].joint_score)

    def test_avs_accepts_normalizers(self) -> None:
        packs = [_pack("p1"), _pack("p2")]
        items = AvsProcessor().rank(
            "cảnh ngoài trời", packs,
            retrieval_scores={"p1": 0.9, "p2": 0.3},
            normalizers=ScoreNormalizers(best_retrieval_score=0.9),
        )
        self.assertTrue(items)


if __name__ == "__main__":
    unittest.main()
