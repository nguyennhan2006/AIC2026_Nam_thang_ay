"""Fusion họ ĐỌC ĐIỂM THẬT — Phase D, docs/31.

Bài test cốt lõi là `test_rrf_barely_separates_tail_from_head`: nó khoá lại
CƠ CHẾ thật đằng sau nhóm method này.

Tôi ban đầu giải thích sai — tưởng lợi ích đến từ "nhánh chắc chắn được thắng
nhánh đoán mò". Test đầu tiên tôi viết theo giả thuyết đó THẤT BẠI, và nó đúng:
một nhánh chắc chắn với sáu nhánh yếu đồng thuận vẫn thua, ở cả `norm_sum` lẫn
`norm_max`. Cơ chế thật là **đập đuôi**, xem dưới.
"""

from __future__ import annotations

import unittest

from online.domain.models import Candidate, Modality
from online.services.fusion import _confidence, _margin, _minmax, fuse_candidates

RRF_K = 60


def _hit(source: str, modality: Modality, scene: str, score: float, rank: int) -> Candidate:
    return Candidate(
        candidate_id=scene, entity_type="scene", scene_id=scene, video_id="V",
        source=source, modality=modality, raw_score=score, score_kind="bm25", rank=rank,
    )


class NormalizerTests(unittest.TestCase):
    def test_minmax_maps_to_unit_range(self) -> None:
        self.assertEqual(_minmax([27.06, 6.5, 6.0])[0], 1.0)
        self.assertEqual(_minmax([27.06, 6.5, 6.0])[-1], 0.0)

    def test_minmax_all_equal_keeps_full_vote(self) -> None:
        # Branch không phân biệt được gì vẫn bỏ phiếu "tất cả đều liên quan";
        # trả 0.0 sẽ là xoá phiếu của nó chứ không phải trung lập.
        self.assertEqual(_minmax([5.0, 5.0, 5.0]), [1.0, 1.0, 1.0])

    def test_confidence_separates_peaked_from_flat(self) -> None:
        self.assertGreater(_confidence([27.06, 6.5, 6.0]), _confidence([0.51, 0.50, 0.49]))
        self.assertAlmostEqual(_confidence([1.0] * 20), 0.0, places=6)

    def test_margin_is_relative_not_absolute(self) -> None:
        self.assertGreater(_margin([27.06, 6.5]), _margin([0.51, 0.50]))


class TailSuppressionTests(unittest.TestCase):
    """CƠ CHẾ THẬT: RRF cho đuôi của mỗi nhánh gần bằng đỉnh của chính nó."""

    def test_rrf_barely_separates_tail_from_head(self) -> None:
        head, tail = 1.0 / (RRF_K + 1), 1.0 / (RRF_K + 100)
        # Candidate hạng 100 vẫn được 38% số phiếu của hạng 1. Bảy nhánh x 100
        # candidate = 700 lá phiếu gần bằng nhau, và tín hiệu thật chìm trong đó.
        self.assertGreater(tail / head, 0.35)

    def test_normalization_crushes_the_tail(self) -> None:
        scores = [100.0 - index for index in range(100)]
        normalized = _minmax(scores)
        self.assertEqual(normalized[0], 1.0)
        # Cùng vị trí hạng 100 giờ còn ~0 thay vì 38%.
        self.assertLess(normalized[-1], 0.01)

    def test_tail_votes_cannot_outrank_a_head_vote(self) -> None:
        """Tính chất THẬT: phiếu ở đuôi không lật được phiếu ở đỉnh.

        Không phải "nhánh chắc chắn thắng đồng thuận" — tôi đã thử dựng tình
        huống đó hai lần và cả hai lần `norm_max` vẫn chọn phía đồng thuận, ĐÚNG
        như nó nên làm: ba nhánh cùng xếp một candidate hạng 1 thì đó là đồng
        thuận thật.
        """

        target = [_hit("b0.raw", Modality.OCR, "S_target", 27.0, 1)]
        # Ba nhánh nhiễu: mỗi nhánh có candidate riêng ở đỉnh, còn `S_tail` nằm
        # tít hạng 30. Dưới RRF, ba lá phiếu hạng-30 cộng lại vẫn thắng một lá
        # phiếu hạng-1; sau chuẩn hoá thì không.
        noisy = []
        for branch in range(1, 4):
            rows = [
                _hit(f"b{branch}.raw", Modality.CAPTION, f"S_own{branch}_{index}",
                     1.0 - index * 0.01, index + 1)
                for index in range(30)
            ]
            rows.append(_hit(f"b{branch}.raw", Modality.CAPTION, "S_tail", 0.70, 31))
            noisy.append(rows)

        weights = {Modality.OCR: 1.0, Modality.CAPTION: 1.0}
        rrf_scores = {
            item.scene_id: item.raw_score
            for item in fuse_candidates([target, *noisy], weights, method="rrf", limit=200)
        }
        self.assertGreater(rrf_scores["S_tail"], rrf_scores["S_target"])

        norm_scores = {
            item.scene_id: item.raw_score
            for item in fuse_candidates([target, *noisy], weights, method="norm_max", limit=200)
        }
        self.assertGreater(norm_scores["S_target"], norm_scores["S_tail"])


class DeterminismTests(unittest.TestCase):
    def test_score_methods_stay_deterministic(self) -> None:
        lists = [
            [_hit(f"b{b}.raw", Modality.CAPTION, f"S{i}", 1.0 - i * 0.1, i + 1) for i in range(5)]
            for b in range(3)
        ]
        weights = {Modality.CAPTION: 1.0}
        for method in ("norm_sum", "norm_max", "margin_sum", "entropy_sum"):
            first = [x.scene_id for x in fuse_candidates(lists, weights, method=method, limit=5)]
            second = [x.scene_id for x in fuse_candidates(lists, weights, method=method, limit=5)]
            self.assertEqual(first, second, method)


if __name__ == "__main__":
    unittest.main()
