"""PR-07: AVS relevance grading + MMR diversity.

`_diversify_avs` (trước PR-07) chỉ giới hạn N kết quả mỗi video — không phân
biệt segment của các sự kiện khác nhau với segment gần trùng nhau của cùng
một sự kiện.
"""

from __future__ import annotations

import unittest

from online.domain.candidate import FrameEvidence
from online.domain.evidence import EvidencePack
from online.services.avs import AvsConfig, AvsProcessor, extract_criteria, jaccard


def pack(candidate_id: str, video: str, text: str, *, start: int = 0) -> EvidencePack:
    frame = FrameEvidence(
        keyframe_id=f"{candidate_id}_F{start:06d}", video_id=video, scene_id=candidate_id,
        frame_idx=start, timestamp_sec=start / 30, image_path="f.jpg", captions=[text],
    )
    return EvidencePack(
        candidate_id=candidate_id, video_id=video, scene_id=candidate_id,
        start_frame=start, end_frame_exclusive=start + 100,
        start_sec=start / 30, end_sec=(start + 100) / 30,
        keyframes=[frame], caption_text=text, best_frame_idx=start,
    )


class CriteriaTests(unittest.TestCase):
    def test_and_of_or_groups_are_extracted(self) -> None:
        criteria = extract_criteria(
            "người lớn và trẻ em trong vườn, đang dạy hoặc tưới cây"
        )
        self.assertEqual(len(criteria.inclusion), 3)
        last_group = criteria.inclusion[-1]
        self.assertEqual(len(last_group), 2)

    def test_negative_clause_becomes_exclusion_not_inclusion(self) -> None:
        criteria = extract_criteria("cảnh có xe máy, không có ô tô")
        self.assertTrue(criteria.exclusion)
        joined_inclusion = " ".join(" ".join(group) for group in criteria.inclusion)
        self.assertNotIn("to", joined_inclusion.split())

    def test_full_match_grades_three(self) -> None:
        criteria = extract_criteria("người lớn và trẻ em")
        self.assertEqual(criteria.grade("người lớn cùng trẻ em chơi trong vườn"), 3)

    def test_partial_match_grades_between_one_and_two(self) -> None:
        criteria = extract_criteria("người lớn và trẻ em và xe đạp")
        grade = criteria.grade("người lớn đang đứng cạnh xe đạp")
        self.assertIn(grade, (1, 2))

    def test_no_match_grades_zero(self) -> None:
        criteria = extract_criteria("người lớn và trẻ em")
        self.assertEqual(criteria.grade("cánh đồng lúa buổi sáng"), 0)

    def test_excluded_text_grades_zero_even_if_inclusion_matches(self) -> None:
        criteria = extract_criteria("cảnh có xe máy, không có ô tô")
        self.assertEqual(criteria.grade("xe máy đậu cạnh ô tô"), 0)


class JaccardTests(unittest.TestCase):
    def test_identical_sets_have_similarity_one(self) -> None:
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets_have_similarity_zero(self) -> None:
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)

    def test_empty_set_has_similarity_zero(self) -> None:
        self.assertEqual(jaccard(set(), {"a"}), 0.0)


class AvsProcessorTests(unittest.TestCase):
    def test_irrelevant_segments_are_dropped(self) -> None:
        packs = [
            pack("s1", "L21_V001", "cứu hộ y tế cấp cứu ban đêm"),
            pack("s2", "L21_V001", "cánh đồng lúa yên bình buổi sáng"),
        ]
        results = AvsProcessor().rank("cứu hộ y tế ban đêm", packs)
        self.assertEqual([item.segment_id for item in results], ["s1"])

    def test_near_duplicate_segments_are_diversified_by_mmr(self) -> None:
        packs = [
            pack("s1", "L21_V001", "cứu hộ y tế cấp cứu ban đêm tại hiện trường"),
            pack("s2", "L21_V001", "cứu hộ y tế cấp cứu ban đêm ở hiện trường"),
            pack("s3", "L21_V002", "lính cứu hỏa dập lửa ban đêm"),
        ]
        results = AvsProcessor().rank(
            "cứu hộ y tế hoặc cứu hỏa ban đêm", packs,
            retrieval_scores={"s1": 0.9, "s2": 0.85, "s3": 0.8},
        )
        # s3 mô tả một sự kiện khác hẳn nên phải lọt vào top dù điểm thấp hơn
        # s2 — nếu không, MMR không thực sự đa dạng hóa.
        self.assertIn("s3", [item.segment_id for item in results[:2]])

    def test_max_per_video_caps_results_from_one_video(self) -> None:
        packs = [
            pack(f"s{i}", "L21_V001", f"cứu hộ y tế sự kiện {i} ban đêm") for i in range(5)
        ]
        results = AvsProcessor(AvsConfig(max_per_video=2)).rank("cứu hộ y tế ban đêm", packs)
        self.assertLessEqual(len(results), 2)

    def test_similar_segments_share_a_cluster_id(self) -> None:
        packs = [
            pack("s1", "L21_V001", "cứu hộ y tế cấp cứu ban đêm tại hiện trường"),
            pack("s2", "L21_V001", "cứu hộ y tế cấp cứu ban đêm ở hiện trường"),
        ]
        results = AvsProcessor().rank("cứu hộ y tế ban đêm", packs)
        cluster_ids = {item.cluster_id for item in results}
        self.assertEqual(len(cluster_ids), 1)

    def test_relevance_grade_is_within_zero_to_three(self) -> None:
        packs = [pack("s1", "L21_V001", "cứu hộ y tế ban đêm")]
        results = AvsProcessor().rank("cứu hộ y tế ban đêm", packs)
        self.assertTrue(all(0 <= item.relevance_grade <= 3 for item in results))

    def test_empty_pack_list_returns_no_results(self) -> None:
        self.assertEqual(AvsProcessor().rank("bất kỳ", []), [])


if __name__ == "__main__":
    unittest.main()
