"""PR-04: mọi control trong SearchOptions hoặc chạy thật, hoặc bị từ chối.

Trạng thái trước PR-04: khoảng 20 field được schema chấp nhận nhưng không có
consumer nào. Request đặt `min_score` nhận 200 OK rồi backend bỏ qua — người
dùng tin là đã chỉnh ngưỡng, còn số liệu ablation thì vô nghĩa.
"""

from __future__ import annotations

import unittest

from online.domain.execution import BranchCapabilities
from online.domain.models import (
    Candidate,
    Modality,
    QueryEvent,
    QueryPlan,
    SearchFilters,
    TaskType,
)
from online.domain.search_config import (
    BranchRuntimeOptions,
    FusionOptions,
    QueryProcessingOptions,
    RerankOptions,
    ResultOptions,
    SearchOptions,
    TextRerankOptions,
)
from online.services.capabilities import (
    UnsupportedSearchOptionError,
    validate_search_options,
)
from online.services.score_normalization import normalize_branch
from online.services.thresholding import apply_threshold

CAPABILITIES = [
    BranchCapabilities(
        branch_id="bm25_ocr",
        execution_ids=["bm25_ocr.raw"],
        modality=Modality.OCR,
        backend_kind="lexical",
        supported_controls=["enabled", "weight", "top_k", "timeout_ms"],
    ),
    BranchCapabilities(
        branch_id="ocr_fuzzy",
        execution_ids=["ocr_fuzzy.raw"],
        modality=Modality.OCR,
        backend_kind="fuzzy",
        supported_controls=["enabled", "weight", "top_k", "min_score", "timeout_ms"],
    ),
]


def plan_with(options: SearchOptions) -> QueryPlan:
    return QueryPlan(
        task=TaskType.TEXTUAL_KIS,
        original_query="q",
        normalized_query="q",
        events=[QueryEvent(event_idx=0, text="q")],
        modality_weights={Modality.OCR: 1.0},
        filters=SearchFilters(),
        search_options=options,
    )


def candidates(source: str, scores: list[float], kind: str = "bm25") -> list[Candidate]:
    return [
        Candidate(
            candidate_id=f"L01_V001_S{index:04d}",
            scene_id=f"L01_V001_S{index:04d}",
            video_id="L01_V001",
            source=source,
            modality=Modality.OCR,
            raw_score=score,
            score_kind=kind,
            rank=index + 1,
        )
        for index, score in enumerate(scores)
    ]


class NormalizationTests(unittest.TestCase):
    def test_bm25_becomes_a_percentile_inside_its_own_branch(self) -> None:
        result = normalize_branch(candidates("bm25_ocr.raw", [12.0, 6.0, 1.0]))
        self.assertEqual([item.normalized_score for item in result], [1.0, 0.5, 0.0])
        self.assertTrue(all(0.0 <= item.percentile_score <= 1.0 for item in result))

    def test_cosine_uses_its_known_domain_not_the_result_list(self) -> None:
        # Affine (s+1)/2: không phụ thuộc các candidate khác, nên hai danh sách
        # khác nhau vẫn cho cùng normalized_score cho cùng một cosine.
        one = normalize_branch(candidates("dense.raw", [0.8, 0.2], kind="cosine"))
        two = normalize_branch(candidates("dense.raw", [0.8, -0.6], kind="cosine"))
        self.assertAlmostEqual(one[0].normalized_score, 0.9)
        self.assertAlmostEqual(two[0].normalized_score, 0.9)

    def test_bounded_scores_are_left_alone(self) -> None:
        result = normalize_branch(
            candidates("ocr_fuzzy.raw", [0.91, 0.90], kind="fuzzy_ratio")
        )
        # Min-max sẽ kéo hai giá trị sát nhau thành 1.0 và 0.0 — phá ý nghĩa.
        self.assertAlmostEqual(result[0].normalized_score, 0.91)
        self.assertAlmostEqual(result[1].normalized_score, 0.90)

    def test_ties_share_a_percentile(self) -> None:
        result = normalize_branch(candidates("bm25_ocr.raw", [5.0, 5.0, 1.0]))
        self.assertEqual(result[0].percentile_score, result[1].percentile_score)


class ThresholdTests(unittest.TestCase):
    def _plan(self, **kwargs) -> QueryPlan:
        return plan_with(
            SearchOptions(branches={"ocr_fuzzy.raw": BranchRuntimeOptions(**kwargs)})
        )

    def test_hard_policy_removes_and_reranks(self) -> None:
        items = normalize_branch(
            candidates("ocr_fuzzy.raw", [0.9, 0.5, 0.2], kind="fuzzy_ratio")
        )
        kept, affected = apply_threshold(
            items, self._plan(min_score=0.6, threshold_policy="hard")
        )
        self.assertEqual([item.raw_score for item in kept], [0.9])
        self.assertEqual(affected, 2)
        # Hạng phải được đánh lại: RRF dùng hạng, không đánh lại thì cắt vô nghĩa.
        self.assertEqual([item.rank for item in kept], [1])

    def test_soft_policy_keeps_but_demotes(self) -> None:
        items = normalize_branch(
            candidates("ocr_fuzzy.raw", [0.9, 0.5, 0.2], kind="fuzzy_ratio")
        )
        kept, affected = apply_threshold(
            items, self._plan(min_score=0.6, threshold_policy="soft")
        )
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0].raw_score, 0.9)
        self.assertEqual(affected, 2)
        # Candidate dưới ngưỡng tụt xuống cuối nên đóng góp RRF nhỏ hẳn.
        self.assertEqual([item.rank for item in kept], [1, 2, 3])
        self.assertEqual({item.raw_score for item in kept[1:]}, {0.5, 0.2})

    def test_no_min_score_leaves_the_list_untouched(self) -> None:
        items = normalize_branch(candidates("ocr_fuzzy.raw", [0.9, 0.5]))
        kept, affected = apply_threshold(items, self._plan(weight=2.0))
        self.assertEqual(kept, items)
        self.assertEqual(affected, 0)

    def test_raw_space_thresholds_on_the_untouched_score(self) -> None:
        items = normalize_branch(candidates("bm25_ocr.raw", [12.0, 6.0, 1.0]))
        plan = plan_with(
            SearchOptions(
                branches={
                    "bm25_ocr.raw": BranchRuntimeOptions(
                        min_score=5.0, threshold_space="raw", threshold_policy="hard"
                    )
                }
            )
        )
        kept, _ = apply_threshold(items, plan)
        self.assertEqual([item.raw_score for item in kept], [12.0, 6.0])


class OptionValidationTests(unittest.TestCase):
    def test_defaults_are_always_accepted(self) -> None:
        validate_search_options(SearchOptions(), CAPABILITIES)
        validate_search_options(None, CAPABILITIES)

    def test_unknown_branch_is_rejected_with_the_available_list(self) -> None:
        options = SearchOptions(branches={"clip_search": BranchRuntimeOptions(weight=2.0)})
        with self.assertRaises(UnsupportedSearchOptionError) as ctx:
            validate_search_options(options, CAPABILITIES)
        self.assertIn("clip_search", str(ctx.exception))
        self.assertIn("bm25_ocr", str(ctx.exception))

    def test_control_a_branch_does_not_read_is_rejected(self) -> None:
        # bm25_ocr không khai báo min_score trong supported_controls, nhưng
        # min_score là control toàn cục (ThresholdService xử lý) nên hợp lệ.
        validate_search_options(
            SearchOptions(branches={"bm25_ocr": BranchRuntimeOptions(min_score=0.5)}),
            CAPABILITIES,
        )
        # field_weights thì thật sự không có consumer nào.
        with self.assertRaises(UnsupportedSearchOptionError) as ctx:
            validate_search_options(
                SearchOptions(
                    branches={"bm25_ocr": BranchRuntimeOptions(field_weights={"caption": 2.0})}
                ),
                CAPABILITIES,
            )
        self.assertIn("field_weights", str(ctx.exception))

    def test_branch_level_key_is_accepted_as_well_as_execution_level(self) -> None:
        for key in ("bm25_ocr", "bm25_ocr.raw"):
            with self.subTest(key=key):
                validate_search_options(
                    SearchOptions(branches={key: BranchRuntimeOptions(weight=2.0)}),
                    CAPABILITIES,
                )

    def test_rerank_that_has_no_model_server_is_rejected(self) -> None:
        options = SearchOptions(rerank=RerankOptions(text=TextRerankOptions(enabled=True)))
        with self.assertRaises(UnsupportedSearchOptionError) as ctx:
            validate_search_options(options, CAPABILITIES)
        self.assertIn("BGE", str(ctx.exception))

    def test_hyde_and_translation_flags_are_rejected(self) -> None:
        for options in (
            SearchOptions(query=QueryProcessingOptions(enable_hyde=True)),
            SearchOptions(query=QueryProcessingOptions(generate_english_variant=True)),
        ):
            with self.subTest(options=options):
                with self.assertRaises(UnsupportedSearchOptionError):
                    validate_search_options(options, CAPABILITIES)

    def test_implemented_dedup_scopes_are_accepted(self) -> None:
        # PR-05 cài dedup thật cho các scope này.
        for scope in ("none", "frame", "scene", "event"):
            with self.subTest(scope=scope):
                validate_search_options(
                    SearchOptions(fusion=FusionOptions(dedup_scope=scope)), CAPABILITIES
                )

    def test_visual_near_duplicate_dedup_is_still_rejected(self) -> None:
        with self.assertRaises(UnsupportedSearchOptionError) as ctx:
            validate_search_options(
                SearchOptions(fusion=FusionOptions(dedup_similarity=0.9)), CAPABILITIES
            )
        self.assertIn("dedup_similarity", str(ctx.exception))

    def test_group_by_other_than_none_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedSearchOptionError):
            validate_search_options(
                SearchOptions(results=ResultOptions(group_by="video")), CAPABILITIES
            )
        validate_search_options(
            SearchOptions(results=ResultOptions(group_by="none")), CAPABILITIES
        )

    def test_implemented_controls_pass_through(self) -> None:
        validate_search_options(
            SearchOptions(
                branches={
                    "ocr_fuzzy": BranchRuntimeOptions(
                        weight=3.0, top_k=50, min_score=0.4,
                        threshold_policy="hard", timeout_ms=500,
                    )
                },
                fusion=FusionOptions(method="weighted_sum", rrf_k=30, fusion_top_k=500),
                results=ResultOptions(display_top_k=20, sort_by="time"),
            ),
            CAPABILITIES,
        )


if __name__ == "__main__":
    unittest.main()
