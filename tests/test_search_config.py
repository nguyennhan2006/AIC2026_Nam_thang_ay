"""W0 — Search Mixing Console domain contracts.

Chỉ test contract (import, default, bound) — CHƯA có branch/service nào đọc
`SearchOptions` thật (đó là W3/W5), nên không test end-to-end ở đây.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from online.domain.models import SearchRequest
from online.domain.search_config import (
    BranchRuntimeOptions,
    FusionOptions,
    QueryProcessingOptions,
    RerankOptions,
    ResultOptions,
    SearchOptions,
    TemporalOptions,
    TextRerankOptions,
    VlmRerankOptions,
)


class SearchOptionsDefaultsTests(unittest.TestCase):
    def test_search_options_has_sane_defaults(self) -> None:
        options = SearchOptions()
        self.assertIsNone(options.profile_id)
        self.assertEqual(options.branches, {})
        self.assertEqual(options.fusion.method, "rrf")
        self.assertEqual(options.fusion.rrf_k, 60)
        self.assertTrue(options.rerank.enable_rules)
        self.assertTrue(options.temporal.same_video_required)
        self.assertEqual(options.results.display_top_k, 100)

    def test_branch_runtime_options_defaults(self) -> None:
        branch = BranchRuntimeOptions()
        self.assertTrue(branch.enabled)
        self.assertEqual(branch.weight, 1.0)
        self.assertEqual(branch.top_k, 300)
        self.assertEqual(branch.threshold_space, "normalized")
        self.assertEqual(branch.threshold_policy, "soft")
        self.assertEqual(branch.parameters, {})

    def test_branch_runtime_options_accepts_arbitrary_parameters(self) -> None:
        branch = BranchRuntimeOptions(
            weight=1.4,
            top_k=300,
            min_score=0.72,
            threshold_policy="hard",
            parameters={
                "match_mode": "fuzzy",
                "fuzzy_ratio": 0.8,
                "max_edit_distance": 2,
                "normalize_diacritics": True,
            },
        )
        self.assertEqual(branch.parameters["match_mode"], "fuzzy")
        self.assertEqual(branch.parameters["max_edit_distance"], 2)


class SearchOptionsBoundsTests(unittest.TestCase):
    def test_weight_out_of_bounds_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BranchRuntimeOptions(weight=10.1)
        with self.assertRaises(ValidationError):
            BranchRuntimeOptions(weight=-0.1)

    def test_rrf_k_out_of_bounds_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            FusionOptions(rrf_k=0)
        with self.assertRaises(ValidationError):
            FusionOptions(rrf_k=501)

    def test_unknown_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchOptions.model_validate({"unknown_field": True})

    def test_invalid_literal_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            FusionOptions.model_validate({"method": "not_a_real_method"})


class SearchRequestBackwardCompatTests(unittest.TestCase):
    def test_search_options_defaults_to_none(self) -> None:
        request = SearchRequest(query="test")
        self.assertIsNone(request.search_options)

    def test_existing_request_shape_still_valid_without_search_options(self) -> None:
        # Đúng shape request cũ (trước W0) — phải parse y hệt, không cần search_options.
        request = SearchRequest.model_validate({"query": "test", "top_k": 5})
        self.assertIsNone(request.search_options)
        self.assertEqual(request.top_k, 5)

    def test_search_options_can_be_attached(self) -> None:
        request = SearchRequest(
            query="test",
            search_options=SearchOptions(
                branches={"ocr_fuzzy": BranchRuntimeOptions(weight=1.4, min_score=0.72)}
            ),
        )
        assert request.search_options is not None
        self.assertEqual(request.search_options.branches["ocr_fuzzy"].weight, 1.4)


class SearchOptionsSubModelSmokeTests(unittest.TestCase):
    """Mỗi sub-model import/construct được với default — bắt lỗi typo/thiếu field sớm."""

    def test_all_sub_models_construct_with_defaults(self) -> None:
        for cls in (
            BranchRuntimeOptions,
            QueryProcessingOptions,
            FusionOptions,
            TextRerankOptions,
            VlmRerankOptions,
            RerankOptions,
            TemporalOptions,
            ResultOptions,
            SearchOptions,
        ):
            with self.subTest(cls=cls.__name__):
                cls()  # không raise


if __name__ == "__main__":
    unittest.main()
