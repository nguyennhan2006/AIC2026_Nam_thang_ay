"""Nhiều index dense song song — CLIP / SigLIP / Jina.

Trọng tâm là hai chốt an toàn. Cả hai chặn đúng loại lỗi mà hệ này hay dính:
cấu hình sai nhưng mọi thứ VẪN chạy, VẪN ra số, chỉ là số vô nghĩa.
"""

from __future__ import annotations

import unittest

from online.adapters.encoders import LocalClipTextEncoder, LocalTextEncoder, infer_encoder_kind
from online.api.container import _assert_dimension_matches, parse_dense_indexes


class ParseSpecTests(unittest.TestCase):
    def test_name_and_path(self) -> None:
        self.assertEqual(
            parse_dense_indexes("clip_v1:storage/models/clip-vit-large-patch14"),
            [("clip_v1", "storage/models/clip-vit-large-patch14", None)],
        )

    def test_explicit_kind_suffix(self) -> None:
        self.assertEqual(
            parse_dense_indexes("jina_v2:jinaai/jina-clip-v2:jina"),
            [("jina_v2", "jinaai/jina-clip-v2", "jina")],
        )

    def test_windows_path_keeps_its_colon(self) -> None:
        # Tách bừa theo mọi dấu hai chấm sẽ cắt nát `D:/models/...`.
        self.assertEqual(
            parse_dense_indexes("w:D:/models/siglip-base:siglip"),
            [("w", "D:/models/siglip-base", "siglip")],
        )

    def test_multiple_separated_by_comma(self) -> None:
        parsed = parse_dense_indexes("a:path/a, b:path/b:jina")
        self.assertEqual([item[0] for item in parsed], ["a", "b"])
        self.assertEqual(parsed[1][2], "jina")

    def test_malformed_raises_instead_of_guessing(self) -> None:
        with self.assertRaises(ValueError):
            parse_dense_indexes("khong-co-dau-hai-cham")


class EncoderKindTests(unittest.TestCase):
    def test_inferred_from_path(self) -> None:
        self.assertEqual(infer_encoder_kind("jinaai/jina-clip-v2"), "jina")
        self.assertEqual(infer_encoder_kind("google/siglip-base-patch16-224"), "siglip")
        self.assertEqual(infer_encoder_kind("storage/models/clip-vit-large-patch14"), "clip")

    def test_unknown_path_defaults_to_clip(self) -> None:
        self.assertEqual(infer_encoder_kind("some/unknown-model"), "clip")

    def test_explicit_kind_beats_inference(self) -> None:
        # Tên thư mục không nói lên gì thì phải khai được tường minh.
        self.assertEqual(LocalTextEncoder("some/model", kind="jina").kind, "jina")

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            LocalTextEncoder("some/model", kind="khong-ton-tai")

    def test_old_name_is_the_same_class(self) -> None:
        self.assertIs(LocalClipTextEncoder, LocalTextEncoder)


class DimensionGuardTests(unittest.TestCase):
    """Trộn hai model là cosine vô nghĩa — nhưng phép nhân vẫn chạy và vẫn ra
    thứ hạng, nên không có gì tự báo. Phải chặn lúc khởi động."""

    @staticmethod
    def _rows(dim: int) -> list:
        return [("kf", "V", [0.0] * dim, {})]

    def test_matching_dimension_passes(self) -> None:
        _assert_dimension_matches("ok", [0.0] * 768, self._rows(768))

    def test_mismatch_raises_with_both_numbers(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _assert_dimension_matches("lech", [0.0] * 512, self._rows(768))
        message = str(ctx.exception)
        self.assertIn("512", message)
        self.assertIn("768", message)

    def test_empty_rows_is_not_an_error(self) -> None:
        # Index khai nhưng chưa có vector được xử lý ở chỗ khác (thông báo
        # riêng, có liệt kê tên đang có); ở đây không được nổ.
        _assert_dimension_matches("rong", [0.0] * 512, [])


if __name__ == "__main__":
    unittest.main()


class BranchWeightSpecTests(unittest.TestCase):
    """Trọng số mặc định mức triển khai — thay cho việc TẮT hẳn nhánh.

    Nhánh tắt thì không bao giờ cứu được truy vấn mà chỉ nó tìm ra; trọng số
    thấp thì vẫn còn cơ hội. Đo được `R@20 = 1.000` ở mọi trọng số, nên chi phí
    trên chỉ số mục tiêu bằng 0.
    """

    def test_parses_pairs(self) -> None:
        from online.api.container import parse_branch_weights

        self.assertEqual(
            parse_branch_weights("bm25_ocr:1.0,ocr_fuzzy:0.25"),
            {"bm25_ocr": 1.0, "ocr_fuzzy": 0.25},
        )

    def test_empty_spec_means_no_defaults(self) -> None:
        from online.api.container import parse_branch_weights

        self.assertEqual(parse_branch_weights(""), {})

    def test_rejects_weight_outside_schema_range(self) -> None:
        # `BranchRuntimeOptions.weight` ràng [0, 10]; bắt ở đây để lỗi hiện lúc
        # khởi động chứ không phải giữa request đầu tiên.
        from online.api.container import parse_branch_weights

        with self.assertRaises(ValueError):
            parse_branch_weights("bm25_ocr:99")

    def test_rejects_non_numeric(self) -> None:
        from online.api.container import parse_branch_weights

        with self.assertRaises(ValueError):
            parse_branch_weights("bm25_ocr:nhieu")

    def test_rejects_missing_separator(self) -> None:
        from online.api.container import parse_branch_weights

        with self.assertRaises(ValueError):
            parse_branch_weights("bm25_ocr")
