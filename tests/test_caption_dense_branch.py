"""Nhánh dense trên TEXT caption (DENSE-TEXT-01) — hành vi + ba chốt an toàn.

Trọng tâm là các chốt, không phải chất lượng model: cả ba ca bị chặn dưới đây
đều **vẫn chạy được** nếu để lọt — nhánh trả đủ candidate, `branch_status` báo
`success`, chỉ có kết quả là vô nghĩa. Đúng loại lỗi đã cắn nhiều lần trong repo
này (`_boxes` làm chết `bm25_ocr` trên 765 scene trong im lặng).

Test KHÔNG cần torch: encoder chỉ là một object có `.encode(list[str])`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from online.adapters.dense_text import CaptionDenseRetriever, build_document_text
from online.domain.models import Modality, QueryEvent, QueryPlan, SearchFilters
from online.domain.search_config import BranchRuntimeOptions, SearchOptions
from online.domain.tasks import TaskType


def run(coro):
    return asyncio.run(coro)


class FakeEncoder:
    """Trả vector cố định theo từ khoá — đủ để kiểm thứ hạng, không cần model."""

    def __init__(self, table: dict[str, list[float]], dim: int = 3) -> None:
        self.table = table
        self.dim = dim
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        rows = []
        for text in texts:
            for key, vector in self.table.items():
                if key in text:
                    rows.append(vector)
                    break
            else:
                rows.append([0.0] * self.dim)
        return np.asarray(rows, dtype="float32")


def write_index(directory: Path, scene_ids: list[str], matrix: np.ndarray, **manifest) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", matrix.astype("float32"))
    (directory / "scene_ids.json").write_text(json.dumps(scene_ids), encoding="utf-8")
    payload = {
        "model_id": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "index_fingerprint": "deadbeef",
        "metadata_source": "storage/exports_multivideo/scenes.jsonl",
    }
    payload.update(manifest)
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def plan_for(query: str = "đàn cá quẫy", options: SearchOptions | None = None) -> QueryPlan:
    return QueryPlan(
        task=TaskType.TEXTUAL_KIS,
        original_query=query,
        normalized_query=query,
        events=[QueryEvent(event_idx=0, text=query)],
        modality_weights={Modality.CAPTION: 1.0, Modality.VISUAL: 1.0},
        filters=SearchFilters(),
        search_options=options or SearchOptions(),
    )


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.index = write_index(
            Path(self.tmp.name) / "caption_dense",
            ["L21_V001_S0001", "L21_V001_S0002"],
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        self.addCleanup(self.tmp.cleanup)

    def build(self, table=None) -> CaptionDenseRetriever:
        encoder = FakeEncoder(table or {"cá": [0.0, 1.0, 0.0]})
        return CaptionDenseRetriever(self.index, encoder)

    def test_ranks_by_cosine_and_tags_the_branch(self) -> None:
        retriever = self.build()
        hits = run(retriever.search(plan_for(), limit=10))
        self.assertEqual([hit.scene_id for hit in hits], ["L21_V001_S0002", "L21_V001_S0001"])
        self.assertEqual(hits[0].source, "caption_dense.raw")
        self.assertEqual(hits[0].modality, Modality.CAPTION)
        self.assertEqual(hits[0].video_id, "L21_V001")
        self.assertAlmostEqual(hits[0].raw_score, 1.0, places=5)

    def test_query_prefix_from_manifest_is_applied(self) -> None:
        # E5 bất đối xứng: thiếu `query: ` là lỗi IM LẶNG, model vẫn trả số.
        retriever = self.build()
        run(retriever.search(plan_for("đàn cá quẫy"), limit=1))
        self.assertEqual(retriever.encoder.seen, ["query: đàn cá quẫy"])

    def test_single_event_plan_uses_event_text(self) -> None:
        # TRAKE dựng lại plan cho TỪNG bước với events=[event]. Đọc
        # `normalized_query` ở đây sẽ làm mọi bước dùng chung một truy vấn.
        plan = plan_for("cả câu").model_copy(
            update={"events": [QueryEvent(event_idx=0, text="bước một")]}
        )
        retriever = self.build()
        run(retriever.search(plan, limit=1))
        self.assertEqual(retriever.encoder.seen, ["query: bước một"])

    def test_zero_weight_skips_the_branch_entirely(self) -> None:
        options = SearchOptions(branches={"caption_dense": BranchRuntimeOptions(weight=0.0)})
        retriever = self.build()
        self.assertEqual(run(retriever.search(plan_for(options=options), limit=10)), [])
        self.assertEqual(retriever.encoder.seen, [])  # không tốn một lần encode nào

    def test_corrupt_index_raises_instead_of_zipping_short(self) -> None:
        broken = write_index(
            Path(self.tmp.name) / "broken", ["only_one"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        with self.assertRaises(ValueError):
            CaptionDenseRetriever(broken, FakeEncoder({}))

    def test_missing_manifest_names_the_builder_script(self) -> None:
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as ctx:
            CaptionDenseRetriever(empty, FakeEncoder({}))
        self.assertIn("build_caption_dense_index", str(ctx.exception))


class CoverageGuardTests(unittest.TestCase):
    """Chốt quan trọng nhất: index dựng cho export KHÁC với corpus đang phục vụ.

    Index là thư mục rời, không nằm trong export — đổi `AIC_METADATA_JSONL` mà
    quên dựng lại thì nhánh vẫn chạy và không bao giờ đề xuất nổi scene của
    video không có trong index.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = write_index(
            Path(self.tmp.name) / "l21_only",
            ["L21_V001_S0001", "L21_V001_S0002"],
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            metadata_source="storage/exports_l21/scenes.jsonl",
        )
        self.retriever = CaptionDenseRetriever(self.index, FakeEncoder({}))

    def test_matching_corpus_passes(self) -> None:
        self.retriever.assert_covers(["L21_V001_S0001", "L21_V001_S0002"])

    def test_index_from_another_export_is_blocked(self) -> None:
        corpus = ["L21_V001_S0001", "L21_V001_S0002", "L21_V002_S0001", "L21_V003_S0001"]
        with self.assertRaises(ValueError) as ctx:
            self.retriever.assert_covers(corpus)
        message = str(ctx.exception)
        self.assertIn("2/4", message)
        self.assertIn("exports_l21", message)      # index này dựng từ đâu
        self.assertIn("L21_V002", message)         # video nào đang thiếu
        self.assertIn("build_caption_dense_index", message)  # sửa bằng cách nào

    def test_small_gap_is_tolerated(self) -> None:
        # Scene không có caption/tag bị bỏ khỏi index là chuyện bình thường
        # (đo thật: 764/765 scene có caption), không phải lỗi cấu hình.
        big = write_index(
            Path(self.tmp.name) / "almost",
            [f"L21_V001_S{index:04d}" for index in range(99)],
            np.zeros((99, 3)),
        )
        CaptionDenseRetriever(big, FakeEncoder({})).assert_covers(
            [f"L21_V001_S{index:04d}" for index in range(100)]
        )

    def test_empty_corpus_is_not_an_error(self) -> None:
        self.retriever.assert_covers([])


class DimensionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.retriever = CaptionDenseRetriever(
            write_index(Path(self.tmp.name) / "idx", ["L21_V001_S0001"], np.zeros((1, 1024))),
            FakeEncoder({}),
        )

    def test_matching_dimension_passes(self) -> None:
        self.retriever.assert_dimension(np.zeros(1024, dtype="float32"))

    def test_wrong_model_is_blocked(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.retriever.assert_dimension(np.zeros(768, dtype="float32"))
        self.assertIn("768", str(ctx.exception))
        self.assertIn("1024", str(ctx.exception))


class EncoderKindGuardTests(unittest.TestCase):
    """Ca mà `assert_dimension` MÙ: e5 và jina_v3 cùng ra 1024 chiều.

    Đổi encoder mà quên dựng lại index thì chiều vẫn khớp, nhánh vẫn trả đủ
    candidate, `branch_status` vẫn `success` — và mọi điểm cosine đều là rác vì
    hai model không chung không gian embedding. `assert_encoder_kind` là chốt
    DUY NHẤT trong hệ bắt được ca này.
    """

    class Encoder:
        def __init__(self, kind: str) -> None:
            self.kind = kind

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, **manifest) -> CaptionDenseRetriever:
        return CaptionDenseRetriever(
            write_index(
                Path(self.tmp.name) / "idx", ["L21_V001_S0001"], np.zeros((1, 1024)), **manifest
            ),
            FakeEncoder({}),
        )

    def test_matching_kind_passes(self) -> None:
        retriever = self.build(encoder_kind="jina_v3")
        retriever.assert_encoder_kind(self.Encoder("jina_v3"))

    def test_mismatched_kind_is_blocked_despite_equal_dimension(self) -> None:
        retriever = self.build(encoder_kind="e5")
        # Chiều khớp hoàn toàn — chốt cũ cho qua sạch.
        retriever.assert_dimension(np.zeros(1024, dtype="float32"))
        with self.assertRaises(ValueError) as ctx:
            retriever.assert_encoder_kind(self.Encoder("jina_v3"))
        message = str(ctx.exception)
        self.assertIn("e5", message)
        self.assertIn("jina_v3", message)
        self.assertIn("AIC_CAPTION_DENSE_ENCODER", message)

    def test_manifest_without_kind_is_treated_as_e5(self) -> None:
        # Mọi index dựng trước khi có jina đều là E5 và không ghi field này.
        retriever = self.build()
        retriever.assert_encoder_kind(self.Encoder("e5"))
        with self.assertRaises(ValueError):
            retriever.assert_encoder_kind(self.Encoder("jina_v3"))


class DocumentTextTests(unittest.TestCase):
    class Scene:
        def __init__(self, **kwargs) -> None:
            self.captions = kwargs.get("captions", [])
            self.object_labels = kwargs.get("object_labels", [])
            self.action_tags = kwargs.get("action_tags", [])
            self.keywords = kwargs.get("keywords", [])

    def test_caption_first_then_tags(self) -> None:
        text = build_document_text(
            self.Scene(captions=["đàn cá quẫy"], object_labels=["cá", "nước"], keywords=["sông"])
        )
        self.assertEqual(text, "đàn cá quẫy | cá, nước | sông")

    def test_duplicate_labels_collapse(self) -> None:
        text = build_document_text(self.Scene(object_labels=["cá", "cá", "nước"]))
        self.assertEqual(text, "cá, nước")

    def test_scene_without_text_is_empty(self) -> None:
        self.assertEqual(build_document_text(self.Scene()), "")


if __name__ == "__main__":
    unittest.main()
