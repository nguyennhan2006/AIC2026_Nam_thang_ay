"""online/api/container.py phải chọn đúng nhánh dense theo dữ liệu thật có gì
(PR-13), không theo cờ cấu hình tĩnh:

- Export chưa có embedding thật (fixture demo cũ) -> giữ nguyên
  `lexical_hash_fallback` như trước PR-13 (test_container_flags.py đã phủ).
- Export có embedding thật -> `dense_visual` + `LocalClipTextEncoder`.

Không gọi `.encode()` thật ở đây (cần tải CLIP ~1.7GB, không phù hợp unit
test) — chỉ kiểm tra wiring, đúng cách `LocalClipTextEncoder` lazy-load model
trong `_load()`/`encode()` chứ không phải ở `__init__`.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import LocalClipTextEncoder
from online.api.container import build_container
from online.config import Settings
from tests.test_frame_vector_store import _BASE_SCENE, _embedding_ref


def run(coro):
    return asyncio.run(coro)


def _settings(path: Path, data_root: Path, **overrides) -> Settings:
    base = dict(
        app_name="test", environment="test", log_level="INFO", backend="local",
        metadata_jsonl=path, qdrant_url=None, qdrant_api_key=None,
        qdrant_scene_collection="aic_scenes_v1", qdrant_vector_name="visual",
        embedding_url=None, embedding_api_key=None, request_timeout_sec=10.0,
        candidate_limit=100, rrf_k=60, data_root=data_root, cors_origins=(),
        api_key=None, enable_ocr_fuzzy=False, enable_query_prep=False,
        enable_expansion=False, enable_rules=False, enable_object_search=False,
        enable_action_search=False, enable_color_search=False, enable_event_search=False,
    )
    base.update(overrides)
    return Settings(**base)


class ContainerDenseVisualTests(unittest.TestCase):
    def _write_export(self, tmp: Path) -> Path:
        scene = copy.deepcopy(_BASE_SCENE)
        keyframe = scene["keyframes"][0]
        vector_uri = "processed/embeddings/L01_V001/frame_000150.json"
        keyframe["embedding_refs"] = [_embedding_ref(vector_uri)]
        vector_path = tmp / vector_uri
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        vector_path.write_text(json.dumps([0.1, 0.2, 0.3]), encoding="utf-8")

        scenes_path = tmp / "scenes.jsonl"
        scenes_path.write_text(json.dumps(scene), encoding="utf-8")
        (tmp / "keyframes.jsonl").write_text(json.dumps(keyframe), encoding="utf-8")
        return scenes_path

    def test_real_embedding_export_wires_dense_visual_with_clip_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scenes_path = self._write_export(tmp)
            # Chỉ kiểm tra WIRING, nên vô hiệu hoá warmup: nạp CLIP thật cần
            # ~1.7GB tải về, không phù hợp unit test. Hành vi khi warmup hỏng
            # được khoá riêng ở test dưới.
            with mock.patch.object(LocalClipTextEncoder, "warmup", lambda self: None):
                container = run(build_container(_settings(scenes_path, tmp)))
            dense = next(
                r for r in container.search_service.retrievers if isinstance(r, DenseRetriever)
            )
            self.assertEqual(dense.branch_id, "dense_visual")
            self.assertEqual(dense.backend_kind, "vector")
            self.assertIsInstance(dense.encoder, LocalClipTextEncoder)

    def test_unloadable_visual_model_blocks_startup_instead_of_degrading(self) -> None:
        """Model không nạp được PHẢI làm hỏng khởi động, không được chỉ cảnh báo.

        Từng là lỗi thật: `.env.fpt.local` trỏ `AIC_VISUAL_EMBEDDING_MODEL` vào
        repo HuggingFace `openai/clip-vit-large-patch14`, mà máy này chặn SSL
        tới huggingface.co. `warmup()` nuốt exception, nên server vẫn lên,
        `/capabilities` vẫn quảng cáo `dense_visual`, mọi request vẫn trả 200 —
        chỉ có nhánh dense `failed` âm thầm ở từng request. Số đo thu được khi
        đó trông hoàn toàn hợp lệ nhưng thiếu hẳn một nhánh.

        Điều kiện phải giữ: thông báo lỗi nêu ĐÍCH DANH biến môi trường cần
        sửa, vì đây là lỗi cấu hình chứ không phải lỗi lập trình.
        """

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scenes_path = self._write_export(tmp)

            def explode(self) -> None:
                raise OSError("couldn't connect to 'https://huggingface.co'")

            with mock.patch.object(LocalClipTextEncoder, "warmup", explode):
                with self.assertRaises(ValueError) as ctx:
                    run(build_container(_settings(scenes_path, tmp)))
            self.assertIn("AIC_VISUAL_EMBEDDING_MODEL", str(ctx.exception))

    def test_export_without_embeddings_does_not_require_the_clip_model(self) -> None:
        """Fixture chưa chạy PR-13 vẫn phải khởi động được.

        Fail-fast chỉ áp cho export CÓ embedding thật; không thì `dense_visual`
        đâu có được đăng ký, và bắt máy dev phải tải CLIP cho một export chạy
        `lexical_hash_fallback` là vô nghĩa.
        """

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            scene = copy.deepcopy(_BASE_SCENE)
            scenes_path = tmp / "scenes.jsonl"
            scenes_path.write_text(json.dumps(scene), encoding="utf-8")

            def explode(self) -> None:
                raise AssertionError("không được nạp CLIP khi export chưa có embedding")

            with mock.patch.object(LocalClipTextEncoder, "warmup", explode):
                container = run(build_container(_settings(scenes_path, tmp)))
            dense = next(
                r for r in container.search_service.retrievers if isinstance(r, DenseRetriever)
            )
            self.assertEqual(dense.branch_id, "lexical_hash_fallback")


if __name__ == "__main__":
    unittest.main()
