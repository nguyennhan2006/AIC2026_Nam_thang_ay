"""Luồng online phải thật sự đọc được file `.env.*`.

Lỗi gốc: KHÔNG có một chỗ nào trong `online/` hay `scripts/` gọi `load_dotenv`
hay tự parse `.env`. Chỉ `scripts/enrich_keyframes_fpt.py` và
`scripts/fpt_api_preflight.py` tự xử lý `--env-file` cho riêng chúng. Hệ quả:
`uvicorn online.api.app:app` khởi động với `fpt_enabled=False`, không rerank,
không QA LLM, và KHÔNG một dòng cảnh báo nào — nên một buổi đo tưởng là "có
bật FPT" thực ra chạy hoàn toàn không có FPT.

Nạp qua biến `AIC_ENV_FILE` là TƯỜNG MINH có chủ ý: file chứa key thật, nên
việc nó được nạp hay không phải nhìn thấy trên dòng lệnh, không phải hệ quả
của việc đang đứng ở thư mục nào.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from online.config import Settings, load_env_file


class LoadEnvFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self._saved)))

    def _write(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp()) / ".env.test"
        tmp.write_text(body, encoding="utf-8")
        return tmp

    def test_parses_the_shapes_that_actually_appear_in_the_env_file(self) -> None:
        path = self._write(
            "# bình luận\n"
            "\n"
            "AIC_FPT_ENABLED=true\n"
            "export AIC_FPT_BASE_URL=https://mkp-api.fptcloud.com\n"
            'AIC_FPT_LLM_MODEL="Qwen3.6-27B"\n'
            "AIC_METADATA_JSONL = storage/exports_l21_enriched/scenes.jsonl\n"
            "AIC_VISUAL_EMBEDDING_MODEL_REVISION=\n"
            "dòng rác không có dấu bằng\n"
        )
        # Phải dọn sạch trước: `override=False` là CÓ CHỦ Ý, nên một test chạy
        # trước còn để lại biến sẽ khiến file bị bỏ qua đúng như thiết kế.
        for key in (
            "AIC_FPT_ENABLED",
            "AIC_FPT_BASE_URL",
            "AIC_FPT_LLM_MODEL",
            "AIC_METADATA_JSONL",
            "AIC_VISUAL_EMBEDDING_MODEL_REVISION",
        ):
            os.environ.pop(key, None)
        applied = load_env_file(path)

        self.assertIn("AIC_FPT_ENABLED", applied)
        self.assertEqual(os.environ["AIC_FPT_ENABLED"], "true")
        self.assertEqual(os.environ["AIC_FPT_BASE_URL"], "https://mkp-api.fptcloud.com")
        # Nháy kép phải bị bóc, nếu không model id thành `"Qwen3.6-27B"` và
        # FPT trả 404 model-not-found.
        self.assertEqual(os.environ["AIC_FPT_LLM_MODEL"], "Qwen3.6-27B")
        # `KEY = value` có khoảng trắng quanh dấu bằng xuất hiện thật trong file.
        self.assertEqual(
            os.environ["AIC_METADATA_JSONL"], "storage/exports_l21_enriched/scenes.jsonl"
        )
        self.assertEqual(os.environ["AIC_VISUAL_EMBEDDING_MODEL_REVISION"], "")

    def test_existing_environment_wins_over_the_file(self) -> None:
        """Cần cho ablation: `AIC_ENABLE_EXPANSION=false python -m scripts.eval_kis`.

        File để `true` mà vẫn ghi đè được biến đặt trên dòng lệnh thì mọi thí
        nghiệm A/B đều âm thầm đo cùng một cấu hình.
        """

        path = self._write("AIC_ENABLE_EXPANSION=true\n")
        os.environ["AIC_ENABLE_EXPANSION"] = "false"
        load_env_file(path)
        self.assertEqual(os.environ["AIC_ENABLE_EXPANSION"], "false")

        load_env_file(path, override=True)
        self.assertEqual(os.environ["AIC_ENABLE_EXPANSION"], "true")

    def test_settings_from_env_reads_the_file_named_by_aic_env_file(self) -> None:
        path = self._write(
            "AIC_FPT_ENABLED=true\n"
            "AIC_FPT_API_KEY=khong-phai-key-that\n"
            "AIC_FPT_RERANK_MODEL=bge-reranker-v2-m3\n"
        )
        for key in ("AIC_FPT_ENABLED", "AIC_FPT_API_KEY", "AIC_FPT_RERANK_MODEL"):
            os.environ.pop(key, None)
        os.environ["AIC_ENV_FILE"] = str(path)

        settings = Settings.from_env()
        self.assertTrue(settings.fpt_enabled)
        self.assertEqual(settings.fpt_rerank_model, "bge-reranker-v2-m3")

    def test_missing_env_file_is_an_error_not_a_silent_skip(self) -> None:
        """Gõ sai đường dẫn mà server vẫn lên chính là cái bẫy đang muốn tránh."""

        os.environ["AIC_ENV_FILE"] = str(Path(tempfile.mkdtemp()) / "khong-ton-tai.env")
        with self.assertRaises(ValueError) as ctx:
            Settings.from_env()
        self.assertIn("AIC_ENV_FILE", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
