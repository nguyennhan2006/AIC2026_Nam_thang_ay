"""scripts/caption_qwen3vl.py: provider config resolution — the only pure/testable
logic in that file (everything else calls a real remote API or reads real images).

Importing the module resolves AIC_QWEN3VL_PROVIDER from the real environment/.env at
import time (same as the original script's OpenRouter-key check it replaced) — patch
OPENROUTER_API_KEY for the duration of the import so it succeeds regardless of the
local .env, then test resolve_provider_config directly with fake env lookups.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-for-import-only"}):
    from scripts.caption_qwen3vl import (
        PROVIDER_OPENROUTER,
        PROVIDER_VLLM,
        _env_int,
        resolve_provider_config,
    )


def _env_from(mapping: dict):
    return lambda key: mapping.get(key)


class ResolveProviderConfigTests(unittest.TestCase):
    def test_defaults_to_openrouter(self) -> None:
        config = resolve_provider_config(_env_from({"OPENROUTER_API_KEY": "sk-test"}))
        self.assertEqual(config.provider, PROVIDER_OPENROUTER)
        self.assertEqual(config.server_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.model, "qwen/qwen3-vl-32b-instruct")
        self.assertEqual(config.api_key, "sk-test")

    def test_openrouter_without_api_key_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            resolve_provider_config(_env_from({"AIC_QWEN3VL_PROVIDER": "openrouter"}))

    def test_vllm_requires_server_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "AIC_QWEN3VL_SERVER_URL"):
            resolve_provider_config(_env_from({"AIC_QWEN3VL_PROVIDER": "vllm"}))

    def test_vllm_with_server_url_resolves_with_defaults(self) -> None:
        config = resolve_provider_config(_env_from({
            "AIC_QWEN3VL_PROVIDER": "vllm",
            "AIC_QWEN3VL_SERVER_URL": "http://127.0.0.1:8001/v1",
        }))
        self.assertEqual(config.provider, PROVIDER_VLLM)
        self.assertEqual(config.server_base_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(config.model, "Qwen/Qwen3-VL-32B-Instruct")
        self.assertEqual(config.api_key, "not-needed")

    def test_vllm_custom_model_and_api_key_override_defaults(self) -> None:
        config = resolve_provider_config(_env_from({
            "AIC_QWEN3VL_PROVIDER": "vllm",
            "AIC_QWEN3VL_SERVER_URL": "http://127.0.0.1:8001/v1",
            "AIC_QWEN3VL_MODEL": "custom/model",
            "AIC_QWEN3VL_API_KEY": "secret",
        }))
        self.assertEqual(config.model, "custom/model")
        self.assertEqual(config.api_key, "secret")

    def test_invalid_provider_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "AIC_QWEN3VL_PROVIDER"):
            resolve_provider_config(_env_from({"AIC_QWEN3VL_PROVIDER": "bogus"}))

    def test_provider_value_is_case_insensitive(self) -> None:
        config = resolve_provider_config(_env_from({
            "AIC_QWEN3VL_PROVIDER": "VLLM",
            "AIC_QWEN3VL_SERVER_URL": "http://127.0.0.1:8001/v1",
        }))
        self.assertEqual(config.provider, PROVIDER_VLLM)


class EnvIntTests(unittest.TestCase):
    """_env_int distinguishes 'key absent' (-> default) from 'key present but empty'
    (-> None, explicit 'no limit' for AIC_QWEN3VL_LIMIT once ready to run the full corpus)."""

    def test_absent_key_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIC_QWEN3VL_TEST_KEY", None)
            self.assertEqual(_env_int("AIC_QWEN3VL_TEST_KEY", 1), 1)

    def test_explicit_empty_value_returns_none(self) -> None:
        with patch.dict(os.environ, {"AIC_QWEN3VL_TEST_KEY": ""}):
            self.assertIsNone(_env_int("AIC_QWEN3VL_TEST_KEY", 1))

    def test_explicit_value_is_parsed(self) -> None:
        with patch.dict(os.environ, {"AIC_QWEN3VL_TEST_KEY": "5"}):
            self.assertEqual(_env_int("AIC_QWEN3VL_TEST_KEY", 1), 5)


if __name__ == "__main__":
    unittest.main()
