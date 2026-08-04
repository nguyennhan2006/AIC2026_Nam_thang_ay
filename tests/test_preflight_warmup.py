from __future__ import annotations

import asyncio
import unittest

from scripts.preflight import GPU_WARMUP_TASKS, check_gpu_warmup


def run(coro):
    return asyncio.run(coro)


class PreflightWarmupTests(unittest.TestCase):
    def test_warmup_succeeds_with_default_mock_provider(self) -> None:
        # AIC_OFFLINE_PROVIDER mặc định "mock" khi không set — không cần env riêng.
        results = run(check_gpu_warmup())
        self.assertEqual({item["task"] for item in results}, set(GPU_WARMUP_TASKS))
        for item in results:
            self.assertEqual(item["status"], "ok", msg=item.get("error"))
            self.assertEqual(item["provider"], "mock")


if __name__ == "__main__":
    unittest.main()
