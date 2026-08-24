"""FB-001: server phải MỞ ngay, nạp ở luồng nền.

Trên corpus thi đấu việc nạp mất ~4 phút. Trước đây toàn bộ quãng đó nằm
trong `lifespan`, nên uvicorn chưa mở cổng và trình duyệt chỉ báo "không kết
nối được" — không phân biệt được "đang nạp" với "đã chết", và cả đội ngồi đợi
một màn hình trắng.

Hợp đồng được khoá ở đây:

* `GET /v1/startup` trả lời TỨC THÌ kể cả khi container còn đang nạp;
* endpoint thường CHỜ nạp xong rồi chạy, không ném 503 bắt người dùng bấm lại;
* nạp thất bại thì cả hai đường đều nói rõ lỗi, không treo vô hạn.
"""

from __future__ import annotations

import asyncio
import dataclasses
import unittest

from fastapi.testclient import TestClient

import online.api.app as app_module
from online.api.app import create_app
from online.config import Settings

REAL_BUILD = app_module.build_container
BUILD_DELAY_SEC = 0.4


class AsyncStartupTest(unittest.TestCase):
    """Container chậm giả lập — đo đúng khoảng thời gian server còn đang nạp."""

    def setUp(self) -> None:
        self.phases: list[str] = []

        async def slow_build(settings, progress=None):
            if progress is not None:
                progress("giả lập nạp")
            await asyncio.sleep(BUILD_DELAY_SEC)
            container = await REAL_BUILD(settings, progress=progress)
            self.phases.append("built")
            return container

        app_module.build_container = slow_build
        self.addCleanup(setattr, app_module, "build_container", REAL_BUILD)

    def test_startup_tra_loi_ngay_trong_luc_con_dang_nap(self):
        with TestClient(create_app(Settings.from_env())) as client:
            first = client.get("/v1/startup")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(
                first.json()["status"], "warming",
                "hỏi lúc còn đang nạp mà đã báo ready — server không thể xong nhanh vậy",
            )
            self.assertIsNone(first.json()["error"])

            # Endpoint thường CHỜ chứ không 503.
            health = client.get("/v1/health")
            self.assertEqual(health.status_code, 200, health.text)
            self.assertEqual(health.json()["status"], "ok")

            after = client.get("/v1/startup")
            self.assertEqual(after.json()["status"], "ready")
            self.assertEqual(after.json()["phase"], "ready")
            self.assertGreater(after.json()["elapsed_sec"], 0.0)

    def test_startup_khong_can_token(self):
        """UI phải hỏi được tiến độ TRƯỚC khi người dùng kịp dán token."""

        settings = dataclasses.replace(Settings.from_env(), api_key="bi-mat")
        with TestClient(create_app(settings)) as client:
            self.assertEqual(client.get("/v1/startup").status_code, 200)
            # Đường có bảo vệ vẫn phải chặn — không được nới nhầm cả cụm /v1.
            self.assertEqual(client.get("/v1/videos").status_code, 401)


class FailedStartupTest(unittest.TestCase):
    def setUp(self) -> None:
        async def broken_build(settings, progress=None):
            if progress is not None:
                progress("metadata")
            await asyncio.sleep(0)
            raise RuntimeError("thiếu scenes.jsonl")

        app_module.build_container = broken_build
        self.addCleanup(setattr, app_module, "build_container", REAL_BUILD)

    def test_nap_hong_thi_noi_ro_thay_vi_treo(self):
        with TestClient(create_app(Settings.from_env())) as client:
            # Ép cho task nạp chạy tới lúc vỡ trước khi hỏi.
            health = client.get("/v1/health")
            self.assertEqual(health.status_code, 503)
            self.assertIn("thiếu scenes.jsonl", health.json()["detail"])

            state = client.get("/v1/startup").json()
            self.assertEqual(state["status"], "failed")
            self.assertIn("thiếu scenes.jsonl", state["error"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
