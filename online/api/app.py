"""FastAPI application factory with lifespan-managed dependencies."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from online.api.container import build_container
from online.api.routes import router
from online.config import Settings
from online.errors import InvalidQueryError, OnlineError, TaskConflictError
from online.services.capabilities import UnsupportedSearchOptionError


class Boot:
    """Tiến độ nạp container, đọc được NGAY cả khi container chưa tồn tại.

    Trên corpus thi đấu (873 video / 87.742 scene / 176.707 keyframe) việc nạp
    mất ~4 phút. Trước đây toàn bộ quãng đó nằm trong `lifespan`, nên uvicorn
    chưa mở cổng: trình duyệt báo "không kết nối được" và không có cách nào
    phân biệt "đang nạp" với "đã chết". Giờ cổng mở ngay, còn tiến độ đọc qua
    `GET /v1/startup`.
    """

    __slots__ = ("task", "started_at", "phase", "error")

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.started_at = time.monotonic()
        self.phase = "starting"
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.phase == "ready"

    @property
    def elapsed_sec(self) -> float:
        return round(time.monotonic() - self.started_at, 1)

    def snapshot(self) -> dict:
        return {
            "status": "ready" if self.ready else ("failed" if self.error else "warming"),
            "phase": self.phase,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, resolved.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        boot = Boot()
        app.state.boot = boot
        app.state.container = None
        logger = logging.getLogger("online.api.boot")

        def build_blocking():
            # `build_container` là coroutine nhưng mọi việc nặng bên trong đều
            # đồng bộ (đọc JSONL, dựng BM25, nạp ma trận vector, nạp model).
            # Chạy nó trong MỘT event loop riêng ở thread nền, thay vì rải
            # `to_thread` khắp hàm: không có gì trong đó bám vào loop phục vụ
            # (đã soát: `asyncio.Lock` của session store bind lười từ 3.10,
            # các Semaphore đều tạo tại chỗ lúc gọi).
            return asyncio.run(build_container(resolved, progress=set_phase))

        def set_phase(name: str) -> None:
            boot.phase = name
            logger.info("boot phase: %s (%.1fs)", name, boot.elapsed_sec)

        async def run_build():
            try:
                container = await asyncio.to_thread(build_blocking)
            except Exception as exc:  # noqa: BLE001 - phải nói được vì sao chết
                boot.error = f"{type(exc).__name__}: {exc}"
                boot.phase = "failed"
                logger.exception("nạp container thất bại")
                raise
            app.state.container = container
            boot.phase = "ready"
            logger.info("sẵn sàng phục vụ sau %.1fs", boot.elapsed_sec)
            return container

        boot.task = asyncio.create_task(run_build())
        try:
            yield
        finally:
            boot.task.cancel()
            # Nuốt kết quả để asyncio không log "Task exception was never
            # retrieved" khi tắt server giữa lúc còn đang nạp.
            try:
                await boot.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            app.state.container = None

    app = FastAPI(
        title=resolved.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        # `/v1/startup` cũng miễn token: UI phải hỏi được "server nạp xong
        # chưa" TRƯỚC khi người dùng kịp dán token vào.
        if (
            resolved.api_key
            and request.url.path.startswith("/v1/")
            and request.url.path not in ("/v1/health", "/v1/startup")
        ):
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {resolved.api_key}"):
                return JSONResponse(status_code=401, content={"detail": "invalid API token"})
        return await call_next(request)

    # Lỗi do request sai (4xx) phải tách khỏi lỗi hạ tầng (503) — client cần
    # biết mình gửi sai chứ không phải "thử lại sau".
    client_errors: dict[type[OnlineError], int] = {
        TaskConflictError: 422,
        InvalidQueryError: 422,
        UnsupportedSearchOptionError: 422,
    }

    @app.exception_handler(OnlineError)
    async def handle_online_error(_: Request, exc: OnlineError) -> JSONResponse:
        return JSONResponse(
            status_code=client_errors.get(type(exc), 503),
            content={"error": {"code": exc.code, "message": str(exc)}},
        )

    app.include_router(router)
    ui_dir = Path(__file__).resolve().parents[1] / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/ui/")
    return app


app = create_app()
