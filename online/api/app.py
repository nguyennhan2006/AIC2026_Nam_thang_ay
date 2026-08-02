"""FastAPI application factory with lifespan-managed dependencies."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import secrets
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


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, resolved.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = await build_container(resolved)
        yield
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
        if resolved.api_key and request.url.path.startswith("/v1/") and request.url.path != "/v1/health":
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
