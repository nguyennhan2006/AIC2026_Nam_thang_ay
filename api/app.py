"""FastAPI application factory with lifespan-managed dependencies."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from online.api.container import build_container
from online.api.routes import router
from online.config import Settings
from online.errors import OnlineError


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
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(OnlineError)
    async def handle_online_error(_: Request, exc: OnlineError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
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
