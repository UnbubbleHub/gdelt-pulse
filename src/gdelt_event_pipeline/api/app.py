"""FastAPI application — slim core with router mounts."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gdelt_event_pipeline.api.middleware import rate_limit_middleware
from gdelt_event_pipeline.api.routers.articles import router as articles_router
from gdelt_event_pipeline.api.routers.clusters import router as clusters_router
from gdelt_event_pipeline.api.routers.health import router as health_router
from gdelt_event_pipeline.api.routers.keys import router as keys_router
from gdelt_event_pipeline.api.routers.search import router as search_router
from gdelt_event_pipeline.config.log_setup import setup_logging
from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.storage.database import close_pool, init_pool
from gdelt_event_pipeline.storage.migrations import ensure_schema

setup_logging()

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _is_serverless = bool(os.environ.get("VERCEL"))
    pool = init_pool(
        settings.db,
        min_size=0 if _is_serverless else 2,
        max_size=2 if _is_serverless else 10,
    )
    ensure_schema(pool)
    yield
    close_pool()


app = FastAPI(
    title="GDELT Pulse API",
    description="Hybrid semantic + keyword search over GDELT news events.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.middleware("http")(rate_limit_middleware)

app.include_router(search_router)
app.include_router(clusters_router)
app.include_router(articles_router)
app.include_router(keys_router)
app.include_router(health_router)


# ── Static page routes ──────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/developers", include_in_schema=False)
def developers_page():
    return FileResponse(STATIC_DIR / "developers.html")


@app.get("/search", include_in_schema=False)
def search_page():
    return FileResponse(STATIC_DIR / "search.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse(STATIC_DIR / "dashboard.html")


# ── Runner ──────────────────────────────────────────────────────────


def run() -> None:
    """Convenience entry point: uvicorn gdelt_event_pipeline.api.app:app"""
    import uvicorn

    uvicorn.run("gdelt_event_pipeline.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
