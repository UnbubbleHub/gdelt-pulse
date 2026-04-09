"""FastAPI application exposing the hybrid search and cluster browsing endpoints."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.query.models import SearchFilters, SearchRequest
from gdelt_event_pipeline.query.search import hybrid_search
from gdelt_event_pipeline.storage.articles import get_recent_articles
from gdelt_event_pipeline.storage.clusters import (
    get_active_clusters,
    get_cluster_articles,
    get_cluster_by_id,
)
from gdelt_event_pipeline.storage.database import close_pool, init_pool

STATIC_DIR = Path(__file__).parent / "static"

# ── Pydantic response models ─────────────────────────────────────────


class ScoredArticleOut(BaseModel):
    article: dict[str, Any]
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0


class ScoredClusterOut(BaseModel):
    cluster: dict[str, Any]
    cosine_distance: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    total_semantic_hits: int = 0
    total_keyword_hits: int = 0
    articles: list[ScoredArticleOut] = []
    clusters: list[ScoredClusterOut] = []


class ClusterDetailOut(BaseModel):
    cluster: dict[str, Any]
    articles: list[dict[str, Any]]


# ── Helpers ───────────────────────────────────────────────────────────


def _strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove large/internal fields before sending to the client."""
    row.pop("embedding", None)
    row.pop("centroid_embedding", None)
    row.pop("title_tsv", None)
    row.pop("raw_payload", None)
    return row


def _split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated query param into a list, or None."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


# ── App lifecycle ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_pool(settings.db)
    yield
    close_pool()


app = FastAPI(
    title="GDELT Pulse API",
    description="Hybrid semantic + keyword search over GDELT news events.",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
_cors_origins = _settings.api.cors_origins or ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Rate limiting ────────────────────────────────────────────────────

RATE_LIMIT_MAX = 30  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Simple per-IP rate limiter for search endpoints."""
    if not request.url.path.startswith("/api/search"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()

    # Prune old entries
    timestamps = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return Response(
            content='{"detail":"Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    semantic_weight: float = Query(0.5, ge=0.0, le=1.0, description="Semantic vs keyword weight"),
    clusters: bool = Query(False, description="Also search cluster centroids"),
    location: str | None = Query(None, description="Filter by location (comma-separated)"),
    person: str | None = Query(None, description="Filter by person (comma-separated)"),
    org: str | None = Query(None, description="Filter by organization (comma-separated)"),
    theme: str | None = Query(None, description="Filter by theme (comma-separated)"),
    domain: str | None = Query(None, description="Filter by domain (comma-separated)"),
    source: str | None = Query(None, description="Filter by source (comma-separated)"),
    date_from: datetime | None = Query(None, description="Start date (ISO format)"),  # noqa: B008
    date_to: datetime | None = Query(None, description="End date (ISO format)"),  # noqa: B008
):
    """Hybrid semantic + keyword search over articles and clusters."""
    filters = SearchFilters(
        locations=_split_csv(location),
        persons=_split_csv(person),
        organizations=_split_csv(org),
        themes=_split_csv(theme),
        domains=_split_csv(domain),
        sources=_split_csv(source),
        date_from=date_from,
        date_to=date_to,
    )
    has_filters = any(getattr(filters, f) is not None for f in filters.__dataclass_fields__)

    request = SearchRequest(
        query=q,
        filters=filters if has_filters else None,
        limit=limit,
        semantic_weight=semantic_weight,
        search_clusters=clusters,
    )

    result = hybrid_search(request)

    return SearchResponse(
        query=result.query,
        total_semantic_hits=result.total_semantic_hits,
        total_keyword_hits=result.total_keyword_hits,
        articles=[
            ScoredArticleOut(
                article=_strip_internal_fields(sa.article),
                semantic_rank=sa.semantic_rank,
                keyword_rank=sa.keyword_rank,
                rrf_score=sa.rrf_score,
            )
            for sa in result.articles
        ],
        clusters=[
            ScoredClusterOut(
                cluster=_strip_internal_fields(sc.cluster),
                cosine_distance=sc.cosine_distance,
                rank=sc.rank,
            )
            for sc in result.clusters
        ],
    )


@app.get("/api/stats")
def get_stats():
    """Dashboard statistics."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS cnt FROM articles")
            total_articles = cur.fetchone()["cnt"]
            cur.execute("SELECT count(*) AS cnt FROM articles WHERE title IS NOT NULL")
            titled = cur.fetchone()["cnt"]
            cur.execute("SELECT count(*) AS cnt FROM articles WHERE embedding IS NOT NULL")
            embedded = cur.fetchone()["cnt"]
            cur.execute("SELECT count(*) AS cnt FROM clusters WHERE is_active = true")
            total_clusters = cur.fetchone()["cnt"]
            cur.execute("SELECT max(article_count) AS val FROM clusters WHERE is_active = true")
            largest_cluster = cur.fetchone()["val"] or 0
            cur.execute("SELECT count(*) AS cnt FROM cluster_memberships")
            total_memberships = cur.fetchone()["cnt"]
    return {
        "total_articles": total_articles,
        "titled_articles": titled,
        "embedded_articles": embedded,
        "total_clusters": total_clusters,
        "largest_cluster": largest_cluster,
        "total_memberships": total_memberships,
    }


@app.get("/api/articles", response_model=list[dict[str, Any]])
def list_articles(
    limit: int = Query(50, ge=1, le=200, description="Max articles to return"),
):
    """List recent articles, newest first."""
    rows = get_recent_articles(limit=limit)
    return [_strip_internal_fields(row) for row in rows]


@app.get("/api/clusters", response_model=list[dict[str, Any]])
def list_clusters(
    limit: int = Query(100, ge=1, le=500, description="Max clusters to return"),
    sort: str = Query("recent", description="Sort: recent, articles, oldest"),
    location: str | None = Query(None, description="Filter by location (comma-separated)"),
    person: str | None = Query(None, description="Filter by person (comma-separated)"),
    org: str | None = Query(None, description="Filter by organization (comma-separated)"),
    theme: str | None = Query(None, description="Filter by theme (comma-separated)"),
    domain: str | None = Query(None, description="Filter by domain (comma-separated)"),
    source: str | None = Query(None, description="Filter by source (comma-separated)"),
    date_from: datetime | None = Query(None, description="Start date (ISO format)"),  # noqa: B008
    date_to: datetime | None = Query(None, description="End date (ISO format)"),  # noqa: B008
):
    """List active clusters, optionally filtered by article metadata."""
    locations = _split_csv(location)
    persons = _split_csv(person)
    organizations = _split_csv(org)
    themes = _split_csv(theme)
    domains = _split_csv(domain)
    sources = _split_csv(source)

    has_filters = any(
        [locations, persons, organizations, themes, domains, sources, date_from, date_to]
    )

    if not has_filters:
        rows = get_active_clusters(limit=limit, sort=sort)
        return [_strip_internal_fields(row) for row in rows]

    from gdelt_event_pipeline.storage.database import get_pool

    order_clause = {
        "articles": "c.article_count DESC",
        "oldest": "c.first_article_at ASC NULLS LAST",
    }.get(sort, "c.last_article_at DESC NULLS LAST")

    article_conditions: list[str] = []
    params: list = []

    if locations:
        article_conditions.append("a.locations::text ILIKE %s")
        params.append(f"%{locations[0]}%")
    if persons:
        article_conditions.append("a.persons::text ILIKE %s")
        params.append(f"%{persons[0]}%")
    if organizations:
        article_conditions.append("a.organizations::text ILIKE %s")
        params.append(f"%{organizations[0]}%")
    if themes:
        article_conditions.append("a.themes::text ILIKE %s")
        params.append(f"%{themes[0]}%")
    if domains:
        article_conditions.append("a.domain ILIKE %s")
        params.append(f"%{domains[0]}%")
    if sources:
        article_conditions.append("(a.source_common_name ILIKE %s OR a.canonical_source ILIKE %s)")
        params.extend([f"%{sources[0]}%", f"%{sources[0]}%"])
    if date_from:
        article_conditions.append("a.gdelt_timestamp >= %s")
        params.append(date_from)
    if date_to:
        article_conditions.append("a.gdelt_timestamp <= %s")
        params.append(date_to)

    article_where = " AND ".join(article_conditions)
    params.append(limit)

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT c.*
                FROM clusters c
                WHERE c.is_active = true AND c.id IN (
                    SELECT DISTINCT cm.cluster_id
                    FROM cluster_memberships cm
                    JOIN articles a ON a.id = cm.article_id
                    WHERE {article_where}
                )
                ORDER BY {order_clause}
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

    return [_strip_internal_fields(row) for row in rows]


@app.get("/api/clusters/{cluster_id}", response_model=ClusterDetailOut)
def get_cluster_detail(cluster_id: str):
    """Get a single cluster and its member articles."""
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    articles = get_cluster_articles(cluster_id)

    return ClusterDetailOut(
        cluster=_strip_internal_fields(cluster),
        articles=[_strip_internal_fields(a) for a in articles],
    )


# ── Runner ────────────────────────────────────────────────────────────


def run() -> None:
    """Convenience entry point: uvicorn gdelt_event_pipeline.api.app:app"""
    import uvicorn

    uvicorn.run("gdelt_event_pipeline.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
