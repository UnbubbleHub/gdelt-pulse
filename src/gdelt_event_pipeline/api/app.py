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


def _ensure_schema() -> None:
    """Create tables if they don't exist (first deploy on Railway etc.)."""
    import logging

    from gdelt_event_pipeline.storage.database import get_pool

    logger = logging.getLogger(__name__)
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_name = 'articles'"
                ")"
            )
            row = cur.fetchone()
            exists = row[0] if isinstance(row, (tuple, list)) else row.get("exists", False)
        if not exists:
            logger.info("Tables not found — running schema initialization...")
            schema_path = Path(__file__).resolve().parents[3] / "sql" / "001_schema.sql"
            if not schema_path.exists():
                schema_path = Path("/app/sql/001_schema.sql")
            sql = schema_path.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Schema created successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_pool(settings.db)
    _ensure_schema()
    yield
    close_pool()


app = FastAPI(
    title="GDELT Pulse API",
    description="Hybrid semantic + keyword search over GDELT news events.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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
    if not (
        request.url.path.startswith("/api/search")
        or request.url.path.startswith("/api/clusters")
        or request.url.path.startswith("/api/globe")
        or request.url.path.startswith("/api/polarization")
        or request.url.path.startswith("/api/gravity")
        or request.url.path.startswith("/api/asymmetry")
        or request.url.path.startswith("/api/sources")
        or request.url.path.startswith("/api/propagation")
        or request.url.path.startswith("/api/velocity")
    ):
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


@app.get("/globe", include_in_schema=False)
def globe():
    """Serve the NewsGlobe frontend."""
    return FileResponse(STATIC_DIR / "globe.html")


@app.get("/polarization", include_in_schema=False)
def polarization_page():
    """Serve the Narrative Polarization frontend."""
    return FileResponse(STATIC_DIR / "polarization.html")


@app.get("/gravity", include_in_schema=False)
def gravity_page():
    """Serve the Geopolitical Gravity Map frontend."""
    return FileResponse(STATIC_DIR / "gravity.html")


@app.get("/asymmetry", include_in_schema=False)
def asymmetry_page():
    """Serve the Attention Asymmetry frontend."""
    return FileResponse(STATIC_DIR / "asymmetry.html")


@app.get("/sources", include_in_schema=False)
def sources_page():
    """Serve the Source DNA frontend."""
    return FileResponse(STATIC_DIR / "sources.html")


@app.get("/propagation", include_in_schema=False)
def propagation_page():
    """Serve the Story Propagation frontend."""
    return FileResponse(STATIC_DIR / "propagation.html")


@app.get("/velocity", include_in_schema=False)
def velocity_page():
    """Serve the Topic Velocity frontend."""
    return FileResponse(STATIC_DIR / "velocity.html")


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


# ── Globe endpoints ──────────────────────────────────────────────────


@app.get("/api/globe/clusters")
def globe_clusters(
    mode: str = Query("live", description="Filter mode: live, rising, silent"),
    limit: int = Query(12, ge=1, le=50, description="Max clusters"),
):
    """Return top clusters with geographic coordinates for the 3D globe.

    Modes:
    - live:   Clusters with articles in the last 2 hours, sorted by article_count.
    - rising: Clusters whose article count grew fastest in the last 6 hours.
    - silent: Large clusters that have gone quiet (no new articles for 6-72h).
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            if mode == "rising":
                # Stories picking up steam: clusters that existed for a while
                # but got a burst of new articles in the last 2 hours.
                # Ranked by recent article count — the surge signal.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           recent.recent_count,
                           CASE WHEN c.article_count > 0
                                THEN recent.recent_count::float / c.article_count
                                ELSE 0 END AS velocity,
                           EXTRACT(EPOCH FROM (now() - c.first_article_at)) / 3600
                               AS age_hours
                    FROM clusters c
                    JOIN (
                        SELECT cm.cluster_id, count(*) AS recent_count
                        FROM cluster_memberships cm
                        JOIN articles a ON a.id = cm.article_id
                        WHERE a.gdelt_timestamp >= now() - interval '2 hours'
                        GROUP BY cm.cluster_id
                        HAVING count(*) >= 2
                    ) recent ON recent.cluster_id = c.id
                    WHERE c.is_active = true
                      AND c.article_count >= 5
                      AND c.first_article_at <= now() - interval '1 hour'
                    ORDER BY recent.recent_count DESC, velocity DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            elif mode == "silent":
                # Big stories that went quiet — no new articles in 3h+
                # but were active within the last 48h.
                # Shows hours since last article so the UI can display
                # "quiet for 8h" etc.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           0 AS recent_count, 0 AS velocity,
                           EXTRACT(EPOCH FROM (now() - c.last_article_at)) / 3600
                               AS silent_hours
                    FROM clusters c
                    WHERE c.is_active = true
                      AND c.article_count >= 5
                      AND c.last_article_at < now() - interval '6 hours'
                      AND c.last_article_at >= now() - interval '72 hours'
                    ORDER BY c.article_count DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            else:
                # live: what newsrooms are writing about RIGHT NOW.
                # Clusters ranked by how many articles arrived in the
                # last 2 hours — the hottest stories this moment.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           COALESCE(recent.recent_count, 0) AS recent_count,
                           0 AS velocity
                    FROM clusters c
                    JOIN (
                        SELECT cm.cluster_id, count(*) AS recent_count
                        FROM cluster_memberships cm
                        JOIN articles a ON a.id = cm.article_id
                        WHERE a.gdelt_timestamp >= now() - interval '2 hours'
                        GROUP BY cm.cluster_id
                    ) recent ON recent.cluster_id = c.id
                    WHERE c.is_active = true
                      AND c.article_count >= 2
                    ORDER BY recent.recent_count DESC, c.article_count DESC
                    LIMIT %s
                    """,
                    (limit,),
                )

            clusters = cur.fetchall()
            cluster_ids = [c["id"] for c in clusters]

            if not cluster_ids:
                return []

            # Fetch primary location (most common lat/lon) for each cluster
            cur.execute(
                """
                SELECT cm.cluster_id,
                       a.locations,
                       a.themes,
                       a.title,
                       a.url,
                       a.domain,
                       a.gdelt_timestamp
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.locations IS NOT NULL
                  AND jsonb_array_length(a.locations) > 0
                ORDER BY a.gdelt_timestamp DESC
                """,
                (cluster_ids,),
            )
            article_rows = cur.fetchall()

    # Build location + metadata per cluster
    cluster_locations: dict[str, list[dict]] = defaultdict(list)
    cluster_articles_data: dict[str, list[dict]] = defaultdict(list)

    for row in article_rows:
        cid = str(row["cluster_id"])
        locs = row["locations"]
        if isinstance(locs, list):
            for loc in locs:
                if loc.get("lat") is not None and loc.get("lon") is not None:
                    cluster_locations[cid].append(loc)
        cluster_articles_data[cid].append(
            {
                "title": row["title"],
                "url": row["url"],
                "domain": row["domain"],
                "gdelt_timestamp": row["gdelt_timestamp"],
                "themes": row["themes"],
            }
        )

    results = []
    for c in clusters:
        cid = str(c["id"])
        locs = cluster_locations.get(cid, [])

        # Pick the most common location as primary
        lat, lon, location_name, country_code = None, None, None, None
        if locs:
            # Count occurrences of each (lat, lon) rounded to 1 decimal
            from collections import Counter

            coord_counts = Counter()
            coord_info: dict[tuple, dict] = {}
            for loc in locs:
                key = (round(loc["lat"], 1), round(loc["lon"], 1))
                coord_counts[key] += 1
                if key not in coord_info:
                    coord_info[key] = loc
            best = coord_counts.most_common(1)[0][0]
            info = coord_info[best]
            lat, lon = info["lat"], info["lon"]
            location_name = info.get("name")
            country_code = info.get("country_code")

        # Collect top themes across articles
        theme_counts: dict[str, int] = {}
        for art in cluster_articles_data.get(cid, [])[:20]:
            if art.get("themes") and isinstance(art["themes"], list):
                for t in art["themes"][:5]:
                    tn = t.get("theme", "")
                    if (
                        tn
                        and not tn.startswith("TAX_ETHNICITY")
                        and not tn.startswith("TAX_WORLDLANGUAGE")
                    ):
                        theme_counts[tn] = theme_counts.get(tn, 0) + 1
        top_themes = sorted(theme_counts, key=theme_counts.get, reverse=True)[:5]

        # Category from top theme
        category = _categorize_themes(top_themes)

        # Sample articles (latest 5)
        sample_articles = []
        for art in cluster_articles_data.get(cid, [])[:5]:
            sample_articles.append(
                {
                    "title": art["title"],
                    "url": art["url"],
                    "domain": art["domain"],
                }
            )

        results.append(
            {
                "id": cid,
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "recent_count": c.get("recent_count", 0),
                "velocity": round(c.get("velocity", 0), 3),
                "silent_hours": round(c["silent_hours"], 1) if c.get("silent_hours") else None,
                "age_hours": round(c["age_hours"], 1) if c.get("age_hours") else None,
                "lat": lat,
                "lon": lon,
                "location_name": location_name,
                "country_code": country_code,
                "category": category,
                "top_themes": top_themes,
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
                "sample_articles": sample_articles,
            }
        )

    return results


def _categorize_themes(themes: list[str]) -> str:
    """Map GDELT themes to a simple category for color coding."""
    theme_str = " ".join(themes).upper()
    if any(
        k in theme_str
        for k in [
            "MILITARY",
            "WAR",
            "ARMED",
            "TERROR",
            "CONFLICT",
            "KILL",
            "WOUND",
            "ARREST",
            "CRIME",
            "ATTACK",
            "REBELLION",
            "INSURGENT",
            "DRONE",
            "WEAPON",
            "BOMB",
            "SHOOT",
            "HOSTAGE",
            "SIEGE",
        ]
    ):
        return "conflict"
    if any(
        k in theme_str
        for k in [
            "ECON",
            "MARKET",
            "TRADE",
            "FINANCE",
            "BUSINESS",
            "TAX_FNCACT",
            "STOCK",
            "INVEST",
            "BANKRUPT",
            "INFLATION",
            "GDP",
            "UNEMPLOY",
            "CRYPTO",
            "TARIFF",
            "DEBT",
            "REVENUE",
        ]
    ):
        return "economy"
    if any(
        k in theme_str
        for k in [
            "ELECT",
            "POLITIC",
            "GOVERN",
            "DIPLOMAT",
            "LEGISLAT",
            "VOTE",
            "PARLIAMENT",
            "CONGRESS",
            "PRESIDENT",
            "MINISTER",
            "SANCTION",
            "TREATY",
            "SUMMIT",
            "CAUCUS",
            "CAMPAIGN",
            "PARTY",
        ]
    ):
        return "politics"
    if any(
        k in theme_str
        for k in [
            "HEALTH",
            "DISEASE",
            "MEDICAL",
            "PANDEMIC",
            "HOSPITAL",
            "VACCINE",
            "DRUG",
            "VIRUS",
            "OUTBREAK",
            "WHO_",
            "SURGEON",
            "CANCER",
        ]
    ):
        return "health"
    if any(
        k in theme_str
        for k in [
            "ENV_",
            "ENVIRON",
            "CLIMATE",
            "DISASTER",
            "QUAKE",
            "FLOOD",
            "HURRICANE",
            "WILDFIRE",
            "DROUGHT",
            "EMISSION",
            "CARBON",
            "STORM",
            "TSUNAMI",
            "TORNADO",
            "VOLCANO",
        ]
    ):
        return "environment"
    if any(
        k in theme_str
        for k in [
            "TECH",
            "CYBER",
            "AI_",
            "DIGITAL",
            "SCIENCE",
            "ROBOT",
            "SPACE",
            "INTERNET",
            "SOFTWARE",
            "HACK",
            "DATA_",
            "COMPUTING",
        ]
    ):
        return "technology"
    if any(
        k in theme_str
        for k in [
            "HUMAN_RIGHTS",
            "PROTEST",
            "REFUGEE",
            "MIGRATION",
            "EDUCATION",
            "WOMEN",
            "CHILD",
            "POVERTY",
            "DISCRIMINATION",
            "RIGHTS",
            "RELIGION",
            "CULTURE",
            "SPORT",
            "ENTERTAINMENT",
        ]
    ):
        return "society"
    return "general"


# ── Polarization endpoints ───────────────────────────────


@app.get("/api/polarization")
def get_polarization(
    limit: int = Query(30, ge=1, le=100, description="Max clusters to return"),
    min_articles: int = Query(20, ge=5, le=500, description="Minimum articles per cluster"),
):
    """Return the most narratively polarized story clusters.

    Polarization score = standard deviation of tone across articles
    within the same cluster.  High stddev means the same story is
    being framed very differently by different sources.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.representative_title,
                    c.article_count,
                    c.first_article_at,
                    c.last_article_at,
                    round(avg((a.tone->>'tone')::float)::numeric, 3)     AS avg_tone,
                    round(stddev((a.tone->>'tone')::float)::numeric, 3)  AS tone_stddev,
                    round(min((a.tone->>'tone')::float)::numeric, 3)     AS tone_min,
                    round(max((a.tone->>'tone')::float)::numeric, 3)     AS tone_max,
                    count(DISTINCT a.domain)                              AS distinct_sources
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id
                WHERE c.is_active = true
                  AND c.article_count >= %s
                  AND a.tone IS NOT NULL
                GROUP BY c.id
                HAVING count(*) >= %s AND stddev((a.tone->>'tone')::float) IS NOT NULL
                ORDER BY tone_stddev DESC
                LIMIT %s
                """,
                (min_articles, min_articles, limit),
            )
            clusters = cur.fetchall()

            if not clusters:
                return []

            cluster_ids = [c["id"] for c in clusters]

            # Per-source tone breakdown for each cluster
            cur.execute(
                """
                SELECT
                    cm.cluster_id,
                    COALESCE(a.canonical_source, a.domain) AS source,
                    round(avg((a.tone->>'tone')::float)::numeric, 3) AS avg_tone,
                    count(*) AS article_count
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.tone IS NOT NULL
                GROUP BY cm.cluster_id, COALESCE(a.canonical_source, a.domain)
                HAVING count(*) >= 2
                ORDER BY cm.cluster_id, avg((a.tone->>'tone')::float) ASC
                """,
                (cluster_ids,),
            )
            source_rows = cur.fetchall()

            # Tone histogram buckets per cluster
            cur.execute(
                """
                SELECT
                    cm.cluster_id,
                    width_bucket((a.tone->>'tone')::float, -10, 10, 20) AS bucket,
                    count(*) AS cnt
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.tone IS NOT NULL
                GROUP BY cm.cluster_id, bucket
                ORDER BY cm.cluster_id, bucket
                """,
                (cluster_ids,),
            )
            hist_rows = cur.fetchall()

    # Build per-cluster source breakdown
    sources_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        cid = str(row["cluster_id"])
        sources_by_cluster[cid].append(
            {
                "source": row["source"],
                "avg_tone": float(row["avg_tone"]),
                "article_count": row["article_count"],
            }
        )

    # Build per-cluster histogram
    hist_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in hist_rows:
        cid = str(row["cluster_id"])
        bucket_idx = row["bucket"]
        # width_bucket(val, -10, 10, 20) → buckets 1..20, each 1.0 wide
        low = -10 + (bucket_idx - 1) * 1.0
        hist_by_cluster[cid].append(
            {
                "range_low": low,
                "range_high": low + 1.0,
                "count": row["cnt"],
            }
        )

    results = []
    for c in clusters:
        cid = str(c["id"])
        sources = sources_by_cluster.get(cid, [])
        histogram = hist_by_cluster.get(cid, [])

        results.append(
            {
                "id": cid,
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
                "polarization_score": float(c["tone_stddev"]),
                "avg_tone": float(c["avg_tone"]),
                "tone_min": float(c["tone_min"]),
                "tone_max": float(c["tone_max"]),
                "distinct_sources": c["distinct_sources"],
                "sources": sources[:15],  # top 15 sources
                "histogram": histogram,
            }
        )

    return results


@app.get("/api/polarization/{cluster_id}")
def get_polarization_detail(cluster_id: str):
    """Detailed polarization breakdown for a single cluster.

    Returns every article with its tone, grouped by source.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, representative_title, article_count, "
                "first_article_at, last_article_at FROM clusters WHERE id = %s",
                (cluster_id,),
            )
            cluster = cur.fetchone()
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            cur.execute(
                """
                SELECT
                    a.id,
                    a.title,
                    a.url,
                    a.canonical_url,
                    a.domain,
                    COALESCE(a.canonical_source, a.domain) AS source,
                    a.gdelt_timestamp,
                    (a.tone->>'tone')::float AS tone,
                    (a.tone->>'polarity')::float AS polarity,
                    (a.tone->>'positive_score')::float AS positive_score,
                    (a.tone->>'negative_score')::float AS negative_score
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = %s
                  AND a.tone IS NOT NULL
                ORDER BY (a.tone->>'tone')::float ASC
                """,
                (cluster_id,),
            )
            articles = cur.fetchall()

    # Group by source
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        source_groups[a["source"]].append(
            {
                "title": a["title"],
                "url": a["canonical_url"] or a["url"],
                "domain": a["domain"],
                "gdelt_timestamp": a["gdelt_timestamp"],
                "tone": round(a["tone"], 3) if a["tone"] is not None else None,
                "polarity": round(a["polarity"], 3) if a["polarity"] is not None else None,
                "positive_score": round(a["positive_score"], 3)
                if a["positive_score"] is not None
                else None,
                "negative_score": round(a["negative_score"], 3)
                if a["negative_score"] is not None
                else None,
            }
        )

    # Sort sources by average tone
    source_summaries = []
    for source, arts in source_groups.items():
        tones = [a["tone"] for a in arts if a["tone"] is not None]
        avg_t = sum(tones) / len(tones) if tones else 0
        source_summaries.append(
            {
                "source": source,
                "avg_tone": round(avg_t, 3),
                "article_count": len(arts),
                "articles": arts,
            }
        )
    source_summaries.sort(key=lambda x: x["avg_tone"])

    # Overall stats
    all_tones = [a["tone"] for a in articles if a["tone"] is not None]
    avg = sum(all_tones) / len(all_tones) if all_tones else 0
    variance = sum((t - avg) ** 2 for t in all_tones) / len(all_tones) if all_tones else 0
    stddev = variance**0.5

    return {
        "cluster": {
            "id": str(cluster["id"]),
            "title": cluster["representative_title"],
            "article_count": cluster["article_count"],
            "first_article_at": cluster["first_article_at"],
            "last_article_at": cluster["last_article_at"],
        },
        "polarization_score": round(stddev, 3),
        "avg_tone": round(avg, 3),
        "tone_min": round(min(all_tones), 3) if all_tones else 0,
        "tone_max": round(max(all_tones), 3) if all_tones else 0,
        "total_articles_with_tone": len(all_tones),
        "sources": source_summaries,
    }


# ── Attention Asymmetry endpoints ─────────────────────────────────────


@app.get("/api/asymmetry")
def get_asymmetry():
    """Return attention asymmetry data: coverage by country vs crisis
    theme intensity, revealing overcovered and underreported regions.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Coverage volume per country
            cur.execute(
                """
                SELECT
                    loc->>'country_code' AS code,
                    COUNT(DISTINCT a.id) AS article_count,
                    round(avg((a.tone->>'tone')::float)::numeric, 3) AS avg_tone
                FROM articles a, jsonb_array_elements(a.locations) AS loc
                WHERE a.locations IS NOT NULL
                  AND loc->>'country_code' IS NOT NULL
                  AND a.tone IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
                """
            )
            countries = cur.fetchall()

            # Crisis-tagged articles per country (themes indicating conflict,
            # disaster, or humanitarian issues)
            cur.execute(
                """
                SELECT
                    loc->>'country_code' AS code,
                    COUNT(DISTINCT a.id) AS crisis_articles
                FROM articles a,
                    jsonb_array_elements(a.locations) AS loc,
                    jsonb_array_elements(a.themes) AS theme
                WHERE a.locations IS NOT NULL
                  AND a.themes IS NOT NULL
                  AND loc->>'country_code' IS NOT NULL
                  AND (
                    theme->>'theme' ILIKE '%%CONFLICT%%'
                    OR theme->>'theme' ILIKE '%%CRISIS%%'
                    OR theme->>'theme' ILIKE '%%DISASTER%%'
                    OR theme->>'theme' ILIKE '%%KILL%%'
                    OR theme->>'theme' ILIKE '%%TERROR%%'
                    OR theme->>'theme' ILIKE '%%ARMED%%'
                    OR theme->>'theme' ILIKE '%%WAR%%'
                    OR theme->>'theme' ILIKE '%%REFUGEE%%'
                    OR theme->>'theme' ILIKE '%%FAMINE%%'
                    OR theme->>'theme' ILIKE '%%WOUND%%'
                  )
                GROUP BY 1
                """
            )
            crisis_rows = cur.fetchall()

            # Top overcovered clusters (most articles, any country)
            cur.execute(
                """
                SELECT id, representative_title, article_count,
                       first_article_at, last_article_at
                FROM clusters
                WHERE is_active = true
                ORDER BY article_count DESC
                LIMIT 10
                """
            )
            overcovered = cur.fetchall()

            # Underreported: clusters where MOST articles are crisis-tagged
            # but total coverage is low
            cur.execute(
                """
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       crisis.crisis_count,
                       round(crisis.crisis_count::numeric / c.article_count, 2) AS crisis_pct
                FROM clusters c
                JOIN (
                    SELECT cm.cluster_id, COUNT(DISTINCT a.id) AS crisis_count
                    FROM cluster_memberships cm
                    JOIN articles a ON a.id = cm.article_id,
                        jsonb_array_elements(a.themes) AS theme
                    WHERE a.themes IS NOT NULL
                      AND (
                        theme->>'theme' ILIKE '%%CONFLICT%%'
                        OR theme->>'theme' ILIKE '%%KILL%%'
                        OR theme->>'theme' ILIKE '%%TERROR%%'
                        OR theme->>'theme' ILIKE '%%ARMED%%'
                        OR theme->>'theme' ILIKE '%%REFUGEE%%'
                        OR theme->>'theme' ILIKE '%%FAMINE%%'
                        OR theme->>'theme' ILIKE '%%WOUND%%'
                      )
                    GROUP BY cm.cluster_id
                    HAVING COUNT(DISTINCT a.id) >= 3
                ) crisis ON crisis.cluster_id = c.id
                WHERE c.is_active = true
                  AND c.article_count BETWEEN 5 AND 40
                  AND crisis.crisis_count::float / c.article_count >= 0.6
                ORDER BY crisis.crisis_count DESC, c.article_count ASC
                LIMIT 20
                """
            )
            underreported = cur.fetchall()

    # Build crisis map
    crisis_map = {r["code"]: r["crisis_articles"] for r in crisis_rows}

    # Total articles for percentage calculation
    total_articles = sum(c["article_count"] for c in countries)

    country_data = []
    for c in countries:
        crisis_count = crisis_map.get(c["code"], 0)
        country_data.append(
            {
                "code": c["code"],
                "article_count": c["article_count"],
                "coverage_pct": round(c["article_count"] / total_articles * 100, 3)
                if total_articles
                else 0,
                "crisis_articles": crisis_count,
                "crisis_ratio": round(crisis_count / c["article_count"], 3)
                if c["article_count"]
                else 0,
                "avg_tone": float(c["avg_tone"]) if c["avg_tone"] is not None else 0,
            }
        )

    return {
        "total_articles": total_articles,
        "total_countries": len(countries),
        "countries": country_data,
        "overcovered": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in overcovered
        ],
        "underreported": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in underreported
        ],
    }


# ── Gravity Map endpoints ────────────────────────────────


@app.get("/api/gravity/graph")
def gravity_graph(
    min_weight: int = Query(50, ge=10, le=5000, description="Minimum co-mention count for edges"),
    limit_edges: int = Query(80, ge=10, le=300, description="Max edges to return"),
):
    """Return country co-mention graph for the geopolitical gravity map.

    Nodes = countries (with article counts and avg tone).
    Edges = co-mention relationships (with weight = co-mention count).
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Edges from materialized view
            cur.execute(
                """
                SELECT source_code AS source, target_code AS target, weight
                FROM mv_country_comentions
                WHERE weight >= %s
                ORDER BY weight DESC
                LIMIT %s
                """,
                (min_weight, limit_edges),
            )
            edges = cur.fetchall()

            # Collect all country codes that appear in edges
            codes = set()
            for e in edges:
                codes.add(e["source"])
                codes.add(e["target"])

            if not codes:
                return {"nodes": [], "edges": []}

            # Nodes from materialized view
            code_list = list(codes)
            cur.execute(
                """
                SELECT code, article_count, avg_tone
                FROM mv_country_stats
                WHERE code = ANY(%s)
                """,
                (code_list,),
            )
            node_rows = cur.fetchall()

    nodes = [
        {
            "id": n["code"],
            "article_count": n["article_count"],
            "avg_tone": float(n["avg_tone"]) if n["avg_tone"] is not None else 0,
        }
        for n in node_rows
    ]

    edge_list = [
        {
            "source": e["source"],
            "target": e["target"],
            "weight": e["weight"],
        }
        for e in edges
    ]

    return {"nodes": nodes, "edges": edge_list}


@app.get("/api/gravity/country/{code}")
def gravity_country_detail(code: str):
    """Detail view for a single country: top connected countries,
    top clusters, and tone breakdown."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Country stats from materialized view
            cur.execute(
                "SELECT code, article_count, avg_tone FROM mv_country_stats WHERE code = %s",
                (code,),
            )
            stats = cur.fetchone()
            if not stats or stats["article_count"] == 0:
                raise HTTPException(status_code=404, detail="Country not found")

            # Top connected countries from materialized view
            cur.execute(
                """
                SELECT
                    CASE WHEN source_code = %s THEN target_code ELSE source_code END
                        AS connected_code,
                    weight
                FROM mv_country_comentions
                WHERE source_code = %s OR target_code = %s
                ORDER BY weight DESC
                LIMIT 15
                """,
                (code, code, code),
            )
            connections = cur.fetchall()

            # Top clusters mentioning this country
            cur.execute(
                """
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       COUNT(*) AS mentions
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id,
                    jsonb_array_elements(a.locations) AS loc
                WHERE a.locations IS NOT NULL
                  AND loc->>'country_code' = %s
                  AND c.is_active = true
                GROUP BY c.id
                ORDER BY mentions DESC
                LIMIT 10
                """,
                (code,),
            )
            top_clusters = cur.fetchall()

    return {
        "code": code,
        "article_count": stats["article_count"],
        "avg_tone": float(stats["avg_tone"]) if stats["avg_tone"] is not None else 0,
        "connections": [
            {"code": c["connected_code"], "weight": c["weight"]}
            for c in connections
        ],
        "top_clusters": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "mentions": c["mentions"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in top_clusters
        ],
    }


# ── Source DNA endpoints ─────────────────────────────────────────────


@app.get("/api/sources/fingerprints")
def source_fingerprints(
    limit: int = Query(50, ge=1, le=200, description="Max sources to return"),
    sort: str = Query("articles", description="Sort: articles, tone_positive, tone_negative"),
    min_articles: int = Query(10, ge=1, description="Minimum articles to include source"),
):
    """Per-source fingerprints: article count, avg tone, top themes, top countries."""
    from gdelt_event_pipeline.storage.database import get_pool

    order_clause = {
        "tone_positive": "avg_tone DESC",
        "tone_negative": "avg_tone ASC",
    }.get(sort, "article_count DESC")

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT
                    COALESCE(source_common_name, domain) AS source_name,
                    domain,
                    count(*) AS article_count,
                    avg((tone->>'tone')::float) FILTER (WHERE tone->>'tone' IS NOT NULL)
                        AS avg_tone,
                    avg((tone->>'positive_score')::float)
                        FILTER (WHERE tone->>'positive_score' IS NOT NULL) AS avg_positive,
                    avg((tone->>'negative_score')::float)
                        FILTER (WHERE tone->>'negative_score' IS NOT NULL) AS avg_negative,
                    avg((tone->>'polarity')::float)
                        FILTER (WHERE tone->>'polarity' IS NOT NULL) AS avg_polarity,
                    min(gdelt_timestamp) AS first_article,
                    max(gdelt_timestamp) AS last_article
                FROM articles
                WHERE title IS NOT NULL
                GROUP BY COALESCE(source_common_name, domain), domain
                HAVING count(*) >= %s
                ORDER BY {order_clause}
                LIMIT %s
                """,
                (min_articles, limit),
            )
            sources = cur.fetchall()

            if not sources:
                return []

            # Get top themes per source (top 8)
            source_domains = [s["domain"] for s in sources]
            cur.execute(
                """
                SELECT domain,
                       t->>'theme' AS theme,
                       count(*) AS cnt
                FROM articles,
                     jsonb_array_elements(themes) AS t
                WHERE domain = ANY(%s)
                  AND t->>'theme' IS NOT NULL
                  AND t->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                  AND t->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                  AND t->>'theme' NOT LIKE 'TAX_FNCACT%%'
                GROUP BY domain, t->>'theme'
                ORDER BY domain, count(*) DESC
                """,
                (source_domains,),
            )
            theme_rows = cur.fetchall()

            # Get top countries per source (top 5)
            cur.execute(
                """
                SELECT domain,
                       loc->>'country_code' AS country_code,
                       count(*) AS cnt
                FROM articles,
                     jsonb_array_elements(locations) AS loc
                WHERE domain = ANY(%s)
                  AND loc->>'country_code' IS NOT NULL
                GROUP BY domain, loc->>'country_code'
                ORDER BY domain, count(*) DESC
                """,
                (source_domains,),
            )
            country_rows = cur.fetchall()

    # Build theme map (top 8 per domain)
    theme_map: dict[str, list[dict]] = defaultdict(list)
    for row in theme_rows:
        d = row["domain"]
        if len(theme_map[d]) < 8:
            category = _categorize_themes([row["theme"]])
            theme_map[d].append(
                {"theme": row["theme"], "count": row["cnt"], "category": category}
            )

    # Build country map (top 5 per domain)
    country_map: dict[str, list[dict]] = defaultdict(list)
    for row in country_rows:
        d = row["domain"]
        if len(country_map[d]) < 5:
            country_map[d].append(
                {"country_code": row["country_code"], "count": row["cnt"]}
            )

    results = []
    for s in sources:
        results.append(
            {
                "source_name": s["source_name"],
                "domain": s["domain"],
                "article_count": s["article_count"],
                "avg_tone": round(s["avg_tone"], 3) if s["avg_tone"] else 0,
                "avg_positive": round(s["avg_positive"], 3) if s["avg_positive"] else 0,
                "avg_negative": round(s["avg_negative"], 3) if s["avg_negative"] else 0,
                "avg_polarity": round(s["avg_polarity"], 3) if s["avg_polarity"] else 0,
                "first_article": s["first_article"],
                "last_article": s["last_article"],
                "top_themes": theme_map.get(s["domain"], []),
                "top_countries": country_map.get(s["domain"], []),
            }
        )

    return results


@app.get("/api/sources/{domain}/detail")
def source_detail(domain: str):
    """Detailed fingerprint for a single source domain."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Basic stats
            cur.execute(
                """
                SELECT
                    COALESCE(source_common_name, domain) AS source_name,
                    domain,
                    count(*) AS article_count,
                    avg((tone->>'tone')::float) FILTER (WHERE tone->>'tone' IS NOT NULL)
                        AS avg_tone,
                    avg((tone->>'positive_score')::float)
                        FILTER (WHERE tone->>'positive_score' IS NOT NULL) AS avg_positive,
                    avg((tone->>'negative_score')::float)
                        FILTER (WHERE tone->>'negative_score' IS NOT NULL) AS avg_negative,
                    avg((tone->>'polarity')::float)
                        FILTER (WHERE tone->>'polarity' IS NOT NULL) AS avg_polarity,
                    min(gdelt_timestamp) AS first_article,
                    max(gdelt_timestamp) AS last_article
                FROM articles
                WHERE domain = %s AND title IS NOT NULL
                GROUP BY COALESCE(source_common_name, domain), domain
                """,
                (domain,),
            )
            stats = cur.fetchone()
            if not stats:
                raise HTTPException(status_code=404, detail="Source not found")

            # Theme distribution (all themes)
            cur.execute(
                """
                SELECT t->>'theme' AS theme, count(*) AS cnt
                FROM articles, jsonb_array_elements(themes) AS t
                WHERE domain = %s
                  AND t->>'theme' IS NOT NULL
                  AND t->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                  AND t->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                  AND t->>'theme' NOT LIKE 'TAX_FNCACT%%'
                GROUP BY t->>'theme'
                ORDER BY count(*) DESC
                LIMIT 20
                """,
                (domain,),
            )
            themes = cur.fetchall()

            # Country distribution
            cur.execute(
                """
                SELECT loc->>'country_code' AS country_code, count(*) AS cnt
                FROM articles, jsonb_array_elements(locations) AS loc
                WHERE domain = %s AND loc->>'country_code' IS NOT NULL
                GROUP BY loc->>'country_code'
                ORDER BY count(*) DESC
                LIMIT 15
                """,
                (domain,),
            )
            countries = cur.fetchall()

            # Category distribution
            cur.execute(
                """
                SELECT t->>'theme' AS theme
                FROM articles, jsonb_array_elements(themes) AS t
                WHERE domain = %s AND t->>'theme' IS NOT NULL
                """,
                (domain,),
            )
            all_theme_rows = cur.fetchall()

            # Tone over time (by day)
            cur.execute(
                """
                SELECT date_trunc('day', gdelt_timestamp) AS day,
                       avg((tone->>'tone')::float) AS avg_tone,
                       count(*) AS article_count
                FROM articles
                WHERE domain = %s
                  AND tone->>'tone' IS NOT NULL
                  AND gdelt_timestamp IS NOT NULL
                GROUP BY date_trunc('day', gdelt_timestamp)
                ORDER BY day
                """,
                (domain,),
            )
            tone_timeline = cur.fetchall()

            # Recent articles (last 10)
            cur.execute(
                """
                SELECT title, url, gdelt_timestamp,
                       (tone->>'tone')::float AS tone_score
                FROM articles
                WHERE domain = %s AND title IS NOT NULL
                ORDER BY gdelt_timestamp DESC
                LIMIT 10
                """,
                (domain,),
            )
            recent = cur.fetchall()

    # Build category breakdown
    cat_counts: dict[str, int] = defaultdict(int)
    for row in all_theme_rows:
        cat = _categorize_themes([row["theme"]])
        cat_counts[cat] += 1
    total_themes = sum(cat_counts.values()) or 1
    categories = [
        {"category": cat, "count": cnt, "pct": round(cnt / total_themes * 100, 1)}
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "source_name": stats["source_name"],
        "domain": stats["domain"],
        "article_count": stats["article_count"],
        "avg_tone": round(stats["avg_tone"], 3) if stats["avg_tone"] else 0,
        "avg_positive": round(stats["avg_positive"], 3) if stats["avg_positive"] else 0,
        "avg_negative": round(stats["avg_negative"], 3) if stats["avg_negative"] else 0,
        "avg_polarity": round(stats["avg_polarity"], 3) if stats["avg_polarity"] else 0,
        "first_article": stats["first_article"],
        "last_article": stats["last_article"],
        "themes": [
            {"theme": t["theme"], "count": t["cnt"], "category": _categorize_themes([t["theme"]])}
            for t in themes
        ],
        "countries": [{"country_code": c["country_code"], "count": c["cnt"]} for c in countries],
        "categories": categories,
        "tone_timeline": [
            {
                "day": t["day"].isoformat() if t["day"] else None,
                "avg_tone": round(t["avg_tone"], 3),
                "article_count": t["article_count"],
            }
            for t in tone_timeline
        ],
        "recent_articles": [dict(r) for r in recent],
    }


# ── Story Propagation endpoints ──────────────────────────────────────


@app.get("/api/propagation/stories")
def propagation_stories(
    limit: int = Query(20, ge=1, le=50, description="Max clusters to return"),
    min_sources: int = Query(3, ge=2, description="Min distinct sources"),
):
    """Top stories suitable for propagation analysis (multi-source clusters)."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       count(DISTINCT a.domain) AS source_count,
                       EXTRACT(EPOCH FROM (c.last_article_at - c.first_article_at)) / 3600
                           AS span_hours
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id
                WHERE c.is_active = true
                  AND c.article_count >= 5
                  AND a.title IS NOT NULL
                GROUP BY c.id
                HAVING count(DISTINCT a.domain) >= %s
                ORDER BY c.article_count DESC
                LIMIT %s
                """,
                (min_sources, limit),
            )
            stories = cur.fetchall()

    return [
        {
            "id": str(s["id"]),
            "title": s["representative_title"],
            "article_count": s["article_count"],
            "source_count": s["source_count"],
            "span_hours": round(s["span_hours"], 1) if s["span_hours"] else 0,
            "first_article_at": s["first_article_at"],
            "last_article_at": s["last_article_at"],
        }
        for s in stories
    ]


@app.get("/api/propagation/{cluster_id}")
def propagation_timeline(cluster_id: str):
    """Timeline of how a story propagated across sources over time."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Verify cluster exists
            cur.execute(
                "SELECT id, representative_title, article_count, "
                "first_article_at, last_article_at "
                "FROM clusters WHERE id = %s",
                (cluster_id,),
            )
            cluster = cur.fetchone()
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            # All articles in chronological order with source info
            cur.execute(
                """
                SELECT a.title, a.url, a.domain,
                       COALESCE(a.source_common_name, a.domain) AS source_name,
                       a.gdelt_timestamp,
                       (a.tone->>'tone')::float AS tone_score,
                       a.locations
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = %s
                  AND a.title IS NOT NULL
                ORDER BY a.gdelt_timestamp ASC
                """,
                (cluster_id,),
            )
            articles = cur.fetchall()

    if not articles:
        raise HTTPException(status_code=404, detail="No articles found for cluster")

    # Build timeline events
    events = []
    seen_sources = set()
    source_first_seen: dict[str, int] = {}

    for i, a in enumerate(articles):
        domain = a["domain"]
        is_first = domain not in seen_sources
        if is_first:
            source_first_seen[domain] = i
            seen_sources.add(domain)

        # Extract primary location
        loc_name = None
        if a["locations"] and isinstance(a["locations"], list) and a["locations"]:
            loc_name = a["locations"][0].get("name")

        events.append(
            {
                "title": a["title"],
                "url": a["url"],
                "domain": domain,
                "source_name": a["source_name"],
                "timestamp": a["gdelt_timestamp"],
                "tone": round(a["tone_score"], 2) if a["tone_score"] is not None else None,
                "location": loc_name,
                "is_first_from_source": is_first,
                "order": i,
            }
        )

    # Source summary: when each source first picked up the story
    source_summary = []
    for domain, first_idx in sorted(source_first_seen.items(), key=lambda x: x[1]):
        arts = [e for e in events if e["domain"] == domain]
        source_summary.append(
            {
                "domain": domain,
                "source_name": arts[0]["source_name"],
                "first_timestamp": arts[0]["timestamp"],
                "article_count": len(arts),
                "avg_tone": round(
                    sum(a["tone"] for a in arts if a["tone"] is not None)
                    / max(1, sum(1 for a in arts if a["tone"] is not None)),
                    2,
                ),
                "order": first_idx,
            }
        )

    # Hourly histogram
    if articles:
        first_ts = articles[0]["gdelt_timestamp"]
        if first_ts.tzinfo is None:
            first_ts = first_ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        hourly: dict[int, int] = defaultdict(int)
        for a in articles:
            ts = a["gdelt_timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=first_ts.tzinfo)
            hour = int((ts - first_ts).total_seconds() / 3600)
            hourly[hour] += 1
        max_hour = max(hourly.keys()) if hourly else 0
        histogram = [{"hour": h, "count": hourly.get(h, 0)} for h in range(max_hour + 1)]
    else:
        histogram = []

    return {
        "cluster_id": str(cluster["id"]),
        "title": cluster["representative_title"],
        "article_count": cluster["article_count"],
        "first_article_at": cluster["first_article_at"],
        "last_article_at": cluster["last_article_at"],
        "source_count": len(source_summary),
        "timeline": events,
        "sources": source_summary,
        "hourly_histogram": histogram,
    }


# ── Topic Velocity endpoints ─────────────────────────────────────────


@app.get("/api/velocity/topics")
def velocity_topics(
    hours: int = Query(48, ge=6, le=168, description="Lookback window in hours"),
    limit: int = Query(30, ge=5, le=100, description="Max topics to return"),
):
    """Trending topics with velocity (acceleration/deceleration) over time windows."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            half = hours // 2

            def _velocity_query(order: str, min_recent: int, min_older: int):
                return f"""
                WITH ref AS (SELECT max(gdelt_timestamp) AS max_ts FROM articles),
                recent AS (
                    SELECT th->>'theme' AS theme, count(*) AS cnt
                    FROM articles, jsonb_array_elements(themes) AS th, ref
                    WHERE gdelt_timestamp > ref.max_ts - interval '{half} hours'
                      AND th->>'theme' IS NOT NULL
                      AND th->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                      AND th->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                      AND th->>'theme' NOT LIKE 'TAX_FNCACT%%'
                    GROUP BY th->>'theme'
                    HAVING count(*) >= {min_recent}
                ),
                older AS (
                    SELECT th->>'theme' AS theme, count(*) AS cnt
                    FROM articles, jsonb_array_elements(themes) AS th, ref
                    WHERE gdelt_timestamp >= ref.max_ts - interval '{hours} hours'
                      AND gdelt_timestamp <= ref.max_ts - interval '{half} hours'
                      AND th->>'theme' IS NOT NULL
                      AND th->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                      AND th->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                      AND th->>'theme' NOT LIKE 'TAX_FNCACT%%'
                    GROUP BY th->>'theme'
                    HAVING count(*) >= {min_older}
                )
                SELECT
                    COALESCE(r.theme, o.theme) AS theme,
                    COALESCE(r.cnt, 0) AS recent_count,
                    COALESCE(o.cnt, 0) AS older_count,
                    COALESCE(r.cnt, 0) + COALESCE(o.cnt, 0) AS total_count,
                    CASE
                        WHEN COALESCE(o.cnt, 0) = 0 AND COALESCE(r.cnt, 0) > 0 THEN 100.0
                        WHEN COALESCE(o.cnt, 0) > 0
                            THEN round(
                                ((COALESCE(r.cnt, 0) - o.cnt)::numeric / o.cnt * 100), 1
                            )
                        ELSE 0
                    END AS velocity_pct
                FROM recent r
                FULL OUTER JOIN older o ON r.theme = o.theme
                WHERE COALESCE(r.cnt, 0) + COALESCE(o.cnt, 0) >= 5
                ORDER BY velocity_pct {order}
                LIMIT %s
                """

            cur.execute(_velocity_query("DESC", 3, 1), (limit,))
            rising = cur.fetchall()

            cur.execute(_velocity_query("ASC", 2, 5), (limit,))
            declining = cur.fetchall()

    def format_topic(row):
        theme = row["theme"]
        return {
            "theme": theme,
            "category": _categorize_themes([theme]),
            "recent_count": row["recent_count"],
            "older_count": row["older_count"],
            "total_count": row["total_count"],
            "velocity_pct": float(row["velocity_pct"]),
        }

    return {
        "window_hours": hours,
        "rising": [format_topic(r) for r in rising],
        "declining": [format_topic(r) for r in declining],
    }


@app.get("/api/velocity/timeline")
def velocity_timeline(
    theme: str = Query(..., description="Theme to get timeline for"),
    hours: int = Query(72, ge=6, le=168, description="Lookback window"),
):
    """Hourly article count for a specific theme."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH ref AS (SELECT max(gdelt_timestamp) AS max_ts FROM articles)
                SELECT date_trunc('hour', gdelt_timestamp) AS hour,
                       count(*) AS cnt,
                       avg((tone->>'tone')::float) FILTER (WHERE tone->>'tone' IS NOT NULL)
                           AS avg_tone
                FROM articles, jsonb_array_elements(themes) AS th, ref
                WHERE th->>'theme' = %s
                  AND gdelt_timestamp >= ref.max_ts - interval '%s hours'
                GROUP BY date_trunc('hour', gdelt_timestamp)
                ORDER BY hour
                """,
                (theme, hours),
            )
            rows = cur.fetchall()

            cur.execute(
                """
                WITH ref AS (SELECT max(gdelt_timestamp) AS max_ts FROM articles)
                SELECT a.title, a.url, a.domain, a.gdelt_timestamp,
                       (a.tone->>'tone')::float AS tone_score
                FROM articles a, jsonb_array_elements(a.themes) AS th, ref
                WHERE th->>'theme' = %s
                  AND a.title IS NOT NULL
                  AND a.gdelt_timestamp >= ref.max_ts - interval '%s hours'
                ORDER BY a.gdelt_timestamp DESC
                LIMIT 10
                """,
                (theme, hours),
            )
            articles = cur.fetchall()

    return {
        "theme": theme,
        "category": _categorize_themes([theme]),
        "timeline": [
            {
                "hour": r["hour"].isoformat() if r["hour"] else None,
                "count": r["cnt"],
                "avg_tone": round(r["avg_tone"], 3) if r["avg_tone"] else None,
            }
            for r in rows
        ],
        "recent_articles": [dict(a) for a in articles],
    }



# ── Runner ────────────────────────────────────────────────────────────


def run() -> None:
    """Convenience entry point: uvicorn gdelt_event_pipeline.api.app:app"""
    import uvicorn

    uvicorn.run("gdelt_event_pipeline.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
