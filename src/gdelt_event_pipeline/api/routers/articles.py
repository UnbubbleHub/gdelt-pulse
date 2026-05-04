"""Articles and stats endpoint router."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query

from gdelt_event_pipeline.api.routers._helpers import split_csv, strip_internal_fields
from gdelt_event_pipeline.storage.articles import get_recent_articles

router = APIRouter()


@router.get("/api/articles", response_model=list[dict[str, Any]])
def list_articles(
    limit: int = Query(50, ge=1, le=200, description="Max articles to return"),
    location: str | None = Query(None, description="Filter by location"),
    person: str | None = Query(None, description="Filter by person"),
    org: str | None = Query(None, description="Filter by organization"),
    theme: str | None = Query(None, description="Filter by GDELT theme"),
    domain: str | None = Query(None, description="Filter by source domain"),
    source: str | None = Query(None, description="Filter by source name"),
    date_from: datetime | None = Query(None, description="Start date (ISO)"),  # noqa: B008
    date_to: datetime | None = Query(None, description="End date (ISO)"),  # noqa: B008
):
    """List recent articles, newest first, with optional filters."""
    locations = split_csv(location)
    persons = split_csv(person)
    organizations = split_csv(org)
    themes = split_csv(theme)
    domains = split_csv(domain)
    sources = split_csv(source)

    has_filters = any(
        [locations, persons, organizations, themes, domains, sources, date_from, date_to]
    )

    if not has_filters:
        rows = get_recent_articles(limit=limit)
        return [strip_internal_fields(row) for row in rows]

    from gdelt_event_pipeline.storage.database import get_pool

    conditions: list[str] = ["a.title IS NOT NULL"]
    params: list[Any] = []

    if locations:
        conditions.append("a.locations::text ILIKE %s")
        params.append(f"%{locations[0]}%")
    if persons:
        conditions.append("a.persons::text ILIKE %s")
        params.append(f"%{persons[0]}%")
    if organizations:
        conditions.append("a.organizations::text ILIKE %s")
        params.append(f"%{organizations[0]}%")
    if themes:
        conditions.append("a.themes::text ILIKE %s")
        params.append(f"%{themes[0]}%")
    if domains:
        conditions.append("a.domain ILIKE %s")
        params.append(f"%{domains[0]}%")
    if sources:
        conditions.append("(a.source_common_name ILIKE %s OR a.canonical_source ILIKE %s)")
        params.extend([f"%{sources[0]}%", f"%{sources[0]}%"])
    if date_from:
        conditions.append("a.gdelt_timestamp >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("a.gdelt_timestamp <= %s")
        params.append(date_to)

    where = " AND ".join(conditions)
    params.append(limit)

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT a.* FROM articles a WHERE {where}"
                " ORDER BY a.gdelt_timestamp DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()

    return [strip_internal_fields(row) for row in rows]


@router.get("/api/stats")
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
