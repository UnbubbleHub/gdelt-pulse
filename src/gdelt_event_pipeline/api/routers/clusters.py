"""Cluster browsing endpoint router."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from gdelt_event_pipeline.api.routers._helpers import split_csv, strip_internal_fields
from gdelt_event_pipeline.storage.clusters import (
    get_active_clusters,
    get_cluster_articles,
    get_cluster_by_id,
)


class ClusterDetailOut(BaseModel):
    cluster: dict[str, Any]
    articles: list[dict[str, Any]]


router = APIRouter()


@router.get("/api/clusters", response_model=list[dict[str, Any]])
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
    locations = split_csv(location)
    persons = split_csv(person)
    organizations = split_csv(org)
    themes = split_csv(theme)
    domains = split_csv(domain)
    sources = split_csv(source)

    has_filters = any(
        [
            locations,
            persons,
            organizations,
            themes,
            domains,
            sources,
            date_from,
            date_to,
        ]
    )

    if not has_filters:
        rows = get_active_clusters(limit=limit, sort=sort)
        return [strip_internal_fields(row) for row in rows]

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

    return [strip_internal_fields(row) for row in rows]


@router.get("/api/clusters/{cluster_id}", response_model=ClusterDetailOut)
def get_cluster_detail(cluster_id: str):
    """Get a single cluster and its member articles."""
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    articles = get_cluster_articles(cluster_id)

    return ClusterDetailOut(
        cluster=strip_internal_fields(cluster),
        articles=[strip_internal_fields(a) for a in articles],
    )
