"""Cluster browsing endpoint router."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from gdelt_event_pipeline.api.routers._helpers import split_csv, strip_internal_fields
from gdelt_event_pipeline.query.filters import build_filter_clauses
from gdelt_event_pipeline.query.models import SearchFilters
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
    domain: str | None = Query(
        None,
        description=(
            "Filter by source domain (comma-separated). Soft match: 'corriere.it' "
            "also matches 'video.corriere.it'."
        ),
    ),
    source: str | None = Query(
        None, description="Filter by canonical source slug (comma-separated)"
    ),
    date_from: datetime | None = Query(None, description="Start date (ISO format)"),  # noqa: B008
    date_to: datetime | None = Query(None, description="End date (ISO format)"),  # noqa: B008
):
    """List active clusters, optionally filtered by article metadata."""
    filters = SearchFilters(
        locations=split_csv(location),
        persons=split_csv(person),
        organizations=split_csv(org),
        themes=split_csv(theme),
        domains=split_csv(domain),
        sources=split_csv(source),
        date_from=date_from,
        date_to=date_to,
    )
    has_filters = any(getattr(filters, f) for f in filters.__dataclass_fields__)

    if not has_filters:
        rows = get_active_clusters(limit=limit, sort=sort)
        return [strip_internal_fields(row) for row in rows]

    from gdelt_event_pipeline.storage.database import get_pool

    order_clause = {
        "articles": "c.article_count DESC",
        "oldest": "c.first_article_at ASC NULLS LAST",
    }.get(sort, "c.last_article_at DESC NULLS LAST")

    # build_filter_clauses returns " AND ..." prefixed; the inner subquery
    # needs a real WHERE, so we seed it with TRUE.
    filter_sql, filter_params = build_filter_clauses(filters, table_alias="a")
    params: list = list(filter_params)
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
                    WHERE TRUE{filter_sql}
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
