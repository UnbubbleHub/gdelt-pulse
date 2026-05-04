"""Articles and stats endpoint router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from gdelt_event_pipeline.api.routers._helpers import strip_internal_fields
from gdelt_event_pipeline.storage.articles import get_recent_articles

router = APIRouter()


@router.get("/api/articles", response_model=list[dict[str, Any]])
def list_articles(
    limit: int = Query(50, ge=1, le=200, description="Max articles to return"),
):
    """List recent articles, newest first."""
    rows = get_recent_articles(limit=limit)
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
