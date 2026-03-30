"""Semantic vector search using pgvector."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from gdelt_event_pipeline.query.filters import build_filter_clauses
from gdelt_event_pipeline.query.models import SearchFilters
from gdelt_event_pipeline.storage.database import get_pool


def search_articles_by_vector(
    embedding: list[float],
    *,
    limit: int = 40,
    filters: SearchFilters | None = None,
) -> list[dict[str, Any]]:
    """Find articles nearest to an embedding vector.

    Returns articles ordered by cosine distance (closest first),
    each with a 'cosine_distance' field.
    """
    filter_sql, filter_params = build_filter_clauses(filters)

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = f"""
                SELECT *, (embedding <=> %s::vector) AS cosine_distance
                FROM articles
                WHERE embedding IS NOT NULL
                  AND title IS NOT NULL
                  {filter_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [embedding, *filter_params, embedding, limit]
            cur.execute(query, params)
            return cur.fetchall()


def search_clusters_by_vector(
    embedding: list[float],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find clusters nearest to an embedding vector.

    Returns clusters ordered by cosine distance (closest first),
    each with a 'cosine_distance' field.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT *, (centroid_embedding <=> %s::vector) AS cosine_distance
                FROM clusters
                WHERE is_active = true
                  AND centroid_embedding IS NOT NULL
                ORDER BY centroid_embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, limit),
            )
            return cur.fetchall()
