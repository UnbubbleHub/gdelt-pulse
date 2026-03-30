"""Full-text keyword search using PostgreSQL tsvector."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from gdelt_event_pipeline.query.filters import build_filter_clauses
from gdelt_event_pipeline.query.models import SearchFilters
from gdelt_event_pipeline.storage.database import get_pool


def search_articles_by_keyword(
    query_text: str,
    *,
    limit: int = 40,
    filters: SearchFilters | None = None,
) -> list[dict[str, Any]]:
    """Search articles by keyword using PostgreSQL full-text search.

    Uses websearch_to_tsquery which supports natural query syntax:
    - "earthquake Turkey" → AND of both terms
    - '"exact phrase"' → phrase match
    - "-exclude" → NOT
    - "word1 OR word2" → OR

    Returns articles ordered by ts_rank (best match first),
    each with a 'rank_score' field.
    """
    filter_sql, filter_params = build_filter_clauses(filters)

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = f"""
                SELECT *,
                       ts_rank(title_tsv, websearch_to_tsquery('english', %s)) AS rank_score
                FROM articles
                WHERE title_tsv @@ websearch_to_tsquery('english', %s)
                  {filter_sql}
                ORDER BY rank_score DESC
                LIMIT %s
            """
            params = [query_text, query_text, *filter_params, limit]
            cur.execute(query, params)
            return cur.fetchall()
