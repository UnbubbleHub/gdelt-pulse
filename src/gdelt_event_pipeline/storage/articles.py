"""Article storage operations against PostgreSQL."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from gdelt_event_pipeline.storage.database import get_pool

# 14 article fields + first_seen_at/last_seen_at reusing gdelt_timestamp = 16 placeholders per row
_UPSERT_COLUMNS = (
    "gkg_record_id",
    "gdelt_timestamp",
    "url",
    "canonical_url",
    "domain",
    "source_common_name",
    "canonical_source",
    "title",
    "themes",
    "locations",
    "organizations",
    "persons",
    "all_names",
    "tone",
)
_ROW_PLACEHOLDERS = "(" + ", ".join(["%s"] * (len(_UPSERT_COLUMNS) + 2)) + ")"

_UPSERT_CONFLICT_CLAUSE = """
ON CONFLICT (canonical_url) DO UPDATE SET
    themes         = CASE
                        WHEN COALESCE(jsonb_array_length(EXCLUDED.themes), 0)
                           > COALESCE(jsonb_array_length(articles.themes), 0)
                        THEN EXCLUDED.themes
                        ELSE articles.themes
                     END,
    locations      = CASE
                        WHEN COALESCE(jsonb_array_length(EXCLUDED.locations), 0)
                           > COALESCE(jsonb_array_length(articles.locations), 0)
                        THEN EXCLUDED.locations
                        ELSE articles.locations
                     END,
    organizations  = CASE
                        WHEN COALESCE(jsonb_array_length(EXCLUDED.organizations), 0)
                           > COALESCE(jsonb_array_length(articles.organizations), 0)
                        THEN EXCLUDED.organizations
                        ELSE articles.organizations
                     END,
    persons        = CASE
                        WHEN COALESCE(jsonb_array_length(EXCLUDED.persons), 0)
                           > COALESCE(jsonb_array_length(articles.persons), 0)
                        THEN EXCLUDED.persons
                        ELSE articles.persons
                     END,
    all_names      = CASE
                        WHEN COALESCE(jsonb_array_length(EXCLUDED.all_names), 0)
                           > COALESCE(jsonb_array_length(articles.all_names), 0)
                        THEN EXCLUDED.all_names
                        ELSE articles.all_names
                     END,
    tone           = COALESCE(EXCLUDED.tone, articles.tone),
    first_seen_at  = LEAST(articles.first_seen_at, EXCLUDED.first_seen_at),
    last_seen_at   = GREATEST(articles.last_seen_at, EXCLUDED.last_seen_at),
    updated_at     = now()
"""


def _flatten_params(article: dict[str, Any]) -> list[Any]:
    """Return positional params in the order the INSERT expects.

    first_seen_at and last_seen_at are both seeded from gdelt_timestamp.
    """
    base = [article.get(col) for col in _UPSERT_COLUMNS]
    ts = article.get("gdelt_timestamp")
    base.extend([ts, ts])
    return base


def upsert_articles(
    articles: list[dict[str, Any]],
    *,
    chunk_size: int = 500,
) -> int:
    """Batch-upsert articles using a multi-row INSERT per chunk.

    The caller MUST deduplicate by canonical_url before calling; a single
    statement cannot touch the same conflict target twice (Postgres raises
    "ON CONFLICT DO UPDATE command cannot affect row a second time").

    Returns the number of rows written.  Raises on database errors — chunks
    are atomic: a single failing chunk rolls back and aborts the call.
    """
    if not articles:
        return 0

    pool = get_pool()
    total_written = 0
    columns_sql = ", ".join([*_UPSERT_COLUMNS, "first_seen_at", "last_seen_at"])

    for start in range(0, len(articles), chunk_size):
        chunk = articles[start : start + chunk_size]
        values_sql = ", ".join([_ROW_PLACEHOLDERS] * len(chunk))
        params: list[Any] = []
        for article in chunk:
            params.extend(_flatten_params(article))

        statement = (
            f"INSERT INTO articles ({columns_sql}) VALUES {values_sql}"
            f"{_UPSERT_CONFLICT_CLAUSE}RETURNING id"
        )

        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(statement, params)
                rows = cur.fetchall()
            conn.commit()

        total_written += len(rows)

    return total_written


def upsert_article(article: dict[str, Any]) -> int:
    """Single-row convenience wrapper around `upsert_articles`."""
    return upsert_articles([article])


def get_article_by_canonical_url(canonical_url: str) -> dict[str, Any] | None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM articles WHERE canonical_url = %s",
                (canonical_url,),
            )
            return cur.fetchone()


def get_recent_articles(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent articles with titles."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM articles
                WHERE title IS NOT NULL
                ORDER BY gdelt_timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def get_articles_since(since: datetime, *, limit: int = 100) -> list[dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM articles
                WHERE gdelt_timestamp >= %s
                ORDER BY gdelt_timestamp ASC
                LIMIT %s
                """,
                (since, limit),
            )
            return cur.fetchall()


def get_unembedded_articles(*, limit: int | None = None) -> list[dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT * FROM articles
                WHERE embedding IS NULL
                ORDER BY gdelt_timestamp ASC
            """
            if limit is not None:
                query += " LIMIT %s"
                cur.execute(query, (limit,))
            else:
                cur.execute(query)
            return cur.fetchall()


def get_unclustered_articles(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Return articles that have an embedding but are not yet assigned to any cluster."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT a.* FROM articles a
                LEFT JOIN cluster_memberships cm ON cm.article_id = a.id
                WHERE a.embedding IS NOT NULL
                  AND a.title IS NOT NULL
                  AND cm.id IS NULL
                ORDER BY a.gdelt_timestamp ASC
            """
            if limit is not None:
                query += " LIMIT %s"
                cur.execute(query, (limit,))
            else:
                cur.execute(query)
            return cur.fetchall()


def get_untitled_articles(
    *, limit: int | None = None, max_attempts: int = 1
) -> list[dict[str, Any]]:
    """Return articles that have no title and haven't exceeded scrape attempts."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT * FROM articles
                WHERE title IS NULL
                  AND scrape_attempts < %s
                ORDER BY gdelt_timestamp DESC
            """
            params: list[Any] = [max_attempts]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            cur.execute(query, params)
            return cur.fetchall()


def increment_scrape_attempts(article_ids: list[str]) -> None:
    """Bump scrape_attempts for the given articles."""
    if not article_ids:
        return
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET scrape_attempts = scrape_attempts + 1, updated_at = now()
                WHERE id = ANY(%s)
                """,
                (article_ids,),
            )
        conn.commit()


def update_article_title(article_id: str, title: str) -> None:
    """Set the title for a single article."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET title = %s, updated_at = now()
                WHERE id = %s
                """,
                (title, article_id),
            )
        conn.commit()


def update_article_titles(titles: dict[str, str]) -> int:
    """Set titles for many articles in a single UPDATE.

    `titles` maps article_id → title.  Returns the number of rows updated.
    """
    if not titles:
        return 0

    pool = get_pool()
    pairs = list(titles.items())
    values_sql = ", ".join(["(%s::uuid, %s)"] * len(pairs))
    params: list[Any] = []
    for article_id, title in pairs:
        params.extend([article_id, title])

    statement = f"""
        UPDATE articles
        SET title = v.title, updated_at = now()
        FROM (VALUES {values_sql}) AS v(id, title)
        WHERE articles.id = v.id
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            affected = cur.rowcount
        conn.commit()
    return affected


def update_article_embedding(article_id: str, embedding: list[float], model: str) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET embedding = %s, embedding_model = %s, updated_at = now()
                WHERE id = %s
                """,
                (embedding, model, article_id),
            )
        conn.commit()
