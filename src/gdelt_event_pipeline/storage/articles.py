"""Article storage operations against PostgreSQL."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from gdelt_event_pipeline.storage.database import get_pool


def upsert_article(article: dict[str, Any]) -> dict[str, Any]:
    """Insert or update an article record.

    Uses canonical_url as the dedupe key.  On conflict:
    - merges metadata if the new observation is richer
    - keeps first_seen_at as the minimum
    - updates last_seen_at to the new gdelt_timestamp
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO articles (
                    gkg_record_id, gdelt_timestamp, url, canonical_url,
                    domain, source_common_name, canonical_source, title,
                    themes, locations, organizations, persons, all_names, tone,
                    raw_payload, first_seen_at, last_seen_at
                ) VALUES (
                    %(gkg_record_id)s, %(gdelt_timestamp)s, %(url)s, %(canonical_url)s,
                    %(domain)s, %(source_common_name)s, %(canonical_source)s, %(title)s,
                    %(themes)s, %(locations)s, %(organizations)s, %(persons)s,
                    %(all_names)s, %(tone)s,
                    %(raw_payload)s, %(gdelt_timestamp)s, %(gdelt_timestamp)s
                )
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
                RETURNING *
                """,
                article,
            )
            row = cur.fetchone()
        conn.commit()
    return row


def get_article_by_canonical_url(canonical_url: str) -> dict[str, Any] | None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM articles WHERE canonical_url = %s",
                (canonical_url,),
            )
            return cur.fetchone()


def get_articles_since(
    since: datetime, *, limit: int = 100
) -> list[dict[str, Any]]:
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


def get_unembedded_articles(*, limit: int = 100) -> list[dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM articles
                WHERE embedding IS NULL
                ORDER BY gdelt_timestamp ASC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def update_article_embedding(
    article_id: str, embedding: list[float], model: str
) -> None:
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
