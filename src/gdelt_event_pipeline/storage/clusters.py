"""Cluster and membership storage operations."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from gdelt_event_pipeline.storage.database import get_pool


def create_cluster(
    *,
    representative_title: str | None = None,
    centroid_embedding: list[float] | None = None,
    first_article_at: str | None = None,
) -> dict[str, Any]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO clusters (representative_title, centroid_embedding,
                                      first_article_at, last_article_at, article_count)
                VALUES (%s, %s, %s, %s, 0)
                RETURNING *
                """,
                (representative_title, centroid_embedding,
                 first_article_at, first_article_at),
            )
            row = cur.fetchone()
        conn.commit()
    return row


def assign_article_to_cluster(
    article_id: str,
    cluster_id: str,
    *,
    similarity_score: float | None = None,
    assignment_method: str | None = None,
) -> dict[str, Any]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO cluster_memberships
                    (article_id, cluster_id, similarity_score, assignment_method)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, cluster_id) DO NOTHING
                RETURNING *
                """,
                (article_id, cluster_id, similarity_score, assignment_method),
            )
            row = cur.fetchone()

            # Update cluster denormalized fields
            cur.execute(
                """
                UPDATE clusters SET
                    article_count    = (SELECT count(*) FROM cluster_memberships
                                        WHERE cluster_id = %s),
                    first_article_at = (SELECT min(a.gdelt_timestamp)
                                        FROM cluster_memberships cm
                                        JOIN articles a ON a.id = cm.article_id
                                        WHERE cm.cluster_id = %s),
                    last_article_at  = (SELECT max(a.gdelt_timestamp)
                                        FROM cluster_memberships cm
                                        JOIN articles a ON a.id = cm.article_id
                                        WHERE cm.cluster_id = %s),
                    updated_at       = now()
                WHERE id = %s
                """,
                (cluster_id, cluster_id, cluster_id, cluster_id),
            )
        conn.commit()
    return row


def get_active_clusters(*, limit: int = 100) -> list[dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM clusters
                WHERE is_active = true
                ORDER BY last_article_at DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def get_cluster_articles(cluster_id: str) -> list[dict[str, Any]]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT a.*, cm.similarity_score, cm.assignment_method, cm.assigned_at
                FROM articles a
                JOIN cluster_memberships cm ON cm.article_id = a.id
                WHERE cm.cluster_id = %s
                ORDER BY a.gdelt_timestamp DESC
                """,
                (cluster_id,),
            )
            return cur.fetchall()


def update_cluster_centroid(
    cluster_id: str, centroid: list[float]
) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE clusters
                SET centroid_embedding = %s, updated_at = now()
                WHERE id = %s
                """,
                (centroid, cluster_id),
            )
        conn.commit()
