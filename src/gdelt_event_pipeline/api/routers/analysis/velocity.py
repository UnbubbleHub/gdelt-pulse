"""Topic Velocity endpoints — trending topics with acceleration/deceleration."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gdelt_event_pipeline.api.routers.analysis._helpers import _categorize_themes

router = APIRouter()


@router.get("/api/velocity/topics")
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


@router.get("/api/velocity/timeline")
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
