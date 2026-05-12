"""Operational health endpoint.

Exposes everything we need to monitor the system without opening psql:
- row counts per table
- last successful pipeline cycle timestamp (heartbeat)
- oldest article timestamp (validates retention is working)
- approximate DB size in MB (validates Neon free-tier headroom)

Suitable for UptimeRobot / Healthchecks.io pings: returns 503 if no cycle
has completed in the last 60 minutes, 200 otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

HEARTBEAT_STALE_AFTER_MINUTES = 60


@router.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT count(*) FROM articles)              AS articles,
                    (SELECT count(*) FROM clusters)              AS clusters,
                    (SELECT count(*) FROM cluster_memberships)   AS memberships,
                    (SELECT count(*) FROM articles WHERE embedding IS NOT NULL) AS embedded,
                    (SELECT min(gdelt_timestamp) FROM articles)  AS oldest_article_at,
                    (SELECT max(gdelt_timestamp) FROM articles)  AS newest_article_at,
                    (SELECT last_successful_run_at FROM pipeline_state
                        WHERE source_name = 'gdelt_gkg')         AS last_cycle_at,
                    pg_database_size(current_database())         AS db_bytes
                """
            )
            row = cur.fetchone()

    now = datetime.now(UTC)
    last_cycle_at = row["last_cycle_at"]
    cycle_age_minutes: float | None = None
    if last_cycle_at is not None:
        cycle_age_minutes = (now - last_cycle_at).total_seconds() / 60

    healthy = (
        last_cycle_at is not None
        and cycle_age_minutes is not None
        and cycle_age_minutes <= HEARTBEAT_STALE_AFTER_MINUTES
    )

    payload = {
        "status": "ok" if healthy else "stale",
        "now": now.isoformat(),
        "counts": {
            "articles": row["articles"],
            "clusters": row["clusters"],
            "memberships": row["memberships"],
            "embedded": row["embedded"],
        },
        "pipeline": {
            "last_cycle_at": last_cycle_at.isoformat() if last_cycle_at else None,
            "cycle_age_minutes": (
                round(cycle_age_minutes, 1) if cycle_age_minutes is not None else None
            ),
            "stale_threshold_minutes": HEARTBEAT_STALE_AFTER_MINUTES,
        },
        "articles": {
            "oldest_at": row["oldest_article_at"].isoformat() if row["oldest_article_at"] else None,
            "newest_at": row["newest_article_at"].isoformat() if row["newest_article_at"] else None,
        },
        "database": {
            "bytes": row["db_bytes"],
            "megabytes": round(row["db_bytes"] / (1024 * 1024), 1),
        },
    }
    status_code = 200 if healthy else 503
    return JSONResponse(payload, status_code=status_code)
