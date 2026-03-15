"""Pipeline state tracking operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from gdelt_event_pipeline.storage.database import get_pool


def get_pipeline_state(source_name: str = "gdelt_gkg") -> dict[str, Any] | None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM pipeline_state WHERE source_name = %s",
                (source_name,),
            )
            return cur.fetchone()


def update_pipeline_state(
    source_name: str,
    *,
    last_processed_timestamp: datetime | None = None,
    last_processed_record_id: str | None = None,
) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_state
                SET last_processed_timestamp = COALESCE(%s, last_processed_timestamp),
                    last_processed_record_id = COALESCE(%s, last_processed_record_id),
                    last_successful_run_at   = now(),
                    updated_at               = now()
                WHERE source_name = %s
                """,
                (last_processed_timestamp, last_processed_record_id, source_name),
            )
        conn.commit()
