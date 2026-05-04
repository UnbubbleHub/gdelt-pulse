"""Schema migration runner shared by API and pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_sql_dir() -> Path:
    """Find the sql/ directory relative to the project root.

    Checks, in order:
    1. Relative to this file (works in local dev and Vercel)
    2. Docker path /app/sql/
    3. SQL_DIR env var override
    """
    env_dir = os.environ.get("SQL_DIR")
    if env_dir:
        return Path(env_dir)

    repo_dir = Path(__file__).resolve().parents[3] / "sql"
    if repo_dir.is_dir():
        return repo_dir

    docker_dir = Path("/app/sql")
    if docker_dir.is_dir():
        return docker_dir

    return repo_dir


def ensure_schema(pool) -> None:
    """Run any missing schema migrations."""
    sql_dir = _resolve_sql_dir()

    migrations = [
        ("articles", sql_dir / "001_schema.sql"),
        ("api_keys", sql_dir / "002_api_keys.sql"),
    ]

    with pool.connection() as conn:
        for table_name, schema_path in migrations:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_name = %s"
                    ")",
                    (table_name,),
                )
                row = cur.fetchone()
                exists = row[0] if isinstance(row, (tuple, list)) else row.get("exists", False)
            if not exists:
                logger.info(
                    "Table '%s' missing — running migration %s",
                    table_name,
                    schema_path.name,
                )
                sql = schema_path.read_text()
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
                logger.info("Migration %s complete.", schema_path.name)
