"""Database connection pool management."""

from __future__ import annotations

import psycopg
from psycopg_pool import ConnectionPool

from gdelt_event_pipeline.config.settings import DatabaseSettings


_pool: ConnectionPool | None = None


def init_pool(db: DatabaseSettings, *, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """Create and return the global connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    _pool = ConnectionPool(
        conninfo=db.dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": False, "row_factory": psycopg.rows.dict_row},
    )
    return _pool


def get_pool() -> ConnectionPool:
    """Return the existing connection pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("Connection pool not initialized. Call init_pool() first.")
    return _pool


def close_pool() -> None:
    """Close the global connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
