"""Continuous pipeline runner.

Runs all pipeline stages (ingest -> scrape titles -> embed -> cluster) in a
loop, sleeping between cycles to match the GDELT 15-minute update cadence.

Usage:
    python -m gdelt_event_pipeline.runner
    # or with env overrides:
    PIPELINE_INTERVAL=600 python -m gdelt_event_pipeline.runner
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime

from gdelt_event_pipeline.clustering.pipeline import run_clustering
from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.embeddings.pipeline import run_embedding
from gdelt_event_pipeline.ingestion.pipeline import run_ingestion, run_title_scraping
from gdelt_event_pipeline.storage.database import close_pool, init_pool

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 15 * 60  # 15 minutes — matches GDELT update frequency
TITLE_SCRAPE_BATCH = None  # no limit — scrape all new untitled articles each cycle


def _ensure_schema() -> None:
    """Run schema SQL if tables don't exist yet."""
    from pathlib import Path

    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_name = 'articles'"
                ")"
            )
            row = cur.fetchone()
            exists = row[0] if isinstance(row, (tuple, list)) else row.get("exists", False)
        if not exists:
            logger.info("Tables not found — running schema initialization...")
            schema_path = Path(__file__).resolve().parents[2] / "sql" / "001_schema.sql"
            if not schema_path.exists():
                # In Docker, sql/ is at /app/sql/
                schema_path = Path("/app/sql/001_schema.sql")
            sql = schema_path.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Schema created successfully.")


def _cleanup_failed_articles() -> int:
    """Delete articles that failed scraping and will never be useful."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM articles
                WHERE title IS NULL
                  AND scrape_attempts >= 1
                  AND embedding IS NULL
                """
            )
            deleted = cur.rowcount
        conn.commit()
    if deleted:
        logger.info("Cleaned up %d failed articles", deleted)
    return deleted


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_cycle(settings) -> dict[str, str]:
    """Execute one full pipeline cycle. Returns a summary dict."""
    summary: dict[str, str] = {}

    # 1. Ingest
    try:
        ing = run_ingestion()
        summary["ingest"] = (
            f"fetched={ing.rows_fetched} upserted={ing.rows_upserted} "
            f"dupes={ing.duplicate_urls} failed={ing.rows_failed}"
        )
    except Exception:
        logger.exception("Ingestion failed")
        summary["ingest"] = "ERROR"

    # 2. Scrape titles
    try:
        attempted, succeeded = run_title_scraping(batch_size=TITLE_SCRAPE_BATCH)
        summary["titles"] = f"attempted={attempted} succeeded={succeeded}"
    except Exception:
        logger.exception("Title scraping failed")
        summary["titles"] = "ERROR"

    # 2b. Delete articles that failed scraping (no title, already attempted)
    try:
        deleted = _cleanup_failed_articles()
        if deleted:
            summary["cleanup"] = f"deleted={deleted}"
    except Exception:
        logger.exception("Cleanup failed")

    # 3. Embed
    try:
        emb = run_embedding(settings.embedding)
        summary["embed"] = (
            f"fetched={emb.articles_fetched} embedded={emb.articles_embedded} "
            f"skipped={emb.articles_skipped} failed={emb.articles_failed}"
        )
    except Exception:
        logger.exception("Embedding failed")
        summary["embed"] = "ERROR"

    # 4. Cluster
    try:
        cl = run_clustering(max_age_hours=settings.clustering.window_hours)
        summary["cluster"] = (
            f"processed={cl.articles_processed} existing={cl.assigned_to_existing} "
            f"new={cl.new_clusters_created} failed={cl.articles_failed}"
        )
    except Exception:
        logger.exception("Clustering failed")
        summary["cluster"] = "ERROR"

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    interval = int(os.environ.get("PIPELINE_INTERVAL", DEFAULT_INTERVAL))
    settings = get_settings()

    logger.info("Starting continuous pipeline (interval=%ds)", interval)
    init_pool(settings.db)

    # Auto-create schema on fresh databases (e.g. Railway first deploy)
    _ensure_schema()

    # Graceful shutdown on SIGTERM/SIGINT
    shutdown = False

    def _handle_signal(signum, _frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down after current cycle...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cycle = 0
    try:
        while not shutdown:
            cycle += 1
            logger.info("=== Cycle %d starting at %s ===", cycle, _utcnow())
            t0 = time.monotonic()

            summary = run_cycle(settings)

            elapsed = time.monotonic() - t0
            parts = [f"{k}: {v}" for k, v in summary.items()]
            logger.info(
                "=== Cycle %d done in %.1fs | %s ===",
                cycle,
                elapsed,
                " | ".join(parts),
            )

            if shutdown:
                break

            # Sleep in short increments so we can respond to signals
            sleep_until = time.monotonic() + interval
            while time.monotonic() < sleep_until and not shutdown:
                time.sleep(1)
    finally:
        logger.info("Shutting down, closing database pool...")
        close_pool()

    logger.info("Pipeline stopped after %d cycles.", cycle)
    sys.exit(0)


if __name__ == "__main__":
    main()
