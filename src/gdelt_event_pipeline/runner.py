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
from gdelt_event_pipeline.storage.database import close_pool, get_pool, init_pool
from gdelt_event_pipeline.storage.migrations import ensure_schema

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 15 * 60  # 15 minutes — matches GDELT update frequency
TITLE_SCRAPE_BATCH = None  # no limit — scrape all new untitled articles each cycle


def _cleanup_failed_articles() -> int:
    """Delete articles that failed scraping and will never be useful."""
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


def _cleanup_old_articles(retention_hours: int) -> int:
    """Delete articles older than the retention window."""
    if retention_hours <= 0:
        return 0
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM articles
                WHERE published_at < NOW() - INTERVAL '%s hours'
                """,
                (retention_hours,),
            )
            deleted = cur.rowcount
        conn.commit()
    if deleted:
        logger.info("Retention: deleted %d articles older than %dh", deleted, retention_hours)
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

    # 2c. Delete articles older than the retention window
    try:
        expired = _cleanup_old_articles(settings.retention.hours)
        if expired:
            summary["retention"] = f"deleted={expired}"
    except Exception:
        logger.exception("Retention cleanup failed")

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

    if not settings.db.url and settings.db.host == "localhost":
        logger.error(
            "No DATABASE_URL or PGHOST set — refusing to connect to localhost. "
            "On Railway, link the Postgres plugin variables to this service."
        )
        sys.exit(1)

    pool = init_pool(settings.db)

    # Auto-create schema on fresh databases (e.g. Railway first deploy)
    ensure_schema(pool)

    # Graceful shutdown on SIGTERM/SIGINT
    shutdown = False

    def _handle_signal(signum, _frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down after current cycle...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _BACKOFF_GRACE = 3  # failures before backoff kicks in
    _BACKOFF_START = 30 * 60  # 30 min
    _BACKOFF_CAP = 120 * 60  # 120 min
    _ALERT_THRESHOLD = 5

    def _next_wait(consecutive_failures: int) -> int:
        if consecutive_failures <= _BACKOFF_GRACE:
            return interval
        step = consecutive_failures - _BACKOFF_GRACE  # 1, 2, 3, ...
        return min(_BACKOFF_START * (2 ** (step - 1)), _BACKOFF_CAP)

    def _is_failed_cycle(summary: dict[str, str]) -> bool:
        tracked = {v for k, v in summary.items() if k != "cleanup"}
        return bool(tracked) and all(v == "ERROR" for v in tracked)

    consecutive_failures = 0
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

            if _is_failed_cycle(summary):
                consecutive_failures += 1
                wait = _next_wait(consecutive_failures)
                logger.error(
                    "Cycle %d: all stages failed (consecutive_failures=%d), retrying in %.0f min",
                    cycle,
                    consecutive_failures,
                    wait / 60,
                )
                if consecutive_failures == _ALERT_THRESHOLD:
                    logger.error(
                        "ALERT: %d consecutive pipeline failures — possible outage",
                        consecutive_failures,
                    )
            else:
                if consecutive_failures > 0:
                    logger.info(
                        "Cycle %d recovered after %d failure(s)", cycle, consecutive_failures
                    )
                consecutive_failures = 0
                wait = max(0, interval - elapsed)

            if shutdown:
                break

            if wait == 0:
                logger.warning(
                    "Cycle took %.1fs (> %ds interval), starting next immediately",
                    elapsed,
                    interval,
                )
            else:
                sleep_until = time.monotonic() + wait
                while time.monotonic() < sleep_until and not shutdown:
                    time.sleep(1)
    finally:
        logger.info("Shutting down, closing database pool...")
        close_pool()

    logger.info("Pipeline stopped after %d cycles.", cycle)
    sys.exit(0)


if __name__ == "__main__":
    main()
