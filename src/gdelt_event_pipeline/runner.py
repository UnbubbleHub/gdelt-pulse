"""Continuous pipeline runner — micro-cycle architecture.

Replaces the previous "everything every 15 minutes" cycle with a fast loop
that runs different stages at their natural cadence:

    INGEST   every  INGEST_INTERVAL_SECONDS  (default 900s — GDELT cadence)
    EMBED    every  PIPELINE_TICK_SECONDS    (default 60s,  batch EMBED_PER_TICK=200)
    CLUSTER  every  PIPELINE_TICK_SECONDS    (only if embed produced rows)
    CLEANUP  every  CLEANUP_INTERVAL_SECONDS (default 3600s — retention + orphan clusters)

Effect on CPU: instead of ~5-10 min of full-CPU bursts every 15 min and 0
otherwise, the runner does ~10-20s of work every minute. The Railway CPU
graph flattens to a near-continuous low band instead of impulse spikes.

Usage:
    python -m gdelt_event_pipeline.runner

Env overrides:
    PIPELINE_TICK_SECONDS      tick cadence (s, default 60)
    INGEST_INTERVAL_SECONDS    ingest cadence (s, default 900)
    EMBED_PER_TICK             articles to embed per tick (default 200)
    CLUSTER_PER_TICK           articles to cluster per tick (default 400)
    CLEANUP_INTERVAL_SECONDS   retention + orphan-cluster cadence (s, default 3600)
    HEALTHCHECKS_PING_URL      optional dead-man-switch URL, pinged after each ingest
    PIPELINE_INTERVAL          legacy alias for INGEST_INTERVAL_SECONDS
"""

from __future__ import annotations

import logging
import os
import resource
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from gdelt_event_pipeline.clustering.pipeline import run_clustering
from gdelt_event_pipeline.config.log_setup import setup_logging
from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.embeddings.pipeline import run_embedding
from gdelt_event_pipeline.ingestion.pipeline import run_ingestion, run_title_scraping
from gdelt_event_pipeline.storage.database import close_pool, get_pool, init_pool
from gdelt_event_pipeline.storage.migrations import ensure_schema

logger = logging.getLogger(__name__)


# ─── env config ────────────────────────────────────────────────────


def _env_int(key: str, default: int, *, legacy: str | None = None) -> int:
    raw = os.environ.get(key) or (os.environ.get(legacy) if legacy else None)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an int, using default %d", key, raw, default)
        return default


@dataclass
class RunnerConfig:
    tick_seconds: int = field(default_factory=lambda: _env_int("PIPELINE_TICK_SECONDS", 60))
    ingest_interval: int = field(
        default_factory=lambda: _env_int(
            "INGEST_INTERVAL_SECONDS", 15 * 60, legacy="PIPELINE_INTERVAL"
        )
    )
    embed_per_tick: int = field(default_factory=lambda: _env_int("EMBED_PER_TICK", 200))
    cluster_per_tick: int = field(default_factory=lambda: _env_int("CLUSTER_PER_TICK", 400))
    cleanup_interval: int = field(
        default_factory=lambda: _env_int("CLEANUP_INTERVAL_SECONDS", 60 * 60)
    )
    healthcheck_url: str = field(
        default_factory=lambda: os.environ.get("HEALTHCHECKS_PING_URL", "").strip()
    )


# ─── DB-touching helpers (kept from prior cycle-based runner) ──────


def _cleanup_old_articles(retention_hours: int) -> int:
    """Delete articles older than the retention window."""
    if retention_hours <= 0:
        return 0
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM articles "
                "WHERE gdelt_timestamp < NOW() - make_interval(hours => %s)",
                (retention_hours,),
            )
            deleted = cur.rowcount
        conn.commit()
    if deleted:
        logger.info("Retention: deleted %d articles older than %dh", deleted, retention_hours)
    return deleted


def _cleanup_orphan_clusters() -> int:
    """Delete clusters whose members were all removed by retention cascade."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM clusters c
                WHERE NOT EXISTS (
                    SELECT 1 FROM cluster_memberships m WHERE m.cluster_id = c.id
                )
                """
            )
            deleted = cur.rowcount
        conn.commit()
    if deleted:
        logger.info("Cleaned up %d orphan clusters", deleted)
    return deleted


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _peak_rss_mb() -> float:
    """Peak resident set size in MB. ru_maxrss is KB on Linux, bytes on macOS."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform == "linux" else raw / (1024 * 1024)


def _collect_metrics() -> dict[str, int]:
    """Snapshot pipeline-wide counters for the METRIC log line."""
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM articles")
        articles_total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL")
        embedded_total = cur.fetchone()[0]
        cur.execute("SELECT pg_database_size(current_database()) / (1024 * 1024)")
        db_mb = cur.fetchone()[0]
    return {
        "articles_total": int(articles_total),
        "embedded_total": int(embedded_total),
        "db_mb": int(db_mb),
    }


def _ping_healthchecks(url: str) -> None:
    """Fire-and-forget dead-man-switch ping. No-op if url is empty."""
    if not url:
        return
    try:
        httpx.get(url, timeout=5.0)
    except Exception as exc:
        logger.warning("Healthchecks ping failed: %s", exc)


# ─── stage runners (return success bool + summary fragment) ────────


def _do_ingest(settings) -> tuple[bool, dict[str, str]]:
    """Ingest GDELT + scrape titles. Returns (ok, summary)."""
    summary: dict[str, str] = {}
    ok = True
    try:
        ing = run_ingestion()
        summary["ingest"] = (
            f"fetched={ing.rows_fetched} upserted={ing.rows_upserted} "
            f"dupes={ing.duplicate_urls} failed={ing.rows_failed}"
        )
    except Exception:
        logger.exception("Ingestion failed")
        summary["ingest"] = "ERROR"
        ok = False

    try:
        attempted, succeeded = run_title_scraping(batch_size=None)
        summary["titles"] = f"attempted={attempted} succeeded={succeeded}"
    except Exception:
        logger.exception("Title scraping failed")
        summary["titles"] = "ERROR"
        ok = False

    return ok, summary


def _do_embed(limit: int) -> tuple[bool, int, dict[str, str]]:
    """Embed up to `limit` articles. Returns (ok, embedded_count, summary)."""
    try:
        emb = run_embedding(limit=limit)
    except Exception:
        logger.exception("Embedding failed")
        return False, 0, {"embed": "ERROR"}
    summary = {
        "embed": (
            f"fetched={emb.articles_fetched} embedded={emb.articles_embedded} "
            f"skipped={emb.articles_skipped} failed={emb.articles_failed}"
        )
    }
    return True, emb.articles_embedded, summary


def _do_cluster(limit: int, window_hours: int) -> tuple[bool, dict[str, str]]:
    try:
        cl = run_clustering(limit=limit, max_age_hours=window_hours)
    except Exception:
        logger.exception("Clustering failed")
        return False, {"cluster": "ERROR"}
    summary = {
        "cluster": (
            f"processed={cl.articles_processed} existing={cl.assigned_to_existing} "
            f"new={cl.new_clusters_created} failed={cl.articles_failed}"
        )
    }
    return True, summary


def _do_cleanup(retention_hours: int) -> dict[str, str]:
    """Run retention + orphan-cluster cleanup. Errors are swallowed and logged."""
    summary: dict[str, str] = {}
    try:
        expired = _cleanup_old_articles(retention_hours)
        if expired:
            summary["retention"] = f"deleted={expired}"
    except Exception:
        logger.exception("Retention cleanup errored")

    try:
        orphans = _cleanup_orphan_clusters()
        if orphans:
            summary["orphans"] = f"deleted={orphans}"
    except Exception:
        logger.exception("Orphan cluster cleanup errored")

    return summary


# ─── main loop ─────────────────────────────────────────────────────


def main() -> None:
    setup_logging()

    cfg = RunnerConfig()
    settings = get_settings()

    logger.info(
        "Starting micro-cycle runner: tick=%ds ingest_every=%ds "
        "embed_per_tick=%d cluster_per_tick=%d cleanup_every=%ds healthcheck=%s",
        cfg.tick_seconds,
        cfg.ingest_interval,
        cfg.embed_per_tick,
        cfg.cluster_per_tick,
        cfg.cleanup_interval,
        "set" if cfg.healthcheck_url else "off",
    )

    if not settings.db.url and settings.db.host == "localhost":
        logger.error(
            "No DATABASE_URL or PGHOST set — refusing to connect to localhost. "
            "On Railway, link the Postgres plugin variables to this service."
        )
        sys.exit(1)

    pool = init_pool(settings.db)
    ensure_schema(pool)

    # Graceful shutdown
    shutdown = False

    def _handle_signal(signum, _frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down at next tick boundary...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Schedule trackers — anchored in the past so first tick runs both immediately.
    last_ingest_at = 0.0
    last_cleanup_at = 0.0

    # Per-stage failure tracking — back off embed if it errors repeatedly.
    embed_failures = 0
    cluster_failures = 0
    EMBED_FAIL_BACKOFF_AT = 3
    EMBED_FAIL_CAP_TICKS = 60  # at tick=60s this caps the skip at ~1 hour
    embed_skip_until_tick = 0

    tick = 0
    try:
        while not shutdown:
            tick += 1
            t0 = time.monotonic()
            now = t0
            summary: dict[str, str] = {}
            embedded_this_tick = 0
            did_ingest = False

            # ── ingest tick (15 min cadence) ─────────────────────
            if now - last_ingest_at >= cfg.ingest_interval:
                ok, frag = _do_ingest(settings)
                summary.update(frag)
                did_ingest = True
                if ok:
                    last_ingest_at = now
                    _ping_healthchecks(cfg.healthcheck_url)
                else:
                    # On error, retry in ~5 min instead of waiting the full interval.
                    last_ingest_at = now - cfg.ingest_interval + 5 * 60

            # ── embed tick (every tick, unless backing off) ──────
            if tick >= embed_skip_until_tick:
                ok, embedded_this_tick, frag = _do_embed(cfg.embed_per_tick)
                # Only surface in summary if something happened (avoid noisy idle ticks).
                if embedded_this_tick > 0 or frag.get("embed") == "ERROR":
                    summary.update(frag)
                if ok:
                    embed_failures = 0
                else:
                    embed_failures += 1
                    if embed_failures >= EMBED_FAIL_BACKOFF_AT:
                        skip = min(
                            2 ** (embed_failures - EMBED_FAIL_BACKOFF_AT + 1),
                            EMBED_FAIL_CAP_TICKS,
                        )
                        embed_skip_until_tick = tick + skip
                        logger.warning(
                            "Embed failed %d times consecutively, skipping next %d ticks",
                            embed_failures,
                            skip,
                        )

            # ── cluster tick (only when fresh embeddings exist) ──
            if embedded_this_tick > 0:
                ok, frag = _do_cluster(cfg.cluster_per_tick, settings.clustering.window_hours)
                summary.update(frag)
                cluster_failures = 0 if ok else cluster_failures + 1

            # ── cleanup tick (hourly) ────────────────────────────
            if now - last_cleanup_at >= cfg.cleanup_interval:
                frag = _do_cleanup(settings.retention.hours)
                summary.update(frag)
                last_cleanup_at = now

            # ── per-tick log + METRIC line ───────────────────────
            elapsed = time.monotonic() - t0
            if summary:
                parts = [f"{k}: {v}" for k, v in summary.items()]
                logger.info(
                    "tick=%d (%.1fs) %s%s",
                    tick,
                    elapsed,
                    "[INGEST] " if did_ingest else "",
                    " | ".join(parts),
                )

            # Emit a parseable metric line: every ingest tick (= 15 min cadence)
            # plus every 15 ticks (= ~15 min) as a fallback heartbeat.
            if did_ingest or tick % 15 == 0:
                try:
                    m = _collect_metrics()
                    logger.info(
                        "METRIC tick=%d tick_seconds=%.1f embedded=%d ingest=%d "
                        "embed_fails=%d cluster_fails=%d articles_total=%d "
                        "embedded_total=%d db_mb=%d peak_rss_mb=%.0f ts=%s",
                        tick,
                        elapsed,
                        embedded_this_tick,
                        1 if did_ingest else 0,
                        embed_failures,
                        cluster_failures,
                        m["articles_total"],
                        m["embedded_total"],
                        m["db_mb"],
                        _peak_rss_mb(),
                        _utcnow(),
                    )
                except Exception:
                    logger.exception("Failed to collect metrics")

            if shutdown:
                break

            # Sleep the remainder of the tick budget (or skip if we overran)
            sleep_for = cfg.tick_seconds - elapsed
            if sleep_for <= 0:
                if elapsed > cfg.tick_seconds * 2:
                    logger.warning(
                        "Tick %d took %.1fs (> 2× budget %ds), proceeding immediately",
                        tick,
                        elapsed,
                        cfg.tick_seconds,
                    )
                continue

            # Sleep in 1s increments so SIGTERM is responsive.
            sleep_until = time.monotonic() + sleep_for
            while time.monotonic() < sleep_until and not shutdown:
                time.sleep(1)
    finally:
        logger.info("Shutting down, closing database pool...")
        close_pool()

    logger.info("Pipeline stopped after %d ticks.", tick)
    sys.exit(0)


if __name__ == "__main__":
    main()
