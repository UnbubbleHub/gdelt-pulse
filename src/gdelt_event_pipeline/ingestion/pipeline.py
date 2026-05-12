"""Ingestion pipeline: fetch GKG → normalize → upsert into database."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gdelt_event_pipeline.ingestion.gkg_fetcher import (
    download_and_parse_gkg,
    get_latest_gkg_url,
)
from gdelt_event_pipeline.ingestion.scraper import MAX_WORKERS as DEFAULT_SCRAPE_WORKERS
from gdelt_event_pipeline.normalization.normalize import normalize_row
from gdelt_event_pipeline.storage.articles import (
    get_untitled_articles,
    increment_scrape_attempts,
    update_article_titles,
    upsert_articles,
)
from gdelt_event_pipeline.storage.pipeline_state import (
    update_pipeline_state,
)

logger = logging.getLogger(__name__)

UPSERT_CHUNK_SIZE = 500


@dataclass
class IngestionResult:
    """Summary of one ingestion run."""

    rows_fetched: int = 0
    rows_normalized: int = 0
    rows_skipped: int = 0
    rows_upserted: int = 0
    rows_failed: int = 0
    duplicate_urls: int = 0


def run_ingestion(
    *,
    gkg_url: str | None = None,
    timeout: int = 30,
    dry_run: bool = False,
) -> IngestionResult:
    """Execute one ingestion cycle.

    1. Resolve the GKG file URL (latest or explicit)
    2. Download and parse the CSV rows
    3. Normalize each row and deduplicate by canonical URL
    4. Batch-upsert into the database in chunks (unless dry_run)
    5. Update pipeline_state checkpoint

    Returns an IngestionResult with counts.
    """
    result = IngestionResult()

    # 1. Resolve URL
    url = gkg_url or get_latest_gkg_url(timeout=timeout)
    logger.info("Ingesting from %s", url)

    # 2. Download and parse
    rows = download_and_parse_gkg(url, timeout=timeout)
    result.rows_fetched = len(rows)
    logger.info("Fetched %d rows", result.rows_fetched)

    # 3. Normalize + deduplicate in memory.  Multi-row INSERTs can't touch the
    # same ON CONFLICT target twice, so we have to dedupe before the DB call.
    seen_canonical_urls: set[str] = set()
    articles_to_upsert: list[dict] = []

    for row in rows:
        article = normalize_row(row)
        if article is None:
            result.rows_skipped += 1
            continue

        result.rows_normalized += 1

        canonical = article["canonical_url"]
        if canonical in seen_canonical_urls:
            result.duplicate_urls += 1
            continue
        seen_canonical_urls.add(canonical)

        articles_to_upsert.append(article)

    # 4. Batch upsert
    last_timestamp = None
    last_record_id = None

    if dry_run:
        result.rows_upserted = len(articles_to_upsert)
    elif articles_to_upsert:
        for start in range(0, len(articles_to_upsert), UPSERT_CHUNK_SIZE):
            chunk = articles_to_upsert[start : start + UPSERT_CHUNK_SIZE]
            try:
                written = upsert_articles(chunk, chunk_size=UPSERT_CHUNK_SIZE)
                result.rows_upserted += written
                last_timestamp = chunk[-1]["gdelt_timestamp"]
                last_record_id = chunk[-1]["gkg_record_id"]
            except Exception:
                logger.exception("Failed to upsert chunk of %d articles", len(chunk))
                result.rows_failed += len(chunk)

    # 5. Update checkpoint
    if not dry_run and last_timestamp is not None:
        update_pipeline_state(
            "gdelt_gkg",
            last_processed_timestamp=last_timestamp,
            last_processed_record_id=last_record_id,
        )
        logger.info("Updated pipeline checkpoint to %s", last_record_id)

    logger.info(
        "Ingestion complete: fetched=%d normalized=%d upserted=%d "
        "skipped=%d duplicates=%d failed=%d",
        result.rows_fetched,
        result.rows_normalized,
        result.rows_upserted,
        result.rows_skipped,
        result.duplicate_urls,
        result.rows_failed,
    )
    return result


def run_title_scraping(
    *,
    batch_size: int | None = None,
    timeout: int = 10,
    max_workers: int = DEFAULT_SCRAPE_WORKERS,
) -> tuple[int, int]:
    """Scrape titles for articles that don't have one yet.

    Returns (attempted, succeeded) counts.
    """
    from gdelt_event_pipeline.ingestion.scraper import scrape_titles

    articles = get_untitled_articles(limit=batch_size)
    if not articles:
        logger.info("No untitled articles to scrape")
        return 0, 0

    logger.info("Scraping titles for %d articles", len(articles))
    titles = scrape_titles(articles, timeout=timeout, max_workers=max_workers)

    # Mark all attempted articles so failures aren't retried indefinitely
    all_ids = [str(a["id"]) for a in articles]
    increment_scrape_attempts(all_ids)

    # Persist successful titles in one batched UPDATE — otherwise each write
    # pays a round trip to Postgres, which dominates cycle time on remote DBs.
    update_article_titles(titles)

    logger.info("Updated %d/%d article titles", len(titles), len(articles))
    return len(articles), len(titles)
