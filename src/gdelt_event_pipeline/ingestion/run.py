"""CLI entry point for running the ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.ingestion.pipeline import run_ingestion, run_title_scraping
from gdelt_event_pipeline.storage.database import close_pool, init_pool


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GDELT GKG ingestion pipeline.")
    parser.add_argument(
        "--url",
        help="Specific GKG zip URL to ingest. Defaults to latest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize and count rows without writing to the database.",
    )
    parser.add_argument(
        "--scrape-titles",
        action="store_true",
        help="After ingestion, scrape titles for untitled articles.",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Skip ingestion — only scrape titles for existing untitled articles.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    settings = get_settings()

    needs_db = not args.dry_run or args.scrape_titles or args.scrape_only
    if needs_db:
        init_pool(settings.db)

    try:
        # Ingestion phase
        if not args.scrape_only:
            result = run_ingestion(
                gkg_url=args.url,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            print(f"\nIngestion summary:")
            print(f"  Rows fetched:     {result.rows_fetched}")
            print(f"  Rows normalized:  {result.rows_normalized}")
            print(f"  Rows upserted:    {result.rows_upserted}")
            print(f"  Rows skipped:     {result.rows_skipped}")
            print(f"  Batch duplicates: {result.duplicate_urls}")
            print(f"  Rows failed:      {result.rows_failed}")

        # Title scraping phase
        if args.scrape_titles or args.scrape_only:
            attempted, succeeded = run_title_scraping(timeout=min(args.timeout, 10))
            print(f"\nTitle scraping summary:")
            print(f"  Articles attempted: {attempted}")
            print(f"  Titles found:       {succeeded}")
    finally:
        if needs_db:
            close_pool()

    if args.scrape_only:
        return 0
    return 0 if result.rows_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
