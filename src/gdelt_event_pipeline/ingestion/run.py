"""CLI entry point for running the ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.ingestion.pipeline import run_ingestion
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

    if not args.dry_run:
        init_pool(settings.db)

    try:
        result = run_ingestion(
            gkg_url=args.url,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    finally:
        if not args.dry_run:
            close_pool()

    print(f"\nIngestion summary:")
    print(f"  Rows fetched:     {result.rows_fetched}")
    print(f"  Rows normalized:  {result.rows_normalized}")
    print(f"  Rows upserted:    {result.rows_upserted}")
    print(f"  Rows skipped:     {result.rows_skipped}")
    print(f"  Batch duplicates: {result.duplicate_urls}")
    print(f"  Rows failed:      {result.rows_failed}")

    return 0 if result.rows_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
