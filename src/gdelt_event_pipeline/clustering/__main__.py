"""Allow running the clustering pipeline as a module: python -m gdelt_event_pipeline.clustering"""

from __future__ import annotations

import argparse
import logging

from gdelt_event_pipeline.clustering.pipeline import run_clustering
from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.storage.database import close_pool, init_pool


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-pass event clustering.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for cluster assignment (default: 0.75).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to process per run (default: no limit).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings = get_settings()
    init_pool(settings.db)

    try:
        result = run_clustering(threshold=args.threshold, limit=args.limit)
        print("\nClustering summary:")
        print(f"  Articles processed:     {result.articles_processed}")
        print(f"  Assigned to existing:   {result.assigned_to_existing}")
        print(f"  New clusters created:   {result.new_clusters_created}")
        print(f"  Failed:                 {result.articles_failed}")
    finally:
        close_pool()

    return 0 if result.articles_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
