"""Allow running the embedding pipeline as a module: python -m gdelt_event_pipeline.embeddings"""

from __future__ import annotations

import logging
import sys

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.embeddings.pipeline import run_embedding
from gdelt_event_pipeline.storage.database import close_pool, init_pool


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings = get_settings()
    init_pool(settings.db)

    try:
        result = run_embedding(settings.embedding)
        print(f"\nEmbedding summary:")
        print(f"  Articles fetched:  {result.articles_fetched}")
        print(f"  Articles embedded: {result.articles_embedded}")
        print(f"  Articles skipped:  {result.articles_skipped}")
        print(f"  Articles failed:   {result.articles_failed}")
    finally:
        close_pool()

    return 0 if result.articles_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
