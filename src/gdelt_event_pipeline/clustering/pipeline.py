"""Clustering pipeline: assign unclustered articles to event clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gdelt_event_pipeline.clustering.assign import assign_article
from gdelt_event_pipeline.storage.articles import get_unclustered_articles

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Summary of one clustering run."""

    articles_processed: int = 0
    assigned_to_existing: int = 0
    new_clusters_created: int = 0
    articles_skipped: int = 0
    articles_failed: int = 0


def run_clustering(
    *,
    threshold: float = 0.75,
    limit: int | None = None,
    max_age_hours: int | None = 72,
) -> ClusteringResult:
    """Execute one clustering cycle.

    Fetches unclustered articles (those with embeddings but no cluster
    membership) and assigns each via single-pass nearest-centroid matching.

    *max_age_hours* limits cluster candidates to those that received an
    article within the given window.  Pass ``None`` to disable.
    """
    result = ClusteringResult()

    articles = get_unclustered_articles(limit=limit)
    if not articles:
        logger.info("No unclustered articles found")
        return result

    window_desc = f"{max_age_hours}h window" if max_age_hours else "no window"
    logger.info(
        "Clustering %d articles (threshold=%.2f, %s)",
        len(articles),
        threshold,
        window_desc,
    )

    for article in articles:
        if not article.get("title"):
            logger.debug(
                "Skipping article %s: no title",
                article.get("id"),
            )
            result.articles_skipped += 1
            continue
        try:
            assignment = assign_article(article, threshold=threshold, max_age_hours=max_age_hours)
            result.articles_processed += 1
            if assignment.is_new_cluster:
                result.new_clusters_created += 1
            else:
                result.assigned_to_existing += 1
        except Exception:
            logger.exception("Failed to cluster article %s", article.get("id"))
            result.articles_failed += 1

    logger.info(
        "Clustering complete: processed=%d existing=%d new=%d skipped=%d failed=%d",
        result.articles_processed,
        result.assigned_to_existing,
        result.new_clusters_created,
        result.articles_skipped,
        result.articles_failed,
    )
    return result
