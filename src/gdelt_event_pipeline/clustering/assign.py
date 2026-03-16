"""Single-pass incremental clustering: assign an article to the nearest cluster or create a new one."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from gdelt_event_pipeline.clustering.centroid import compute_new_centroid
from gdelt_event_pipeline.storage.clusters import (
    assign_article_to_cluster,
    create_cluster,
    find_nearest_cluster,
    update_cluster_centroid,
)

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.75


@dataclass
class AssignmentResult:
    cluster_id: str
    similarity: float
    is_new_cluster: bool


def assign_article(
    article: dict[str, Any],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> AssignmentResult:
    """Assign a single article to the best matching cluster, or create a new one.

    1. Query pgvector for the nearest active cluster centroid.
    2. If cosine similarity >= threshold, join that cluster and update its centroid.
    3. Otherwise, create a new cluster seeded with this article.
    """
    embedding = article["embedding"]
    if isinstance(embedding, str):
        # pgvector may return the vector as a string representation
        embedding = [float(x) for x in embedding.strip("[]").split(",")]

    article_id = str(article["id"])

    # Find nearest cluster
    nearest = find_nearest_cluster(embedding, limit=1)

    if nearest:
        best = nearest[0]
        # pgvector <=> returns cosine distance; similarity = 1 - distance
        similarity = 1.0 - best["cosine_distance"]

        if similarity >= threshold:
            # Assign to existing cluster
            assign_article_to_cluster(
                article_id,
                str(best["id"]),
                similarity_score=similarity,
                assignment_method="nearest_centroid",
            )

            # Update centroid with running average
            current_centroid = best["centroid_embedding"]
            if isinstance(current_centroid, str):
                current_centroid = [
                    float(x) for x in current_centroid.strip("[]").split(",")
                ]
            new_centroid = compute_new_centroid(
                current_centroid, embedding, best["article_count"]
            )
            update_cluster_centroid(str(best["id"]), new_centroid)

            logger.debug(
                "Article %s → cluster %s (similarity=%.3f)",
                article_id, best["id"], similarity,
            )
            return AssignmentResult(
                cluster_id=str(best["id"]),
                similarity=similarity,
                is_new_cluster=False,
            )

    # No match above threshold — create new cluster
    cluster = create_cluster(
        representative_title=article.get("title"),
        centroid_embedding=embedding,
        first_article_at=str(article["gdelt_timestamp"]) if article.get("gdelt_timestamp") else None,
    )
    assign_article_to_cluster(
        article_id,
        str(cluster["id"]),
        similarity_score=1.0,
        assignment_method="new_cluster",
    )

    logger.debug("Article %s → NEW cluster %s", article_id, cluster["id"])
    return AssignmentResult(
        cluster_id=str(cluster["id"]),
        similarity=1.0,
        is_new_cluster=True,
    )
