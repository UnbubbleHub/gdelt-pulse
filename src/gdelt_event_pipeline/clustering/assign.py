"""Single-pass incremental clustering.

Assign an article to the nearest cluster or create a new one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from gdelt_event_pipeline.clustering.centroid import compute_new_centroid
from gdelt_event_pipeline.clustering.scoring import (
    compute_combined_score,
    compute_entity_overlap,
    extract_entity_sets,
    merge_entity_sets,
)
from gdelt_event_pipeline.storage.clusters import (
    assign_article_to_cluster,
    create_cluster,
    find_nearest_cluster,
    get_cluster_entity_sample,
    update_cluster_centroid,
)

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.70
N_CANDIDATES = 5


@dataclass
class AssignmentResult:
    cluster_id: str
    similarity: float
    is_new_cluster: bool


def _parse_embedding(embedding: Any) -> list[float]:
    if isinstance(embedding, str):
        return [float(x) for x in embedding.strip("[]").split(",")]
    return embedding


def assign_article(
    article: dict[str, Any],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_age_hours: int | None = 72,
) -> AssignmentResult:
    """Assign a single article to the best matching cluster, or create a new one.

    Uses a two-stage approach:
    1. Find the top N nearest clusters by centroid cosine similarity.
    2. For each candidate, compute entity overlap and combine into a final score.
    3. Pick the best candidate above threshold, or create a new cluster.
    """
    embedding = _parse_embedding(article["embedding"])
    article_id = str(article["id"])
    article_entities = extract_entity_sets(article)

    # Stage 1: find top candidates by cosine similarity (within temporal window)
    candidates = find_nearest_cluster(embedding, limit=N_CANDIDATES, max_age_hours=max_age_hours)

    # Stage 2: score each candidate with entity overlap
    best_match = None
    best_score = -1.0

    for candidate in candidates:
        cosine_sim = 1.0 - candidate["cosine_distance"]

        # Fetch entities from the cluster's recent articles
        sample = get_cluster_entity_sample(str(candidate["id"]), limit=5)
        cluster_entities = merge_entity_sets([extract_entity_sets(row) for row in sample])

        entity_overlap = compute_entity_overlap(article_entities, cluster_entities)
        combined = compute_combined_score(cosine_sim, entity_overlap)

        if combined > best_score:
            best_score = combined
            best_match = (candidate, cosine_sim, entity_overlap, combined)

    if best_match and best_score >= threshold:
        candidate, cosine_sim, entity_overlap, combined = best_match

        assign_article_to_cluster(
            article_id,
            str(candidate["id"]),
            similarity_score=combined,
            assignment_method="nearest_centroid_entity",
        )

        # Update centroid with running average
        current_centroid = _parse_embedding(candidate["centroid_embedding"])
        new_centroid = compute_new_centroid(current_centroid, embedding, candidate["article_count"])
        update_cluster_centroid(str(candidate["id"]), new_centroid)

        logger.debug(
            "Article %s → cluster %s (cosine=%.3f entity=%.3f combined=%.3f)",
            article_id,
            candidate["id"],
            cosine_sim,
            entity_overlap,
            combined,
        )
        return AssignmentResult(
            cluster_id=str(candidate["id"]),
            similarity=combined,
            is_new_cluster=False,
        )

    # No match above threshold — create new cluster
    cluster = create_cluster(
        representative_title=article.get("title"),
        centroid_embedding=embedding,
        first_article_at=(
            str(article["gdelt_timestamp"]) if article.get("gdelt_timestamp") else None
        ),
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
