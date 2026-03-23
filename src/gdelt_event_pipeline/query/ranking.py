"""Reciprocal Rank Fusion for combining ranked result lists."""

from __future__ import annotations


def reciprocal_rank_fusion(
    semantic_ids: list[str],
    keyword_ids: list[str],
    *,
    k: int = 60,
    semantic_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        semantic_ids: Article IDs from vector search, in rank order.
        keyword_ids: Article IDs from keyword search, in rank order.
        k: RRF constant (default 60, from Cormack et al. 2009).
        semantic_weight: Weight for semantic results (0.0–1.0).
            keyword_weight is implicitly 1 - semantic_weight.

    Returns:
        List of (article_id, rrf_score) sorted by descending score.
    """
    keyword_weight = 1.0 - semantic_weight
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(semantic_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight / (k + rank)

    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
