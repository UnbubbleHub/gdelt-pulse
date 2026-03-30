"""Hybrid search orchestrator: vector + keyword + RRF."""

from __future__ import annotations

import logging
from typing import Any

from gdelt_event_pipeline.embeddings.embed import embed_texts
from gdelt_event_pipeline.query.keyword import search_articles_by_keyword
from gdelt_event_pipeline.query.models import (
    ScoredArticle,
    ScoredCluster,
    SearchRequest,
    SearchResult,
)
from gdelt_event_pipeline.query.ranking import reciprocal_rank_fusion
from gdelt_event_pipeline.query.vector import (
    search_articles_by_vector,
    search_clusters_by_vector,
)

logger = logging.getLogger(__name__)


def hybrid_search(request: SearchRequest) -> SearchResult:
    """Execute a hybrid semantic + keyword search.

    1. Embed the query text.
    2. Run vector search and keyword search (both respect filters).
    3. Merge results with Reciprocal Rank Fusion.
    4. Optionally search cluster centroids.
    """
    result = SearchResult(query=request.query)
    candidate_limit = request.limit * 2

    # 1. Embed the query
    embeddings = embed_texts([request.query])
    query_embedding = embeddings[0]

    # 2. Run both searches
    semantic_hits = search_articles_by_vector(
        query_embedding, limit=candidate_limit, filters=request.filters
    )
    keyword_hits = search_articles_by_keyword(
        request.query, limit=candidate_limit, filters=request.filters
    )

    result.total_semantic_hits = len(semantic_hits)
    result.total_keyword_hits = len(keyword_hits)

    logger.info(
        "Search hits: %d semantic, %d keyword",
        result.total_semantic_hits,
        result.total_keyword_hits,
    )

    # 3. Build lookup and rank lists
    articles_by_id: dict[str, dict[str, Any]] = {}
    semantic_ids: list[str] = []
    keyword_ids: list[str] = []

    for hit in semantic_hits:
        doc_id = str(hit["id"])
        articles_by_id[doc_id] = hit
        semantic_ids.append(doc_id)

    for hit in keyword_hits:
        doc_id = str(hit["id"])
        articles_by_id[doc_id] = hit
        keyword_ids.append(doc_id)

    # 4. Fuse rankings
    fused = reciprocal_rank_fusion(
        semantic_ids,
        keyword_ids,
        semantic_weight=request.semantic_weight,
    )

    # 5. Build scored articles
    semantic_rank_map = {doc_id: rank for rank, doc_id in enumerate(semantic_ids, 1)}
    keyword_rank_map = {doc_id: rank for rank, doc_id in enumerate(keyword_ids, 1)}

    for doc_id, rrf_score in fused[: request.limit]:
        result.articles.append(
            ScoredArticle(
                article=articles_by_id[doc_id],
                semantic_rank=semantic_rank_map.get(doc_id),
                keyword_rank=keyword_rank_map.get(doc_id),
                rrf_score=rrf_score,
            )
        )

    # 6. Optionally search clusters
    if request.search_clusters:
        cluster_hits = search_clusters_by_vector(query_embedding, limit=request.limit)
        for rank, hit in enumerate(cluster_hits, 1):
            result.clusters.append(
                ScoredCluster(
                    cluster=hit,
                    cosine_distance=hit["cosine_distance"],
                    rank=rank,
                )
            )

    return result
