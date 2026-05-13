"""Search endpoint router."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from gdelt_event_pipeline.api.routers._helpers import split_csv, strip_internal_fields
from gdelt_event_pipeline.query.models import SearchFilters, SearchRequest
from gdelt_event_pipeline.query.search import hybrid_search

try:
    import fastembed as _fe_check  # noqa: F401

    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False


class ScoredArticleOut(BaseModel):
    article: dict[str, Any]
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0


class ScoredClusterOut(BaseModel):
    cluster: dict[str, Any]
    cosine_distance: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    total_semantic_hits: int = 0
    total_keyword_hits: int = 0
    articles: list[ScoredArticleOut] = []
    clusters: list[ScoredClusterOut] = []


router = APIRouter()


@router.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    semantic_weight: float = Query(0.5, ge=0.0, le=1.0, description="Semantic vs keyword weight"),
    clusters: bool = Query(False, description="Also search cluster centroids"),
    location: str | None = Query(None, description="Filter by location (comma-separated)"),
    person: str | None = Query(None, description="Filter by person (comma-separated)"),
    org: str | None = Query(None, description="Filter by organization (comma-separated)"),
    theme: str | None = Query(None, description="Filter by theme (comma-separated)"),
    domain: str | None = Query(
        None,
        description=(
            "Filter by domain (comma-separated). Matches the exact domain and any "
            "subdomain — e.g. 'corriere.it' matches both 'corriere.it' and "
            "'video.corriere.it'."
        ),
    ),
    source: str | None = Query(None, description="Filter by source (comma-separated)"),
    date_from: datetime | None = Query(None, description="Start date (ISO format)"),  # noqa: B008
    date_to: datetime | None = Query(None, description="End date (ISO format)"),  # noqa: B008
):
    """Hybrid semantic + keyword search over articles and clusters."""
    if not _SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Semantic search is not available in this deployment.",
        )
    filters = SearchFilters(
        locations=split_csv(location),
        persons=split_csv(person),
        organizations=split_csv(org),
        themes=split_csv(theme),
        domains=split_csv(domain),
        sources=split_csv(source),
        date_from=date_from,
        date_to=date_to,
    )
    has_filters = any(getattr(filters, f) is not None for f in filters.__dataclass_fields__)

    request = SearchRequest(
        query=q,
        filters=filters if has_filters else None,
        limit=limit,
        semantic_weight=semantic_weight,
        search_clusters=clusters,
    )

    result = hybrid_search(request)

    return SearchResponse(
        query=result.query,
        total_semantic_hits=result.total_semantic_hits,
        total_keyword_hits=result.total_keyword_hits,
        articles=[
            ScoredArticleOut(
                article=strip_internal_fields(sa.article),
                semantic_rank=sa.semantic_rank,
                keyword_rank=sa.keyword_rank,
                rrf_score=sa.rrf_score,
            )
            for sa in result.articles
        ],
        clusters=[
            ScoredClusterOut(
                cluster=strip_internal_fields(sc.cluster),
                cosine_distance=sc.cosine_distance,
                rank=sc.rank,
            )
            for sc in result.clusters
        ],
    )
