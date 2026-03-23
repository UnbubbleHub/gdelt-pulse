"""Data structures for the search query layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SearchFilters:
    """Optional filters to narrow search results."""

    locations: list[str] | None = None
    persons: list[str] | None = None
    organizations: list[str] | None = None
    themes: list[str] | None = None
    domains: list[str] | None = None
    sources: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


@dataclass
class SearchRequest:
    """Input for a hybrid search."""

    query: str
    filters: SearchFilters | None = None
    limit: int = 20
    semantic_weight: float = 0.5
    search_clusters: bool = False


@dataclass
class ScoredArticle:
    """An article with its ranking metadata."""

    article: dict[str, Any]
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0


@dataclass
class ScoredCluster:
    """A cluster with its search distance."""

    cluster: dict[str, Any]
    cosine_distance: float
    rank: int


@dataclass
class SearchResult:
    """Output from a hybrid search."""

    articles: list[ScoredArticle] = field(default_factory=list)
    clusters: list[ScoredCluster] = field(default_factory=list)
    query: str = ""
    total_semantic_hits: int = 0
    total_keyword_hits: int = 0
