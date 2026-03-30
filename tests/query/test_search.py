"""Tests for hybrid search orchestrator."""

from unittest.mock import patch

from gdelt_event_pipeline.query.models import SearchFilters, SearchRequest
from gdelt_event_pipeline.query.search import hybrid_search


def _make_article_hit(article_id, title="Test", distance=0.1):
    return {
        "id": article_id,
        "title": title,
        "canonical_url": f"http://example.com/{article_id}",
        "cosine_distance": distance,
    }


def _make_keyword_hit(article_id, title="Test", rank_score=0.5):
    return {
        "id": article_id,
        "title": title,
        "canonical_url": f"http://example.com/{article_id}",
        "rank_score": rank_score,
    }


class TestHybridSearch:
    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_basic_flow(self, mock_embed, mock_vector, mock_keyword):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = [
            _make_article_hit("a1", distance=0.1),
            _make_article_hit("a2", distance=0.2),
        ]
        mock_keyword.return_value = [
            _make_keyword_hit("a2", rank_score=0.9),
            _make_keyword_hit("a3", rank_score=0.5),
        ]

        request = SearchRequest(query="test query", limit=10)
        result = hybrid_search(request)

        assert len(result.articles) == 3
        assert result.total_semantic_hits == 2
        assert result.total_keyword_hits == 2
        # a2 appears in both, should rank highest
        assert result.articles[0].article["id"] == "a2"

    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_respects_limit(self, mock_embed, mock_vector, mock_keyword):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = [_make_article_hit(f"a{i}") for i in range(10)]
        mock_keyword.return_value = [_make_keyword_hit(f"b{i}") for i in range(10)]

        request = SearchRequest(query="test", limit=5)
        result = hybrid_search(request)

        assert len(result.articles) <= 5

    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_filters_passed_through(self, mock_embed, mock_vector, mock_keyword):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = []
        mock_keyword.return_value = []

        filters = SearchFilters(locations=["Turkey"])
        request = SearchRequest(query="earthquake", filters=filters, limit=10)
        hybrid_search(request)

        mock_vector.assert_called_once()
        assert mock_vector.call_args.kwargs["filters"] is filters
        mock_keyword.assert_called_once()
        assert mock_keyword.call_args.kwargs["filters"] is filters

    @patch("gdelt_event_pipeline.query.search.search_clusters_by_vector")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_cluster_search(self, mock_embed, mock_vector, mock_keyword, mock_clusters):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = []
        mock_keyword.return_value = []
        mock_clusters.return_value = [
            {"id": "c1", "representative_title": "Event 1", "cosine_distance": 0.1},
        ]

        request = SearchRequest(query="test", search_clusters=True, limit=10)
        result = hybrid_search(request)

        mock_clusters.assert_called_once()
        assert len(result.clusters) == 1
        assert result.clusters[0].cosine_distance == 0.1

    @patch("gdelt_event_pipeline.query.search.search_clusters_by_vector")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_no_cluster_search_by_default(
        self, mock_embed, mock_vector, mock_keyword, mock_clusters
    ):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = []
        mock_keyword.return_value = []

        request = SearchRequest(query="test", limit=10)
        hybrid_search(request)

        mock_clusters.assert_not_called()

    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_no_results(self, mock_embed, mock_vector, mock_keyword):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = []
        mock_keyword.return_value = []

        request = SearchRequest(query="nonexistent", limit=10)
        result = hybrid_search(request)

        assert result.articles == []
        assert result.total_semantic_hits == 0
        assert result.total_keyword_hits == 0

    @patch("gdelt_event_pipeline.query.search.search_articles_by_keyword")
    @patch("gdelt_event_pipeline.query.search.search_articles_by_vector")
    @patch("gdelt_event_pipeline.query.search.embed_texts")
    def test_rrf_scores_present(self, mock_embed, mock_vector, mock_keyword):
        mock_embed.return_value = [[0.1] * 384]
        mock_vector.return_value = [_make_article_hit("a1")]
        mock_keyword.return_value = [_make_keyword_hit("a1")]

        request = SearchRequest(query="test", limit=10)
        result = hybrid_search(request)

        assert result.articles[0].rrf_score > 0
        assert result.articles[0].semantic_rank == 1
        assert result.articles[0].keyword_rank == 1
