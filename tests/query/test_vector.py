"""Tests for vector search operations."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from gdelt_event_pipeline.query.models import SearchFilters
from tests.conftest import make_mock_pool

MODULE = "gdelt_event_pipeline.query.vector"


@pytest.fixture
def patch_pool():
    pool = make_mock_pool()
    with patch(f"{MODULE}.get_pool", return_value=pool):
        yield pool


class TestSearchArticlesByVector:
    def test_basic_query_structure(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_articles_by_vector([0.1, 0.2])
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "embedding <=> %s::vector" in sql
        assert "embedding IS NOT NULL" in sql
        assert "title IS NOT NULL" in sql
        assert "LIMIT" in sql

    def test_default_limit(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_articles_by_vector([0.1])
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[-1] == 40

    def test_custom_limit(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_articles_by_vector([0.1], limit=10)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[-1] == 10

    def test_embedding_appears_twice_in_params(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        emb = [0.1, 0.2, 0.3]
        search_articles_by_vector(emb)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == emb
        assert params[-2] == emb

    def test_no_filters(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_articles_by_vector([0.1], filters=None)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert len(params) == 3

    def test_with_location_filter(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        filters = SearchFilters(locations=["Rome"])
        search_articles_by_vector([0.1], filters=filters)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert len(params) > 3
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "locations @>" in sql

    def test_with_date_filters(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        filters = SearchFilters(
            date_from=datetime(2026, 1, 1, tzinfo=UTC),
            date_to=datetime(2026, 6, 1, tzinfo=UTC),
        )
        search_articles_by_vector([0.1], filters=filters)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "gdelt_timestamp >=" in sql
        assert "gdelt_timestamp <=" in sql

    def test_with_domain_filter(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        filters = SearchFilters(domains=["bbc.com", "cnn.com"])
        search_articles_by_vector([0.1], filters=filters)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "domain = ANY(%s)" in sql
        assert "domain LIKE ANY(%s)" in sql

    def test_returns_fetchall_result(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        expected = [{"id": "a1", "cosine_distance": 0.1}]
        patch_pool._mock_cur.fetchall.return_value = expected
        result = search_articles_by_vector([0.1])
        assert result == expected

    def test_param_order_with_filters(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_articles_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        filters = SearchFilters(sources=["reuters"])
        emb = [0.5, 0.6]
        search_articles_by_vector(emb, limit=15, filters=filters)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == emb
        assert params[-2] == emb
        assert params[-1] == 15


class TestSearchClustersByVector:
    def test_basic_query_structure(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_clusters_by_vector([0.1, 0.2])
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "centroid_embedding <=> %s::vector" in sql
        assert "is_active = true" in sql
        assert "centroid_embedding IS NOT NULL" in sql

    def test_default_limit(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_clusters_by_vector([0.1])
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[-1] == 10

    def test_custom_limit(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_clusters_by_vector([0.1], limit=5)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[-1] == 5

    def test_embedding_appears_twice(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        emb = [0.1, 0.2]
        search_clusters_by_vector(emb)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == emb
        assert params[1] == emb

    def test_returns_fetchall_result(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        expected = [{"id": "c1", "cosine_distance": 0.05}]
        patch_pool._mock_cur.fetchall.return_value = expected
        result = search_clusters_by_vector([0.1])
        assert result == expected

    def test_uses_dict_row_cursor(self, patch_pool):
        from gdelt_event_pipeline.query.vector import search_clusters_by_vector

        patch_pool._mock_cur.fetchall.return_value = []
        search_clusters_by_vector([0.1])
        call_kwargs = patch_pool._mock_conn.cursor.call_args
        from psycopg.rows import dict_row

        assert call_kwargs.kwargs.get("row_factory") == dict_row
