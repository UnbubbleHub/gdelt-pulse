"""Tests for keyword search query construction."""

from unittest.mock import MagicMock, patch

from gdelt_event_pipeline.query.keyword import search_articles_by_keyword


class TestSearchArticlesByKeyword:
    @patch("gdelt_event_pipeline.query.keyword.get_pool")
    def test_executes_query_with_params(self, mock_get_pool):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        result = search_articles_by_keyword("earthquake", limit=10)

        assert result == []
        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args
        query_sql = call_args[0][0]
        query_params = call_args[0][1]
        assert "websearch_to_tsquery" in query_sql
        assert "earthquake" in query_params
        assert 10 in query_params

    @patch("gdelt_event_pipeline.query.keyword.get_pool")
    def test_includes_filters_in_query(self, mock_get_pool):
        from gdelt_event_pipeline.query.models import SearchFilters

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        filters = SearchFilters(domains=["bbc.co.uk"])
        search_articles_by_keyword("earthquake", limit=10, filters=filters)

        call_args = mock_cur.execute.call_args
        query_sql = call_args[0][0]
        assert "domain = ANY" in query_sql
