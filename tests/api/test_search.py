"""Tests for /api/search endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gdelt_event_pipeline.api.routers.search as search_module


class TestSearchEndpoint:
    def test_missing_q_returns_422(self, client_no_db):
        resp = client_no_db.get("/api/search")
        assert resp.status_code == 422

    def test_returns_501_when_search_unavailable(self, client_no_db, monkeypatch):
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)
        resp = client_no_db.get("/api/search?q=test")
        assert resp.status_code == 501

    def test_happy_path(self, client_no_db, monkeypatch):
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", True)
        mock_result = MagicMock()
        mock_result.query = "test"
        mock_result.total_semantic_hits = 1
        mock_result.total_keyword_hits = 0
        mock_result.articles = []
        mock_result.clusters = []
        with patch(
            "gdelt_event_pipeline.api.routers.search.hybrid_search",
            return_value=mock_result,
        ):
            resp = client_no_db.get("/api/search?q=test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert "articles" in data
        assert "clusters" in data

    def test_limit_bounds(self, client_no_db, monkeypatch):
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)
        resp = client_no_db.get("/api/search?q=test&limit=0")
        assert resp.status_code == 422
        resp = client_no_db.get("/api/search?q=test&limit=101")
        assert resp.status_code == 422

    def test_filters_forwarded(self, client_no_db, monkeypatch):
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", True)
        mock_result = MagicMock()
        mock_result.query = "test"
        mock_result.total_semantic_hits = 0
        mock_result.total_keyword_hits = 0
        mock_result.articles = []
        mock_result.clusters = []
        with patch(
            "gdelt_event_pipeline.api.routers.search.hybrid_search",
            return_value=mock_result,
        ) as mock_search:
            client_no_db.get("/api/search?q=test&location=Rome&domain=bbc.com")
        req = mock_search.call_args[0][0]
        assert req.filters is not None
        assert req.filters.locations == ["Rome"]
        assert req.filters.domains == ["bbc.com"]
