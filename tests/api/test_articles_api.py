"""Tests for /api/articles and /api/stats endpoints."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import make_article, make_mock_pool


class TestListArticles:
    def test_returns_list(self, client_no_db):
        article = make_article()
        with patch(
            "gdelt_event_pipeline.api.routers.articles.get_recent_articles",
            return_value=[article],
        ):
            resp = client_no_db.get("/api/articles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 1

    def test_strips_internal_fields(self, client_no_db):
        article = make_article()
        article["embedding"] = [0.1, 0.2]
        article["raw_payload"] = {"big": "data"}
        with patch(
            "gdelt_event_pipeline.api.routers.articles.get_recent_articles",
            return_value=[article],
        ):
            resp = client_no_db.get("/api/articles")
        data = resp.json()[0]
        assert "embedding" not in data
        assert "raw_payload" not in data

    def test_limit_passed(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.articles.get_recent_articles",
            return_value=[],
        ) as mock:
            client_no_db.get("/api/articles?limit=25")
        assert mock.call_args.kwargs["limit"] == 25

    def test_limit_bounds(self, client_no_db):
        resp = client_no_db.get("/api/articles?limit=0")
        assert resp.status_code == 422
        resp = client_no_db.get("/api/articles?limit=201")
        assert resp.status_code == 422

    def test_filters_use_db_query(self, client_no_db):
        pool = make_mock_pool()
        pool._mock_cur.fetchall.return_value = []
        with patch(
            "gdelt_event_pipeline.storage.database.get_pool",
            return_value=pool,
        ):
            resp = client_no_db.get("/api/articles?location=Rome&theme=MILITARY_CONFLICT")
        assert resp.status_code == 200
        sql = pool._mock_cur.execute.call_args[0][0]
        # Filters now share build_filter_clauses → JSONB containment, not ILIKE
        assert "a.locations @>" in sql
        assert "a.themes @>" in sql

    def test_domain_filter_uses_soft_match(self, client_no_db):
        pool = make_mock_pool()
        pool._mock_cur.fetchall.return_value = []
        with patch(
            "gdelt_event_pipeline.storage.database.get_pool",
            return_value=pool,
        ):
            resp = client_no_db.get("/api/articles?domain=corriere.it,repubblica.it")
        assert resp.status_code == 200
        sql, params = pool._mock_cur.execute.call_args[0]
        assert "a.domain = ANY(%s)" in sql
        assert "a.domain LIKE ANY(%s)" in sql
        # Both CSV values must be honored (used to silently drop the second one)
        assert ["corriere.it", "repubblica.it"] in params
        assert ["%.corriere.it", "%.repubblica.it"] in params

    def test_filters_strip_internal_fields(self, client_no_db):
        pool = make_mock_pool()
        pool._mock_cur.fetchall.return_value = [
            make_article(embedding=[0.1], raw_payload={"x": 1}),
        ]
        with patch(
            "gdelt_event_pipeline.storage.database.get_pool",
            return_value=pool,
        ):
            resp = client_no_db.get("/api/articles?domain=bbc.com")
        data = resp.json()
        assert len(data) == 1
        assert "embedding" not in data[0]

    def test_no_filters_uses_storage_function(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.articles.get_recent_articles",
            return_value=[],
        ) as mock:
            resp = client_no_db.get("/api/articles")
        assert resp.status_code == 200
        mock.assert_called_once()


class TestGetStats:
    def test_returns_all_stat_fields(self, client_no_db):
        pool = make_mock_pool()
        pool._mock_cur.fetchone.side_effect = [
            {"cnt": 1000},
            {"cnt": 800},
            {"cnt": 600},
            {"cnt": 50},
            {"val": 42},
            {"cnt": 900},
        ]
        with patch(
            "gdelt_event_pipeline.storage.database.get_pool",
            return_value=pool,
        ):
            resp = client_no_db.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_articles"] == 1000
        assert data["titled_articles"] == 800
        assert data["embedded_articles"] == 600
        assert data["total_clusters"] == 50
        assert data["largest_cluster"] == 42
        assert data["total_memberships"] == 900

    def test_largest_cluster_defaults_to_zero(self, client_no_db):
        pool = make_mock_pool()
        pool._mock_cur.fetchone.side_effect = [
            {"cnt": 0},
            {"cnt": 0},
            {"cnt": 0},
            {"cnt": 0},
            {"val": None},
            {"cnt": 0},
        ]
        with patch(
            "gdelt_event_pipeline.storage.database.get_pool",
            return_value=pool,
        ):
            resp = client_no_db.get("/api/stats")
        assert resp.json()["largest_cluster"] == 0


class TestAuthConfig:
    def test_returns_clerk_key(self, client_no_db, monkeypatch):
        monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_123")
        resp = client_no_db.get("/api/auth/config")
        assert resp.status_code == 200
        assert resp.json()["clerk_publishable_key"] == "pk_test_123"

    def test_returns_empty_string_when_not_set(self, client_no_db, monkeypatch):
        monkeypatch.delenv("CLERK_PUBLISHABLE_KEY", raising=False)
        resp = client_no_db.get("/api/auth/config")
        assert resp.json()["clerk_publishable_key"] == ""
