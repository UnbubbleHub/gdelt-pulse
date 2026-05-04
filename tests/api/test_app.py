"""Tests for app.py behaviour specific to the Vercel deployment split."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import gdelt_event_pipeline.api.app as app_module
import gdelt_event_pipeline.api.middleware as middleware_module
import gdelt_event_pipeline.api.routers.search as search_module
from tests.conftest import make_mock_pool


class TestSearchGuard:
    def test_search_returns_501_when_search_unavailable(self, client_no_db, monkeypatch):
        """When _SEARCH_AVAILABLE is False the endpoint must return HTTP 501."""
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)
        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501
        assert "not available" in response.json()["detail"].lower()

    def test_search_available_flag_true_in_full_environment(self):
        """fastembed is in the main dependencies, so the module-level flag must be True."""
        assert search_module._SEARCH_AVAILABLE is True

    def test_search_returns_501_when_search_available_flag_is_false(
        self, client_no_db, monkeypatch
    ):
        """When _SEARCH_AVAILABLE is False, /api/search must return 501 regardless of backend."""
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)
        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501


class TestServerlessPoolSizing:
    def test_pool_uses_serverless_sizes_when_vercel_env_set(self, monkeypatch):
        """When VERCEL=1 the lifespan must call init_pool with min_size=0, max_size=2."""
        monkeypatch.setenv("VERCEL", "1")
        with (
            patch("gdelt_event_pipeline.api.app.init_pool") as mock_init,
            patch("gdelt_event_pipeline.api.app.close_pool"),
            patch("gdelt_event_pipeline.api.app.ensure_schema"),
        ):
            with TestClient(app_module.app):
                pass
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["min_size"] == 0
        assert mock_init.call_args.kwargs["max_size"] == 2

    def test_pool_uses_standard_sizes_without_vercel_env(self, monkeypatch):
        """Without VERCEL env the lifespan must call init_pool with min_size=2, max_size=10."""
        monkeypatch.delenv("VERCEL", raising=False)
        with (
            patch("gdelt_event_pipeline.api.app.init_pool") as mock_init,
            patch("gdelt_event_pipeline.api.app.close_pool"),
            patch("gdelt_event_pipeline.api.app.ensure_schema"),
        ):
            with TestClient(app_module.app):
                pass
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["min_size"] == 2
        assert mock_init.call_args.kwargs["max_size"] == 10


class TestRateLimiting:
    def test_redis_rate_limit_returns_429_when_over_limit(self, client_no_db, monkeypatch):
        """When Redis pipeline reports count >= RATE_LIMIT_MAX, middleware returns 429."""
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [0, 30, 1, 1]

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        monkeypatch.setattr(middleware_module, "_redis", mock_redis)

        response = client_no_db.get("/api/clusters")
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()

    def test_redis_rate_limit_passes_when_under_limit(self, client_no_db, monkeypatch):
        """When Redis pipeline reports count < RATE_LIMIT_MAX, middleware passes through."""
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [0, 5, 1, 1]

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        monkeypatch.setattr(middleware_module, "_redis", mock_redis)
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)

        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501

    def test_falls_back_to_in_memory_when_redis_not_configured(self, client_no_db, monkeypatch):
        """When _redis is None (no Upstash credentials), middleware uses in-memory store."""
        monkeypatch.setattr(middleware_module, "_redis", None)
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)

        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501


class TestApiKeyAuth:
    def test_no_key_header_passes_with_anonymous_limit(self, client_no_db, monkeypatch):
        """Requests with no X-API-Key go through with the anonymous rate limit."""
        monkeypatch.setattr(middleware_module, "_redis", None)
        monkeypatch.setattr(search_module, "_SEARCH_AVAILABLE", False)

        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501

    def test_invalid_key_returns_401(self, client_no_db, monkeypatch):
        """When X-API-Key does not match any active key in DB, return 401."""
        monkeypatch.setattr(middleware_module, "_redis", None)

        pool = make_mock_pool(fetchone_return=None)

        import gdelt_event_pipeline.storage.database as db_module

        with patch.object(db_module, "get_pool", return_value=pool):
            response = client_no_db.get(
                "/api/search?q=test", headers={"X-API-Key": "gdp_invalidkey"}
            )

        assert response.status_code == 401
        assert "api key" in response.json()["detail"].lower()

    def test_auth_endpoints_skip_middleware(self, client_no_db):
        """/api/auth/* paths bypass the key check and rate limiter."""
        response = client_no_db.get("/api/auth/keys")
        assert response.status_code == 422

    def test_static_pages_not_protected(self, client_no_db):
        """Static routes (/, etc.) must be accessible without any API key."""
        response = client_no_db.get("/")
        assert response.status_code == 200


class TestDeveloperExperience:
    def test_swagger_ui_route_returns_200(self, client_no_db):
        """GET /api/docs must return 200 (Swagger UI enabled)."""
        response = client_no_db.get("/api/docs")
        assert response.status_code == 200

    def test_openapi_json_route_returns_200(self, client_no_db):
        """GET /api/openapi.json must return valid OpenAPI JSON."""
        response = client_no_db.get("/api/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_developers_page_returns_200(self, client_no_db):
        """GET /developers must return 200 (static HTML served)."""
        response = client_no_db.get("/developers")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_cors_allows_any_origin(self, client_no_db):
        """CORS preflight must respond with Access-Control-Allow-Origin: * for /api/ paths."""
        response = client_no_db.options(
            "/api/stats",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "*"
