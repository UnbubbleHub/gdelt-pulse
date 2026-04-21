"""Tests for app.py behaviour specific to the Vercel deployment split."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import gdelt_event_pipeline.api.app as app_module


@pytest.fixture
def client_no_db():
    """TestClient with DB lifecycle mocked out so no real connection is needed."""
    with (
        patch("gdelt_event_pipeline.api.app.init_pool"),
        patch("gdelt_event_pipeline.api.app.close_pool"),
        patch("gdelt_event_pipeline.api.app._ensure_schema"),
    ):
        with TestClient(app_module.app) as client:
            yield client


class TestSearchGuard:
    def test_search_returns_501_when_search_unavailable(self, client_no_db, monkeypatch):
        """When _SEARCH_AVAILABLE is False the endpoint must return HTTP 501."""
        monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501
        assert "not available" in response.json()["detail"].lower()

    def test_search_available_flag_true_in_full_environment(self):
        """In the dev/Railway environment sentence_transformers is installed,
        so the module-level flag must be True."""
        assert app_module._SEARCH_AVAILABLE is True


class TestServerlessPoolSizing:
    def test_pool_uses_serverless_sizes_when_vercel_env_set(self, monkeypatch):
        """When VERCEL=1 the lifespan must call init_pool with min_size=0, max_size=2."""
        monkeypatch.setenv("VERCEL", "1")
        with (
            patch("gdelt_event_pipeline.api.app.init_pool") as mock_init,
            patch("gdelt_event_pipeline.api.app.close_pool"),
            patch("gdelt_event_pipeline.api.app._ensure_schema"),
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
            patch("gdelt_event_pipeline.api.app._ensure_schema"),
        ):
            with TestClient(app_module.app):
                pass
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["min_size"] == 2
        assert mock_init.call_args.kwargs["max_size"] == 10
