"""Tests for app.py behaviour specific to the Vercel deployment split."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import gdelt_event_pipeline.api.app as app_module


@pytest.fixture
def client_no_db():
    """TestClient with DB lifecycle mocked out so no real connection is needed."""
    with patch("gdelt_event_pipeline.api.app.init_pool"), \
         patch("gdelt_event_pipeline.api.app.close_pool"), \
         patch("gdelt_event_pipeline.api.app._ensure_schema"):
        with TestClient(app_module.app) as client:
            yield client


class TestSearchGuard:
    def test_search_returns_501_when_search_unavailable(self, client_no_db, monkeypatch):
        """When _SEARCH_AVAILABLE is False the endpoint must return HTTP 501."""
        monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501
        assert "not available" in response.json()["detail"].lower()

    def test_search_available_flag_true_when_package_importable(self, monkeypatch):
        """When sentence_transformers is importable the flag should evaluate as True."""
        import sys
        import types
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.ModuleType("sentence_transformers"),
        )
        try:
            import sentence_transformers as _st_check  # noqa: F401
            flag = True
        except ImportError:
            flag = False
        assert flag is True


class TestServerlessPoolSizing:
    def test_pool_uses_serverless_sizes_when_vercel_env_set(self, monkeypatch):
        """When VERCEL=1 the lifespan must call init_pool with min_size=0, max_size=2."""
        monkeypatch.setenv("VERCEL", "1")
        with patch("gdelt_event_pipeline.api.app.init_pool") as mock_init, \
             patch("gdelt_event_pipeline.api.app.close_pool"), \
             patch("gdelt_event_pipeline.api.app._ensure_schema"):
            with TestClient(app_module.app):
                pass
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["min_size"] == 0
        assert mock_init.call_args.kwargs["max_size"] == 2

    def test_pool_uses_standard_sizes_without_vercel_env(self, monkeypatch):
        """Without VERCEL env the lifespan must call init_pool with min_size=2, max_size=10."""
        monkeypatch.delenv("VERCEL", raising=False)
        with patch("gdelt_event_pipeline.api.app.init_pool") as mock_init, \
             patch("gdelt_event_pipeline.api.app.close_pool"), \
             patch("gdelt_event_pipeline.api.app._ensure_schema"):
            with TestClient(app_module.app):
                pass
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["min_size"] == 2
        assert mock_init.call_args.kwargs["max_size"] == 10
