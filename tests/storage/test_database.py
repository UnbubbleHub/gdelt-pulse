"""Tests for database connection pool management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gdelt_event_pipeline.storage.database as db_module


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the global pool before and after each test."""
    original = db_module._pool
    db_module._pool = None
    yield
    db_module._pool = original


class TestInitPool:
    def test_creates_pool(self):
        mock_settings = MagicMock()
        mock_settings.dsn = "postgresql://test:test@localhost/test"
        with patch("gdelt_event_pipeline.storage.database.ConnectionPool") as MockPool:
            pool = db_module.init_pool(mock_settings, min_size=1, max_size=5)
            MockPool.assert_called_once()
            assert pool is MockPool.return_value

    def test_idempotent_returns_same_pool(self):
        mock_settings = MagicMock()
        mock_settings.dsn = "postgresql://test:test@localhost/test"
        with patch("gdelt_event_pipeline.storage.database.ConnectionPool") as MockPool:
            pool1 = db_module.init_pool(mock_settings)
            pool2 = db_module.init_pool(mock_settings)
            assert pool1 is pool2
            MockPool.assert_called_once()

    def test_passes_min_max_size(self):
        mock_settings = MagicMock()
        mock_settings.dsn = "postgresql://test:test@localhost/test"
        with patch("gdelt_event_pipeline.storage.database.ConnectionPool") as MockPool:
            db_module.init_pool(mock_settings, min_size=0, max_size=2)
            call_kwargs = MockPool.call_args
            assert call_kwargs.kwargs["min_size"] == 0
            assert call_kwargs.kwargs["max_size"] == 2


class TestGetPool:
    def test_raises_when_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            db_module.get_pool()

    def test_returns_pool_when_initialized(self):
        mock_pool = MagicMock()
        db_module._pool = mock_pool
        assert db_module.get_pool() is mock_pool


class TestClosePool:
    def test_closes_and_clears(self):
        mock_pool = MagicMock()
        db_module._pool = mock_pool
        db_module.close_pool()
        mock_pool.close.assert_called_once()
        assert db_module._pool is None

    def test_idempotent_when_none(self):
        db_module._pool = None
        db_module.close_pool()
        assert db_module._pool is None
