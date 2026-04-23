"""Tests for /api/auth/keys endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import gdelt_event_pipeline.api.app as app_module
from gdelt_event_pipeline.api.auth import require_clerk_user


TEST_USER_ID = "user_test123"
TEST_NOW = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def client():
    """TestClient with DB lifecycle and require_clerk_user mocked out."""
    app_module.app.dependency_overrides[require_clerk_user] = lambda: TEST_USER_ID
    with (
        patch("gdelt_event_pipeline.api.app.init_pool"),
        patch("gdelt_event_pipeline.api.app.close_pool"),
        patch("gdelt_event_pipeline.api.app._ensure_schema"),
    ):
        with TestClient(app_module.app) as c:
            yield c
    app_module.app.dependency_overrides.clear()


def _mock_pool_no_key():
    """Return a mock pool where the user has no active key."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = None

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn
    return mock_pool


def _mock_pool_with_key(prefix="gdp_a1b2c3d4", created_at=TEST_NOW, last_used_at=None):
    """Return a mock pool where the user has an active key."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = {
        "key_prefix": prefix,
        "created_at": created_at,
        "last_used_at": last_used_at,
    }

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn
    return mock_pool


class TestGetKey:
    def test_returns_inactive_when_no_key(self, client):
        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=_mock_pool_no_key()):
            resp = client.get("/api/auth/keys")
        assert resp.status_code == 200
        assert resp.json() == {"active": False, "prefix": None, "created_at": None, "last_used_at": None}

    def test_returns_active_key_metadata(self, client):
        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=_mock_pool_with_key()):
            resp = client.get("/api/auth/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["prefix"] == "gdp_a1b2c3d4"
        assert data["last_used_at"] is None

    def test_requires_auth(self):
        """Without dependency override, missing Authorization header returns 422."""
        with (
            patch("gdelt_event_pipeline.api.app.init_pool"),
            patch("gdelt_event_pipeline.api.app.close_pool"),
            patch("gdelt_event_pipeline.api.app._ensure_schema"),
        ):
            with TestClient(app_module.app) as c:
                resp = c.get("/api/auth/keys")
        assert resp.status_code == 422  # missing required Header


class TestCreateKey:
    def test_creates_key_and_returns_full_key_once(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.post("/api/auth/keys")

        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("gdp_")
        assert len(data["key"]) == 36  # "gdp_" + 32 hex chars
        assert data["prefix"] == data["key"][:12]
        assert "created_at" in data

    def test_key_format_is_correct(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.post("/api/auth/keys")

        key = resp.json()["key"]
        assert key.startswith("gdp_")
        # The random part should be 32 lowercase hex chars
        random_part = key[4:]
        assert len(random_part) == 32
        assert all(c in "0123456789abcdef" for c in random_part)


class TestRevokeKey:
    def test_revokes_active_key(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = ("some-uuid",)  # RETURNING id

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.delete("/api/auth/keys")

        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}

    def test_returns_404_when_no_active_key(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = None  # no rows affected

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.delete("/api/auth/keys")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "No active key found."


class TestGenerateKeyHelper:
    def test_generate_key_returns_correct_tuple(self):
        from gdelt_event_pipeline.api.keys import _generate_key

        full_key, prefix, key_hash = _generate_key()

        assert full_key.startswith("gdp_")
        assert len(full_key) == 36
        assert prefix == full_key[:12]
        assert len(key_hash) == 64  # SHA-256 hex digest

    def test_generate_key_produces_unique_keys(self):
        from gdelt_event_pipeline.api.keys import _generate_key

        keys = {_generate_key()[0] for _ in range(10)}
        assert len(keys) == 10  # all unique
