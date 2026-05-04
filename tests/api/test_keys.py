"""Tests for /api/auth/keys endpoints."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import make_mock_pool


class TestGetKey:
    def test_returns_inactive_when_no_key(self, client_authed):
        pool = make_mock_pool(fetchone_return=None)
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.get("/api/auth/keys")
        assert resp.status_code == 200
        assert resp.json() == {
            "active": False,
            "prefix": None,
            "created_at": None,
            "last_used_at": None,
        }

    def test_returns_active_key_metadata(self, client_authed):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        pool = make_mock_pool(
            fetchone_return={
                "key_prefix": "gdp_a1b2c3d4",
                "created_at": now,
                "last_used_at": None,
            }
        )
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.get("/api/auth/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["prefix"] == "gdp_a1b2c3d4"
        assert data["last_used_at"] is None

    def test_requires_auth(self, client_no_db):
        """Without Clerk auth, missing Authorization header returns 422."""
        resp = client_no_db.get("/api/auth/keys")
        assert resp.status_code == 422


class TestCreateKey:
    def test_creates_key_and_returns_full_key_once(self, client_authed):
        pool = make_mock_pool()
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.post("/api/auth/keys")

        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("gdp_")
        assert len(data["key"]) == 36
        assert data["prefix"] == data["key"][:12]
        assert "created_at" in data

    def test_key_format_is_correct(self, client_authed):
        pool = make_mock_pool()
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.post("/api/auth/keys")

        key = resp.json()["key"]
        assert key.startswith("gdp_")
        random_part = key[4:]
        assert len(random_part) == 32
        assert all(c in "0123456789abcdef" for c in random_part)


class TestRevokeKey:
    def test_revokes_active_key(self, client_authed):
        pool = make_mock_pool(fetchone_return=("some-uuid",))
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.delete("/api/auth/keys")

        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}

    def test_returns_404_when_no_active_key(self, client_authed):
        pool = make_mock_pool(fetchone_return=None)
        with patch("gdelt_event_pipeline.api.routers.keys.get_pool", return_value=pool):
            resp = client_authed.delete("/api/auth/keys")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "No active key found."


class TestGenerateKeyHelper:
    def test_generate_key_returns_correct_tuple(self):
        from gdelt_event_pipeline.api.routers.keys import _generate_key

        full_key, prefix, key_hash = _generate_key()

        assert full_key.startswith("gdp_")
        assert len(full_key) == 36
        assert prefix == full_key[:12]
        assert len(key_hash) == 64

    def test_generate_key_produces_unique_keys(self):
        from gdelt_event_pipeline.api.routers.keys import _generate_key

        keys = {_generate_key()[0] for _ in range(10)}
        assert len(keys) == 10
