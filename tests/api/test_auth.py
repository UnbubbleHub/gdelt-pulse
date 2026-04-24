"""Tests for Clerk JWT verification dependency."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def clear_jwks_cache():
    """Clear the lru_cache between tests so env changes take effect."""
    import gdelt_event_pipeline.api.auth as auth_module

    auth_module._jwks_client.cache_clear()
    yield
    auth_module._jwks_client.cache_clear()


def _call(authorization: str) -> str:
    """Call require_clerk_user directly (FastAPI Header default is just a default value)."""
    from gdelt_event_pipeline.api.auth import require_clerk_user

    return require_clerk_user(authorization=authorization)


class TestRequireClerkUser:
    def test_missing_bearer_prefix_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            _call("Basic abc123")
        assert exc_info.value.status_code == 401

    def test_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            _call("")
        assert exc_info.value.status_code == 401

    def test_invalid_jwt_raises_401(self, monkeypatch):
        monkeypatch.setenv(
            "CLERK_JWKS_URL", "https://example.clerk.accounts.dev/.well-known/jwks.json"
        )

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.side_effect = Exception("invalid token")

        with patch("gdelt_event_pipeline.api.auth._jwks_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                _call("Bearer invalid.token.here")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized."

    def test_valid_jwt_returns_user_id(self, monkeypatch):
        monkeypatch.setenv(
            "CLERK_JWKS_URL", "https://example.clerk.accounts.dev/.well-known/jwks.json"
        )

        mock_signing_key = MagicMock()
        mock_signing_key.key = "fake-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key

        with (
            patch("gdelt_event_pipeline.api.auth._jwks_client", return_value=mock_client),
            patch("jwt.decode", return_value={"sub": "user_abc123"}),
        ):
            user_id = _call("Bearer valid.token.here")

        assert user_id == "user_abc123"

    def test_missing_jwks_url_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("CLERK_JWKS_URL", raising=False)
        import gdelt_event_pipeline.api.auth as auth_module

        auth_module._jwks_client.cache_clear()

        with pytest.raises(RuntimeError, match="CLERK_JWKS_URL"):
            auth_module._jwks_client()
