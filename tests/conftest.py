"""Shared test fixtures for the GDELT Pulse test suite."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_USER_ID = "user_test123"


# ── Mock database fixtures ──────────────────────────────────────────


def make_mock_pool(
    *,
    fetchone_return=None,
    fetchall_return=None,
    rowcount=0,
):
    """Build a mock connection pool with pre-configured cursor returns.

    The returned pool supports the `pool.connection() -> conn -> cursor` pattern
    used throughout the storage layer.
    """
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = fetchone_return
    mock_cur.fetchall.return_value = fetchall_return or []
    mock_cur.rowcount = rowcount

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn

    mock_pool._mock_conn = mock_conn
    mock_pool._mock_cur = mock_cur
    return mock_pool


@pytest.fixture
def mock_pool():
    """A fresh mock pool per test — configure fetchone/fetchall as needed."""
    return make_mock_pool()


@pytest.fixture
def patch_pool(mock_pool):
    """Patch get_pool globally so all storage modules see the mock."""
    with patch(
        "gdelt_event_pipeline.storage.database.get_pool",
        return_value=mock_pool,
    ):
        yield mock_pool


# ── FastAPI TestClient fixtures ─────────────────────────────────────


@pytest.fixture
def client_no_db():
    """TestClient with DB lifecycle mocked out — no real connection needed."""
    import gdelt_event_pipeline.api.app as app_module

    with (
        patch("gdelt_event_pipeline.api.app.init_pool"),
        patch("gdelt_event_pipeline.api.app.close_pool"),
        patch("gdelt_event_pipeline.api.app.ensure_schema"),
    ):
        with TestClient(app_module.app) as client:
            yield client


@pytest.fixture
def client_authed():
    """TestClient with DB lifecycle mocked + Clerk auth returning TEST_USER_ID."""
    import gdelt_event_pipeline.api.app as app_module
    from gdelt_event_pipeline.api.auth import require_clerk_user

    app_module.app.dependency_overrides[require_clerk_user] = lambda: TEST_USER_ID
    with (
        patch("gdelt_event_pipeline.api.app.init_pool"),
        patch("gdelt_event_pipeline.api.app.close_pool"),
        patch("gdelt_event_pipeline.api.app.ensure_schema"),
    ):
        with TestClient(app_module.app) as client:
            yield client
    app_module.app.dependency_overrides.clear()


# ── Sample data factories ───────────────────────────────────────────


def make_article(**overrides) -> dict:
    """Return a dict matching the articles table schema with sensible defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "gkg_record_id": f"gkg-{uuid.uuid4().hex[:8]}",
        "gdelt_timestamp": datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
        "url": "https://example.com/article-1",
        "canonical_url": f"example.com/article-{uuid.uuid4().hex[:6]}",
        "domain": "example.com",
        "source_common_name": "Example News",
        "canonical_source": "example-news",
        "title": "Test Article Title",
        "themes": [{"theme": "POLITICS"}],
        "locations": [{"name": "Washington", "country_code": "US", "lat": 38.9, "lon": -77.0}],
        "organizations": [{"name": "UN"}],
        "persons": [{"name": "John Doe"}],
        "all_names": ["John Doe", "UN"],
        "tone": {"tone": -1.5, "positive_score": 2.0, "negative_score": 3.5, "polarity": 5.5},
        "scrape_attempts": 0,
        "embedding": None,
        "embedding_model": None,
        "title_tsv": None,
        "raw_payload": None,
        "first_seen_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
        "last_seen_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
        "created_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return defaults


def make_cluster(**overrides) -> dict:
    """Return a dict matching the clusters table schema with sensible defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "representative_title": "Test Cluster Title",
        "summary": None,
        "centroid_embedding": None,
        "article_count": 5,
        "first_article_at": datetime(2026, 4, 20, 10, 0, 0, tzinfo=UTC),
        "last_article_at": datetime(2026, 4, 20, 14, 0, 0, tzinfo=UTC),
        "is_active": True,
        "created_at": datetime(2026, 4, 20, 10, 0, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 4, 20, 14, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return defaults
