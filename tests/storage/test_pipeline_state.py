"""Tests for pipeline state tracking operations."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from tests.conftest import make_mock_pool

MODULE = "gdelt_event_pipeline.storage.pipeline_state"


@pytest.fixture
def patch_pool():
    pool = make_mock_pool()
    with patch(f"{MODULE}.get_pool", return_value=pool):
        yield pool


class TestGetPipelineState:
    def test_default_source_name(self, patch_pool):
        from gdelt_event_pipeline.storage.pipeline_state import get_pipeline_state

        patch_pool._mock_cur.fetchone.return_value = {"source_name": "gdelt_gkg"}
        result = get_pipeline_state()
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ("gdelt_gkg",)
        assert result["source_name"] == "gdelt_gkg"

    def test_custom_source_name(self, patch_pool):
        from gdelt_event_pipeline.storage.pipeline_state import get_pipeline_state

        patch_pool._mock_cur.fetchone.return_value = None
        result = get_pipeline_state("custom_source")
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ("custom_source",)
        assert result is None


class TestUpdatePipelineState:
    def test_passes_timestamp_and_record_id(self, patch_pool):
        from gdelt_event_pipeline.storage.pipeline_state import update_pipeline_state

        ts = datetime(2026, 5, 1, tzinfo=UTC)
        update_pipeline_state(
            "gdelt_gkg", last_processed_timestamp=ts, last_processed_record_id="rec-1"
        )
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (ts, "rec-1", "gdelt_gkg")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "COALESCE" in sql

    def test_none_params_use_coalesce_fallback(self, patch_pool):
        from gdelt_event_pipeline.storage.pipeline_state import update_pipeline_state

        update_pipeline_state("gdelt_gkg")
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (None, None, "gdelt_gkg")
