"""Tests for pipeline runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.conftest import make_mock_pool

MODULE = "gdelt_event_pipeline.runner"


class TestCleanupFailedArticles:
    def test_deletes_articles(self):
        from gdelt_event_pipeline.runner import _cleanup_failed_articles

        pool = make_mock_pool(rowcount=5)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_failed_articles()
        assert result == 5
        sql = pool._mock_cur.execute.call_args[0][0]
        assert "DELETE FROM articles" in sql
        assert "title IS NULL" in sql
        assert "scrape_attempts >= 1" in sql

    def test_returns_zero_when_none_deleted(self):
        from gdelt_event_pipeline.runner import _cleanup_failed_articles

        pool = make_mock_pool(rowcount=0)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_failed_articles()
        assert result == 0


class TestCleanupOldArticles:
    def test_deletes_old_articles(self):
        from gdelt_event_pipeline.runner import _cleanup_old_articles

        pool = make_mock_pool(rowcount=12)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_old_articles(168)
        assert result == 12
        sql = pool._mock_cur.execute.call_args[0][0]
        assert "DELETE FROM articles" in sql
        assert "published_at" in sql

    def test_returns_zero_when_none_deleted(self):
        from gdelt_event_pipeline.runner import _cleanup_old_articles

        pool = make_mock_pool(rowcount=0)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_old_articles(168)
        assert result == 0

    def test_disabled_when_zero(self):
        from gdelt_event_pipeline.runner import _cleanup_old_articles

        result = _cleanup_old_articles(0)
        assert result == 0


class TestRunCycle:
    def test_runs_all_stages(self):
        from gdelt_event_pipeline.runner import run_cycle

        mock_settings = MagicMock()
        mock_settings.embedding = MagicMock()
        mock_settings.clustering.window_hours = 72
        mock_settings.retention.hours = 168

        ing_result = MagicMock()
        ing_result.rows_fetched = 10
        ing_result.rows_upserted = 8
        ing_result.duplicate_urls = 2
        ing_result.rows_failed = 0

        emb_result = MagicMock()
        emb_result.articles_fetched = 5
        emb_result.articles_embedded = 5
        emb_result.articles_skipped = 0
        emb_result.articles_failed = 0

        cl_result = MagicMock()
        cl_result.articles_processed = 5
        cl_result.assigned_to_existing = 3
        cl_result.new_clusters_created = 2
        cl_result.articles_failed = 0

        pool = make_mock_pool(rowcount=0)

        with (
            patch(f"{MODULE}.run_ingestion", return_value=ing_result),
            patch(f"{MODULE}.run_title_scraping", return_value=(5, 4)),
            patch(f"{MODULE}.run_embedding", return_value=emb_result),
            patch(f"{MODULE}.run_clustering", return_value=cl_result),
            patch(f"{MODULE}.get_pool", return_value=pool),
        ):
            summary = run_cycle(mock_settings)

        assert "ingest" in summary
        assert "titles" in summary
        assert "embed" in summary
        assert "cluster" in summary
        assert "ERROR" not in summary["ingest"]
        assert "ERROR" not in summary["embed"]

    def test_stage_failure_isolated(self):
        from gdelt_event_pipeline.runner import run_cycle

        mock_settings = MagicMock()
        mock_settings.embedding = MagicMock()
        mock_settings.clustering.window_hours = 72

        pool = make_mock_pool(rowcount=0)

        with (
            patch(f"{MODULE}.run_ingestion", side_effect=RuntimeError("boom")),
            patch(f"{MODULE}.run_title_scraping", return_value=(0, 0)),
            patch(f"{MODULE}.run_embedding", side_effect=RuntimeError("boom")),
            patch(f"{MODULE}.run_clustering", side_effect=RuntimeError("boom")),
            patch(f"{MODULE}.get_pool", return_value=pool),
        ):
            summary = run_cycle(mock_settings)

        assert summary["ingest"] == "ERROR"
        assert summary["embed"] == "ERROR"
        assert summary["cluster"] == "ERROR"
        assert "titles" in summary

    def test_no_gravity_refresh(self):
        """Gravity refresh was removed — verify it's not in the summary."""
        from gdelt_event_pipeline.runner import run_cycle

        mock_settings = MagicMock()
        mock_settings.embedding = MagicMock()
        mock_settings.clustering.window_hours = 72

        pool = make_mock_pool(rowcount=0)

        with (
            patch(f"{MODULE}.run_ingestion", return_value=MagicMock()),
            patch(f"{MODULE}.run_title_scraping", return_value=(0, 0)),
            patch(f"{MODULE}.run_embedding", return_value=MagicMock()),
            patch(f"{MODULE}.run_clustering", return_value=MagicMock()),
            patch(f"{MODULE}.get_pool", return_value=pool),
        ):
            summary = run_cycle(mock_settings)

        assert "gravity_refresh" not in summary


class TestBackoffLogic:
    """Test the backoff functions defined inside main().

    Since _next_wait and _is_failed_cycle are closures, we recreate
    their logic here to validate the algorithm.
    """

    def test_next_wait_within_grace_period(self):
        grace = 3
        interval = 900
        backoff_start = 30 * 60

        def _next_wait(failures):
            if failures <= grace:
                return interval
            step = failures - grace
            return min(backoff_start * (2 ** (step - 1)), 120 * 60)

        assert _next_wait(0) == interval
        assert _next_wait(1) == interval
        assert _next_wait(3) == interval

    def test_next_wait_after_grace(self):
        grace = 3
        interval = 900
        backoff_start = 30 * 60
        cap = 120 * 60

        def _next_wait(failures):
            if failures <= grace:
                return interval
            step = failures - grace
            return min(backoff_start * (2 ** (step - 1)), cap)

        assert _next_wait(4) == 30 * 60
        assert _next_wait(5) == 60 * 60
        assert _next_wait(6) == 120 * 60
        assert _next_wait(10) == cap

    def test_is_failed_cycle_all_errors(self):
        def _is_failed(summary):
            tracked = {v for k, v in summary.items() if k != "cleanup"}
            return bool(tracked) and all(v == "ERROR" for v in tracked)

        assert _is_failed({"ingest": "ERROR", "embed": "ERROR"}) is True

    def test_is_failed_cycle_partial_success(self):
        def _is_failed(summary):
            tracked = {v for k, v in summary.items() if k != "cleanup"}
            return bool(tracked) and all(v == "ERROR" for v in tracked)

        assert _is_failed({"ingest": "ok", "embed": "ERROR"}) is False

    def test_is_failed_cycle_cleanup_excluded(self):
        def _is_failed(summary):
            tracked = {v for k, v in summary.items() if k != "cleanup"}
            return bool(tracked) and all(v == "ERROR" for v in tracked)

        assert _is_failed({"ingest": "ERROR", "cleanup": "deleted=3"}) is True

    def test_is_failed_cycle_empty_summary(self):
        def _is_failed(summary):
            tracked = {v for k, v in summary.items() if k != "cleanup"}
            return bool(tracked) and all(v == "ERROR" for v in tracked)

        assert _is_failed({}) is False


class TestEnsureSchemaIntegration:
    def test_runner_uses_shared_ensure_schema(self):
        """Verify runner imports ensure_schema from storage.migrations."""
        import gdelt_event_pipeline.runner as runner_mod

        assert hasattr(runner_mod, "ensure_schema")
        from gdelt_event_pipeline.storage.migrations import (
            ensure_schema as migrations_fn,
        )

        assert runner_mod.ensure_schema is migrations_fn
