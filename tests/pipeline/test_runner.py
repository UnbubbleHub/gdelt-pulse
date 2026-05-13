"""Tests for pipeline runner (micro-cycle architecture)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.conftest import make_mock_pool

MODULE = "gdelt_event_pipeline.runner"


# ── Cleanup helpers (preserved from cycle-based runner) ────────────


class TestCleanupOldArticles:
    def test_deletes_old_articles(self):
        from gdelt_event_pipeline.runner import _cleanup_old_articles

        pool = make_mock_pool(rowcount=12)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_old_articles(168)
        assert result == 12
        sql = pool._mock_cur.execute.call_args[0][0]
        assert "DELETE FROM articles" in sql
        assert "gdelt_timestamp" in sql
        params = pool._mock_cur.execute.call_args[0][1]
        assert params == (168,)

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


class TestCleanupOrphanClusters:
    def test_deletes_orphan_clusters(self):
        from gdelt_event_pipeline.runner import _cleanup_orphan_clusters

        pool = make_mock_pool(rowcount=42)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_orphan_clusters()
        assert result == 42
        sql = pool._mock_cur.execute.call_args[0][0]
        assert "DELETE FROM clusters" in sql
        assert "cluster_memberships" in sql

    def test_returns_zero_when_none_deleted(self):
        from gdelt_event_pipeline.runner import _cleanup_orphan_clusters

        pool = make_mock_pool(rowcount=0)
        with patch(f"{MODULE}.get_pool", return_value=pool):
            result = _cleanup_orphan_clusters()
        assert result == 0


# ── Stage helpers (new in micro-cycle architecture) ────────────────


class TestDoIngest:
    def test_success(self):
        from gdelt_event_pipeline.runner import _do_ingest

        ing = MagicMock(rows_fetched=10, rows_upserted=8, duplicate_urls=2, rows_failed=0)
        with (
            patch(f"{MODULE}.run_ingestion", return_value=ing),
            patch(f"{MODULE}.run_title_scraping", return_value=(5, 4)),
        ):
            ok, summary = _do_ingest(MagicMock())
        assert ok is True
        assert "fetched=10" in summary["ingest"]
        assert "upserted=8" in summary["ingest"]
        assert "attempted=5" in summary["titles"]
        assert "succeeded=4" in summary["titles"]

    def test_ingestion_failure(self):
        from gdelt_event_pipeline.runner import _do_ingest

        with (
            patch(f"{MODULE}.run_ingestion", side_effect=RuntimeError("boom")),
            patch(f"{MODULE}.run_title_scraping", return_value=(0, 0)),
        ):
            ok, summary = _do_ingest(MagicMock())
        assert ok is False
        assert summary["ingest"] == "ERROR"
        # Title scrape still runs and reports
        assert summary["titles"] == "attempted=0 succeeded=0"

    def test_scrape_failure(self):
        from gdelt_event_pipeline.runner import _do_ingest

        ing = MagicMock(rows_fetched=1, rows_upserted=1, duplicate_urls=0, rows_failed=0)
        with (
            patch(f"{MODULE}.run_ingestion", return_value=ing),
            patch(f"{MODULE}.run_title_scraping", side_effect=RuntimeError("net")),
        ):
            ok, summary = _do_ingest(MagicMock())
        assert ok is False
        assert summary["ingest"].startswith("fetched=1")
        assert summary["titles"] == "ERROR"


class TestDoEmbed:
    def test_success(self):
        from gdelt_event_pipeline.runner import _do_embed

        emb = MagicMock(
            articles_fetched=5, articles_embedded=4, articles_skipped=1, articles_failed=0
        )
        with patch(f"{MODULE}.run_embedding", return_value=emb) as p:
            ok, count, summary = _do_embed(limit=200)
        assert ok is True
        assert count == 4
        assert "embedded=4" in summary["embed"]
        # Confirm limit is passed through
        assert p.call_args.kwargs.get("limit") == 200

    def test_zero_embedded_returns_ok(self):
        from gdelt_event_pipeline.runner import _do_embed

        emb = MagicMock(
            articles_fetched=0, articles_embedded=0, articles_skipped=0, articles_failed=0
        )
        with patch(f"{MODULE}.run_embedding", return_value=emb):
            ok, count, _ = _do_embed(limit=200)
        assert ok is True
        assert count == 0

    def test_failure(self):
        from gdelt_event_pipeline.runner import _do_embed

        with patch(f"{MODULE}.run_embedding", side_effect=RuntimeError("boom")):
            ok, count, summary = _do_embed(limit=200)
        assert ok is False
        assert count == 0
        assert summary["embed"] == "ERROR"


class TestDoCluster:
    def test_success(self):
        from gdelt_event_pipeline.runner import _do_cluster

        cl = MagicMock(
            articles_processed=5,
            assigned_to_existing=3,
            new_clusters_created=2,
            articles_failed=0,
        )
        with patch(f"{MODULE}.run_clustering", return_value=cl) as p:
            ok, summary = _do_cluster(limit=400, window_hours=72)
        assert ok is True
        assert "processed=5" in summary["cluster"]
        assert p.call_args.kwargs.get("limit") == 400
        assert p.call_args.kwargs.get("max_age_hours") == 72

    def test_failure(self):
        from gdelt_event_pipeline.runner import _do_cluster

        with patch(f"{MODULE}.run_clustering", side_effect=RuntimeError("boom")):
            ok, summary = _do_cluster(limit=400, window_hours=72)
        assert ok is False
        assert summary["cluster"] == "ERROR"


class TestDoCleanup:
    def test_reports_when_both_deletes_happen(self):
        from gdelt_event_pipeline.runner import _do_cleanup

        with (
            patch(f"{MODULE}._cleanup_old_articles", return_value=12),
            patch(f"{MODULE}._cleanup_orphan_clusters", return_value=3),
        ):
            summary = _do_cleanup(retention_hours=168)
        assert summary["retention"] == "deleted=12"
        assert summary["orphans"] == "deleted=3"

    def test_omits_keys_when_nothing_deleted(self):
        from gdelt_event_pipeline.runner import _do_cleanup

        with (
            patch(f"{MODULE}._cleanup_old_articles", return_value=0),
            patch(f"{MODULE}._cleanup_orphan_clusters", return_value=0),
        ):
            summary = _do_cleanup(retention_hours=168)
        assert "retention" not in summary
        assert "orphans" not in summary

    def test_swallows_exceptions(self):
        from gdelt_event_pipeline.runner import _do_cleanup

        with (
            patch(f"{MODULE}._cleanup_old_articles", side_effect=RuntimeError("boom")),
            patch(f"{MODULE}._cleanup_orphan_clusters", return_value=5),
        ):
            summary = _do_cleanup(retention_hours=168)
        # Retention errored; orphan-cluster cleanup still ran and reported.
        assert summary.get("orphans") == "deleted=5"
        assert "retention" not in summary


# ── Config / env wiring ────────────────────────────────────────────


_RUNNER_ENV_VARS = (
    "PIPELINE_TICK_SECONDS",
    "INGEST_INTERVAL_SECONDS",
    "EMBED_PER_TICK",
    "CLUSTER_PER_TICK",
    "CLEANUP_INTERVAL_SECONDS",
    "PIPELINE_INTERVAL",
    "HEALTHCHECKS_PING_URL",
)


class TestRunnerConfig:
    def test_defaults(self, monkeypatch):
        for k in _RUNNER_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        from gdelt_event_pipeline.runner import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.tick_seconds == 60
        assert cfg.ingest_interval == 15 * 60
        assert cfg.embed_per_tick == 200
        assert cfg.cluster_per_tick == 400
        assert cfg.cleanup_interval == 60 * 60
        assert cfg.healthcheck_url == ""

    def test_legacy_pipeline_interval_alias(self, monkeypatch):
        for k in _RUNNER_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PIPELINE_INTERVAL", "1800")
        from gdelt_event_pipeline.runner import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.ingest_interval == 1800

    def test_new_var_overrides_legacy(self, monkeypatch):
        for k in _RUNNER_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PIPELINE_INTERVAL", "9999")
        monkeypatch.setenv("INGEST_INTERVAL_SECONDS", "600")
        from gdelt_event_pipeline.runner import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.ingest_interval == 600

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        for k in _RUNNER_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("EMBED_PER_TICK", "not-a-number")
        from gdelt_event_pipeline.runner import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.embed_per_tick == 200

    def test_healthcheck_url_trimmed(self, monkeypatch):
        for k in _RUNNER_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("HEALTHCHECKS_PING_URL", "  https://hc-ping.com/abc  ")
        from gdelt_event_pipeline.runner import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.healthcheck_url == "https://hc-ping.com/abc"


# ── Integration touch points ───────────────────────────────────────


class TestEnsureSchemaIntegration:
    def test_runner_uses_shared_ensure_schema(self):
        """Verify runner imports ensure_schema from storage.migrations."""
        import gdelt_event_pipeline.runner as runner_mod

        assert hasattr(runner_mod, "ensure_schema")
        from gdelt_event_pipeline.storage.migrations import (
            ensure_schema as migrations_fn,
        )

        assert runner_mod.ensure_schema is migrations_fn
