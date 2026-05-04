"""Tests for application settings."""

from __future__ import annotations

from gdelt_event_pipeline.config.settings import (
    ClusteringSettings,
    DatabaseSettings,
    EmbeddingSettings,
    Settings,
)


class TestDatabaseSettings:
    def test_dsn_from_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
        s = DatabaseSettings()
        assert s.dsn == "postgresql://u:p@host/db"

    def test_dsn_from_public_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DATABASE_PUBLIC_URL", "postgresql://pub:p@host/db")
        s = DatabaseSettings()
        assert s.dsn == "postgresql://pub:p@host/db"

    def test_dsn_from_components(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_PUBLIC_URL", raising=False)
        monkeypatch.setenv("PGHOST", "myhost")
        monkeypatch.setenv("PGPORT", "5433")
        monkeypatch.setenv("PGUSER", "myuser")
        monkeypatch.setenv("PGPASSWORD", "secret")
        monkeypatch.setenv("PGDATABASE", "mydb")
        s = DatabaseSettings()
        assert "myhost" in s.dsn
        assert "5433" in s.dsn
        assert "myuser" in s.dsn
        assert "mydb" in s.dsn

    def test_special_chars_in_password(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_PUBLIC_URL", raising=False)
        monkeypatch.setenv("PGPASSWORD", "p@ss:w0rd/special")
        s = DatabaseSettings()
        assert "p%40ss%3Aw0rd%2Fspecial" in s.dsn

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_PUBLIC_URL", raising=False)
        monkeypatch.delenv("PGHOST", raising=False)
        monkeypatch.delenv("PGPORT", raising=False)
        monkeypatch.delenv("PGUSER", raising=False)
        monkeypatch.delenv("PGPASSWORD", raising=False)
        monkeypatch.delenv("PGDATABASE", raising=False)
        s = DatabaseSettings()
        assert "localhost" in s.dsn
        assert "5432" in s.dsn

    def test_url_takes_precedence_over_components(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://url-wins@host/db")
        monkeypatch.setenv("PGHOST", "component-host")
        s = DatabaseSettings()
        assert s.dsn == "postgresql://url-wins@host/db"


class TestEmbeddingSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
        monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
        s = EmbeddingSettings()
        assert "MiniLM" in s.model_name
        assert s.dimension == 384
        assert s.batch_size == 64

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
        monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "32")
        s = EmbeddingSettings()
        assert s.model_name == "custom/model"
        assert s.dimension == 768
        assert s.batch_size == 32


class TestClusteringSettings:
    def test_default_window(self, monkeypatch):
        monkeypatch.delenv("CLUSTER_WINDOW_HOURS", raising=False)
        s = ClusteringSettings()
        assert s.window_hours == 72

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CLUSTER_WINDOW_HOURS", "48")
        s = ClusteringSettings()
        assert s.window_hours == 48


class TestSettings:
    def test_composes_all_subsettings(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test@host/db")
        s = Settings()
        assert isinstance(s.db, DatabaseSettings)
        assert isinstance(s.embedding, EmbeddingSettings)
        assert isinstance(s.clustering, ClusteringSettings)
