"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = field(default_factory=lambda: os.environ.get("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("PGPORT", "5432")))
    user: str = field(default_factory=lambda: os.environ.get("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: os.environ.get("PGPASSWORD", ""))
    database: str = field(default_factory=lambda: os.environ.get("PGDATABASE", "gdelt_pulse"))
    url: str = field(
        default_factory=lambda: (
            os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL") or ""
        )
    )

    @property
    def dsn(self) -> str:
        if self.url:
            return self.url
        return (
            f"postgresql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dimension: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIMENSION", "384"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
    )
    # Cap embedding work per cycle so a constrained container cannot OOM
    # by trying to embed thousands of articles in a single call.
    per_cycle_limit: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_PER_CYCLE_LIMIT", "300"))
    )


@dataclass(frozen=True)
class RetentionSettings:
    hours: int = field(default_factory=lambda: int(os.environ.get("RETENTION_HOURS", "72")))


@dataclass(frozen=True)
class ClusteringSettings:
    window_hours: int = field(
        default_factory=lambda: int(os.environ.get("CLUSTER_WINDOW_HOURS", "72"))
    )


@dataclass(frozen=True)
class Settings:
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    retention: RetentionSettings = field(default_factory=RetentionSettings)


def get_settings() -> Settings:
    return Settings()
