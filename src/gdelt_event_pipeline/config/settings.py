"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = field(default_factory=lambda: os.environ.get("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("PGPORT", "5432")))
    user: str = field(default_factory=lambda: os.environ.get("PGUSER", "postgres"))
    password: str = field(default_factory=lambda: os.environ.get("PGPASSWORD", ""))
    database: str = field(default_factory=lambda: os.environ.get("PGDATABASE", "gdelt_pulse"))

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


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


def get_settings() -> Settings:
    return Settings()
