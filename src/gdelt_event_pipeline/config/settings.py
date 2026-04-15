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
    url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))

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


@dataclass(frozen=True)
class ClusteringSettings:
    window_hours: int = field(
        default_factory=lambda: int(os.environ.get("CLUSTER_WINDOW_HOURS", "72"))
    )


@dataclass(frozen=True)
class ApiSettings:
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
        ]
    )


@dataclass(frozen=True)
class Settings:
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    api: ApiSettings = field(default_factory=ApiSettings)


def get_settings() -> Settings:
    return Settings()
