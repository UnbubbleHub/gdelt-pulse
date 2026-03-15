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
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class Settings:
    db: DatabaseSettings = field(default_factory=DatabaseSettings)


def get_settings() -> Settings:
    return Settings()
