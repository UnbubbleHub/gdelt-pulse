"""Shared helpers used across API routers."""

from __future__ import annotations

from typing import Any


def strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove large/internal fields before sending to the client."""
    row.pop("embedding", None)
    row.pop("centroid_embedding", None)
    row.pop("title_tsv", None)
    row.pop("raw_payload", None)
    return row


def split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated query param into a list, or None."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]
