"""Compose embedding input text from article metadata."""

from __future__ import annotations

import json
from typing import Any


def compose_embedding_text(article: dict[str, Any]) -> str:
    """Build a single text string from an article dict for embedding.

    Prioritises the title, then appends structured metadata to give the
    model extra semantic signal for clustering.
    """
    parts: list[str] = []

    title = article.get("title")
    if title:
        parts.append(title)

    themes = _load_json_list(article.get("themes"))
    if themes:
        names = [t["theme"] for t in themes if isinstance(t, dict) and t.get("theme")]
        if names:
            parts.append("Themes: " + ", ".join(names))

    locations = _load_json_list(article.get("locations"))
    if locations:
        names = [loc["name"] for loc in locations if isinstance(loc, dict) and loc.get("name")]
        if names:
            parts.append("Locations: " + ", ".join(names))

    persons = _load_json_list(article.get("persons"))
    if persons:
        # persons is a flat list of strings
        parts.append("Persons: " + ", ".join(persons))

    organizations = _load_json_list(article.get("organizations"))
    if organizations:
        parts.append("Organizations: " + ", ".join(organizations))

    return ". ".join(parts)


def _load_json_list(value: Any) -> list:
    """Normalise a value that may be a JSON string or already a list."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(value, list):
        return value
    return []
