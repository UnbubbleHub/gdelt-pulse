"""Entity-aware similarity scoring for cluster assignment."""

from __future__ import annotations

import json
from typing import Any


def extract_entity_sets(article: dict[str, Any]) -> dict[str, set[str]]:
    """Extract normalized entity sets from an article dict.

    Returns a dict with keys 'locations', 'persons', 'organizations',
    each mapping to a set of lowercase entity names.
    """
    return {
        "locations": _extract_location_names(article.get("locations")),
        "persons": _extract_flat_names(article.get("persons")),
        "organizations": _extract_flat_names(article.get("organizations")),
    }


def merge_entity_sets(
    sets_list: list[dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Merge multiple entity sets into one (union across articles)."""
    merged: dict[str, set[str]] = {"locations": set(), "persons": set(), "organizations": set()}
    for s in sets_list:
        for key in merged:
            merged[key] |= s.get(key, set())
    return merged


def compute_entity_overlap(
    entities_a: dict[str, set[str]],
    entities_b: dict[str, set[str]],
) -> float:
    """Compute weighted entity overlap between two entity sets.

    Returns a score between 0.0 and 1.0.  Each entity type contributes
    its own Jaccard similarity, weighted by importance:
      - locations:     0.50  (strongest signal for same event)
      - persons:       0.35
      - organizations: 0.15
    """
    weights = {"locations": 0.50, "persons": 0.35, "organizations": 0.15}

    score = 0.0
    for key, weight in weights.items():
        a = entities_a.get(key, set())
        b = entities_b.get(key, set())
        if not a and not b:
            continue
        if not a or not b:
            # One side has entities, the other doesn't — no overlap
            continue
        intersection = a & b
        union = a | b
        score += weight * (len(intersection) / len(union))

    return score


def compute_combined_score(
    cosine_similarity: float,
    entity_overlap: float,
    *,
    entity_weight: float = 0.25,
) -> float:
    """Combine cosine similarity with entity overlap into a final score.

    final = cosine_similarity + entity_weight * entity_overlap

    Entity overlap acts as a bonus that can push borderline cosine matches
    above threshold, but cannot create a match from nothing (cosine must
    still be reasonably close).
    """
    return cosine_similarity + entity_weight * entity_overlap


def _extract_location_names(raw: Any) -> set[str]:
    data = _load_json(raw)
    return {
        loc["name"].lower()
        for loc in data
        if isinstance(loc, dict) and loc.get("name")
    }


def _extract_flat_names(raw: Any) -> set[str]:
    data = _load_json(raw)
    return {name.lower() for name in data if isinstance(name, str) and name}


def _load_json(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
