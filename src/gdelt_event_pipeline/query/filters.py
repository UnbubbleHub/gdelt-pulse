"""Build SQL WHERE clauses from SearchFilters."""

from __future__ import annotations

import json
from typing import Any

from gdelt_event_pipeline.query.models import SearchFilters


def build_filter_clauses(
    filters: SearchFilters | None,
    table_alias: str | None = None,
) -> tuple[str, list[Any]]:
    """Convert SearchFilters into a SQL fragment and parameter list.

    Returns a tuple of (sql_fragment, params) where sql_fragment contains
    AND-prefixed conditions ready to append after a WHERE clause.

    If ``table_alias`` is provided (e.g. "a"), every column reference is
    prefixed with it (e.g. "a.domain") so the fragment can be embedded in
    queries that join multiple tables.
    """
    if filters is None:
        return "", []

    p = f"{table_alias}." if table_alias else ""

    clauses: list[str] = []
    params: list[Any] = []

    if filters.locations:
        for loc in filters.locations:
            clauses.append(f"{p}locations @> %s::jsonb")
            params.append(json.dumps([{"name": loc}]))

    if filters.persons:
        clauses.append(f"{p}persons @> %s::jsonb")
        params.append(json.dumps(filters.persons))

    if filters.organizations:
        clauses.append(f"{p}organizations @> %s::jsonb")
        params.append(json.dumps(filters.organizations))

    if filters.themes:
        for theme in filters.themes:
            clauses.append(f"{p}themes @> %s::jsonb")
            params.append(json.dumps([{"theme": theme}]))

    if filters.domains:
        # Match the exact domain OR any subdomain of it (e.g. "corriere.it"
        # matches both "corriere.it" and "video.corriere.it").
        subdomain_patterns = [f"%.{d}" for d in filters.domains]
        clauses.append(f"({p}domain = ANY(%s) OR {p}domain LIKE ANY(%s))")
        params.append(filters.domains)
        params.append(subdomain_patterns)

    if filters.sources:
        clauses.append(f"{p}canonical_source = ANY(%s)")
        params.append(filters.sources)

    if filters.date_from:
        clauses.append(f"{p}gdelt_timestamp >= %s")
        params.append(filters.date_from)

    if filters.date_to:
        clauses.append(f"{p}gdelt_timestamp <= %s")
        params.append(filters.date_to)

    if not clauses:
        return "", []

    sql = " AND " + " AND ".join(clauses)
    return sql, params
