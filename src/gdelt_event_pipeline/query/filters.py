"""Build SQL WHERE clauses from SearchFilters."""

from __future__ import annotations

import json
from typing import Any

from gdelt_event_pipeline.query.models import SearchFilters


def build_filter_clauses(
    filters: SearchFilters | None,
) -> tuple[str, list[Any]]:
    """Convert SearchFilters into a SQL fragment and parameter list.

    Returns a tuple of (sql_fragment, params) where sql_fragment contains
    AND-prefixed conditions ready to append after a WHERE clause.
    """
    if filters is None:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    if filters.locations:
        for loc in filters.locations:
            clauses.append("locations @> %s::jsonb")
            params.append(json.dumps([{"name": loc}]))

    if filters.persons:
        clauses.append("persons @> %s::jsonb")
        params.append(json.dumps(filters.persons))

    if filters.organizations:
        clauses.append("organizations @> %s::jsonb")
        params.append(json.dumps(filters.organizations))

    if filters.themes:
        for theme in filters.themes:
            clauses.append("themes @> %s::jsonb")
            params.append(json.dumps([{"theme": theme}]))

    if filters.domains:
        clauses.append("domain = ANY(%s)")
        params.append(filters.domains)

    if filters.sources:
        clauses.append("canonical_source = ANY(%s)")
        params.append(filters.sources)

    if filters.date_from:
        clauses.append("gdelt_timestamp >= %s")
        params.append(filters.date_from)

    if filters.date_to:
        clauses.append("gdelt_timestamp <= %s")
        params.append(filters.date_to)

    if not clauses:
        return "", []

    sql = " AND " + " AND ".join(clauses)
    return sql, params
