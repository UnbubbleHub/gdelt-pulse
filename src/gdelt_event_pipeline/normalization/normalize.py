"""Normalize a raw GKG row into an article dict ready for storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from gdelt_event_pipeline.normalization.gkg_fields import (
    parse_all_names,
    parse_v2_locations,
    parse_v2_organizations,
    parse_v2_persons,
    parse_v2_themes,
    parse_v2_tone,
)
from gdelt_event_pipeline.normalization.source import normalize_source
from gdelt_event_pipeline.normalization.url import canonicalize_url, extract_domain


# GKG v2 column indices (0-based)
COL_GKGRECORDID = 0
COL_DATE = 1
COL_SOURCE_COLLECTION_ID = 2
COL_SOURCE_COMMON_NAME = 3
COL_DOCUMENT_IDENTIFIER = 4
COL_V2_THEMES = 8
COL_V2_LOCATIONS = 10
COL_V2_PERSONS = 12
COL_V2_ORGANIZATIONS = 14
COL_V2_TONE = 15
COL_ALL_NAMES = 23

EXPECTED_MIN_COLUMNS = 16


def parse_gkg_timestamp(raw: str) -> datetime | None:
    """Parse a GKG DATE field (YYYYMMDDHHmmSS) into a timezone-aware datetime."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def normalize_row(row: list[str]) -> dict[str, Any] | None:
    """Convert a raw GKG tab-separated row into an article dict.

    Returns None if the row is too short, has no URL, or has no parseable timestamp.
    """
    if len(row) < EXPECTED_MIN_COLUMNS:
        return None

    gkg_record_id = _col(row, COL_GKGRECORDID)
    if not gkg_record_id:
        return None

    gdelt_timestamp = parse_gkg_timestamp(_col(row, COL_DATE))
    if gdelt_timestamp is None:
        return None

    url = _col(row, COL_DOCUMENT_IDENTIFIER)
    if not url:
        return None

    canonical_url = canonicalize_url(url)
    if not canonical_url:
        return None

    domain = extract_domain(canonical_url)
    source_common_name = _col(row, COL_SOURCE_COMMON_NAME) or None
    canonical_source = normalize_source(source_common_name, domain)

    themes = parse_v2_themes(_col(row, COL_V2_THEMES))
    locations = parse_v2_locations(_col(row, COL_V2_LOCATIONS))
    persons = parse_v2_persons(_col(row, COL_V2_PERSONS))
    organizations = parse_v2_organizations(_col(row, COL_V2_ORGANIZATIONS))
    tone = parse_v2_tone(_col(row, COL_V2_TONE))
    all_names = parse_all_names(_col(row, COL_ALL_NAMES))

    # Build raw_payload from all columns for debugging
    raw_payload = {f"col_{i}": val for i, val in enumerate(row)}

    return {
        "gkg_record_id": gkg_record_id,
        "gdelt_timestamp": gdelt_timestamp,
        "url": url,
        "canonical_url": canonical_url,
        "domain": domain,
        "source_common_name": source_common_name,
        "canonical_source": canonical_source,
        "title": None,  # GKG does not provide a title; filled later
        "themes": json.dumps(themes) if themes else None,
        "locations": json.dumps(locations) if locations else None,
        "organizations": json.dumps(organizations) if organizations else None,
        "persons": json.dumps(persons) if persons else None,
        "all_names": json.dumps(all_names) if all_names else None,
        "tone": json.dumps(tone) if tone else None,
        "raw_payload": json.dumps(raw_payload),
    }


def _col(row: list[str], index: int) -> str:
    """Safely get a column value, returning empty string if out of range."""
    if index < len(row):
        return row[index].strip()
    return ""
