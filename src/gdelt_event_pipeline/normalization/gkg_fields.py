"""Parsers for GKG semi-structured fields into typed JSONB-ready structures."""

from __future__ import annotations

from typing import Any


def parse_v2_themes(raw: str) -> list[dict[str, Any]]:
    """Parse V2Themes field.

    Format: THEME,OFFSET;THEME,OFFSET;...
    Returns: [{"theme": "...", "offset": int}, ...]
    """
    if not raw or not raw.strip():
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        theme = parts[0].strip()
        if not theme or theme in seen:
            continue
        seen.add(theme)

        item: dict[str, Any] = {"theme": theme}
        if len(parts) > 1:
            try:
                item["offset"] = int(parts[1])
            except ValueError:
                pass
        results.append(item)
    return results


def parse_v2_tone(raw: str) -> dict[str, float] | None:
    """Parse V2Tone field.

    Format: tone,pos,neg,polarity,activity_ref_density,self_group_ref_density,word_count
    Returns dict with named fields or None if unparseable.
    """
    if not raw or not raw.strip():
        return None

    parts = raw.split(",")
    if len(parts) < 5:
        return None

    try:
        return {
            "tone": float(parts[0]),
            "positive_score": float(parts[1]),
            "negative_score": float(parts[2]),
            "polarity": float(parts[3]),
            "activity_ref_density": float(parts[4]),
        }
    except (ValueError, IndexError):
        return None


def parse_v2_locations(raw: str) -> list[dict[str, Any]]:
    """Parse V2Locations field.

    Format: type#name#countrycode#adm1#adm2#lat#long#featureid;...
    Returns list of structured location dicts.
    """
    if not raw or not raw.strip():
        return []

    results: list[dict[str, Any]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("#")
        if len(parts) < 7:
            continue

        location: dict[str, Any] = {
            "type": _int_or_none(parts[0]),
            "name": parts[1] or None,
            "country_code": parts[2] or None,
            "adm1": parts[3] or None,
            "adm2": parts[4] if len(parts) > 4 else None,
        }

        lat = _float_or_none(parts[5])
        lon = _float_or_none(parts[6])
        if lat is not None and lon is not None:
            location["lat"] = lat
            location["lon"] = lon

        if len(parts) > 7 and parts[7]:
            location["feature_id"] = parts[7]

        results.append(location)
    return results


def parse_v2_persons(raw: str) -> list[str]:
    """Parse V2Persons field.

    Format: name,offset;name,offset;...
    Returns deduplicated list of person names.
    """
    return _parse_name_offset_field(raw)


def parse_v2_organizations(raw: str) -> list[str]:
    """Parse V2Organizations field.

    Format: name,offset;name,offset;...
    Returns deduplicated list of organization names.
    """
    return _parse_name_offset_field(raw)


def parse_all_names(raw: str) -> list[str]:
    """Parse AllNames field.

    Format: name,offset;name,offset;...
    Returns deduplicated list of names.
    """
    return _parse_name_offset_field(raw)


def _parse_name_offset_field(raw: str) -> list[str]:
    """Shared parser for semicolon-delimited name,offset fields."""
    if not raw or not raw.strip():
        return []

    seen: set[str] = set()
    results: list[str] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name = entry.split(",")[0].strip()
        if name and name not in seen:
            seen.add(name)
            results.append(name)
    return results


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
