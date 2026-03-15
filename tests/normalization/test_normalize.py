"""Tests for the GKG row normalizer."""

import json
from datetime import datetime, timezone

from gdelt_event_pipeline.normalization.normalize import normalize_row, parse_gkg_timestamp


def _make_row(**overrides) -> list[str]:
    """Build a minimal 27-column GKG row with sensible defaults."""
    defaults = [""] * 27
    defaults[0] = overrides.get("gkg_record_id", "20240101120000-1")
    defaults[1] = overrides.get("date", "20240101120000")
    defaults[2] = overrides.get("source_collection", "1")
    defaults[3] = overrides.get("source_common_name", "bbc.co.uk")
    defaults[4] = overrides.get("url", "https://www.bbc.co.uk/news/world-12345")
    defaults[8] = overrides.get("v2_themes", "ARMEDCONFLICT,100")
    defaults[10] = overrides.get("v2_locations", "")
    defaults[12] = overrides.get("v2_persons", "Joe Biden,50")
    defaults[14] = overrides.get("v2_organizations", "NATO,100")
    defaults[15] = overrides.get("v2_tone", "-1.19,2.08,3.27,5.35,17.26")
    defaults[23] = overrides.get("all_names", "Joe Biden,50;NATO,100")
    return defaults


class TestParseGkgTimestamp:
    def test_valid_timestamp(self):
        result = parse_gkg_timestamp("20240101120000")
        assert result == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_empty_string(self):
        assert parse_gkg_timestamp("") is None

    def test_invalid_format(self):
        assert parse_gkg_timestamp("not-a-timestamp") is None


class TestNormalizeRow:
    def test_basic_row(self):
        row = _make_row()
        result = normalize_row(row)
        assert result is not None
        assert result["gkg_record_id"] == "20240101120000-1"
        assert result["gdelt_timestamp"] == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert result["canonical_url"] == "https://bbc.co.uk/news/world-12345"
        assert result["domain"] == "bbc.co.uk"
        assert result["canonical_source"] == "bbc"

    def test_themes_parsed(self):
        row = _make_row(v2_themes="WAR,10;PEACE,20")
        result = normalize_row(row)
        themes = json.loads(result["themes"])
        assert len(themes) == 2
        assert themes[0]["theme"] == "WAR"

    def test_tone_parsed(self):
        row = _make_row()
        result = normalize_row(row)
        tone = json.loads(result["tone"])
        assert tone["tone"] == -1.19

    def test_persons_parsed(self):
        row = _make_row(v2_persons="Alice,10;Bob,20")
        result = normalize_row(row)
        persons = json.loads(result["persons"])
        assert persons == ["Alice", "Bob"]

    def test_organizations_parsed(self):
        row = _make_row()
        result = normalize_row(row)
        orgs = json.loads(result["organizations"])
        assert orgs == ["NATO"]

    def test_raw_payload_included(self):
        row = _make_row()
        result = normalize_row(row)
        raw = json.loads(result["raw_payload"])
        assert "col_0" in raw

    def test_row_too_short_returns_none(self):
        assert normalize_row(["only", "three", "cols"]) is None

    def test_missing_url_returns_none(self):
        row = _make_row(url="")
        assert normalize_row(row) is None

    def test_missing_timestamp_returns_none(self):
        row = _make_row(date="")
        assert normalize_row(row) is None

    def test_missing_record_id_returns_none(self):
        row = _make_row(gkg_record_id="")
        assert normalize_row(row) is None

    def test_url_canonicalization_applied(self):
        row = _make_row(url="https://www.example.com/path/?utm_source=twitter&id=1")
        result = normalize_row(row)
        assert result["canonical_url"] == "https://example.com/path?id=1"

    def test_title_is_none(self):
        row = _make_row()
        result = normalize_row(row)
        assert result["title"] is None

    def test_empty_optional_fields_are_none(self):
        row = _make_row(v2_themes="", v2_persons="", v2_organizations="", v2_tone="", all_names="")
        result = normalize_row(row)
        assert result["themes"] is None
        assert result["persons"] is None
        assert result["organizations"] is None
        assert result["tone"] is None
        assert result["all_names"] is None
