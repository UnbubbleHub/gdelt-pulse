"""Tests for GKG field parsers."""

from gdelt_event_pipeline.normalization.gkg_fields import (
    filter_persons_against_locations,
    parse_all_names,
    parse_v2_locations,
    parse_v2_organizations,
    parse_v2_persons,
    parse_v2_themes,
    parse_v2_tone,
)


class TestParseV2Themes:
    def test_basic_themes(self):
        raw = "ARMEDCONFLICT,100;EPU_CATS_NATIONAL_SECURITY,200"
        result = parse_v2_themes(raw)
        assert len(result) == 2
        assert result[0] == {"theme": "ARMEDCONFLICT", "offset": 100}
        assert result[1] == {"theme": "EPU_CATS_NATIONAL_SECURITY", "offset": 200}

    def test_deduplicates_themes(self):
        raw = "ARMEDCONFLICT,100;ARMEDCONFLICT,200;TAX,300"
        result = parse_v2_themes(raw)
        assert len(result) == 2
        themes = [r["theme"] for r in result]
        assert themes == ["ARMEDCONFLICT", "TAX"]

    def test_empty_string(self):
        assert parse_v2_themes("") == []

    def test_none(self):
        assert parse_v2_themes(None) == []

    def test_theme_without_offset(self):
        raw = "ARMEDCONFLICT"
        result = parse_v2_themes(raw)
        assert len(result) == 1
        assert result[0] == {"theme": "ARMEDCONFLICT"}


class TestParseV2Tone:
    def test_basic_tone(self):
        raw = "-1.19,2.08,3.27,5.35,17.26"
        result = parse_v2_tone(raw)
        assert result is not None
        assert result["tone"] == -1.19
        assert result["positive_score"] == 2.08
        assert result["negative_score"] == 3.27
        assert result["polarity"] == 5.35
        assert result["activity_ref_density"] == 17.26

    def test_empty_string(self):
        assert parse_v2_tone("") is None

    def test_too_few_values(self):
        assert parse_v2_tone("1.0,2.0") is None

    def test_non_numeric(self):
        assert parse_v2_tone("a,b,c,d,e") is None


class TestParseV2Locations:
    def test_basic_location(self):
        raw = "4#Washington#US#USDC#DC#38.8951#-77.0364#NI38895100"
        result = parse_v2_locations(raw)
        assert len(result) == 1
        loc = result[0]
        assert loc["type"] == 4
        assert loc["name"] == "Washington"
        assert loc["country_code"] == "US"
        assert loc["lat"] == 38.8951
        assert loc["lon"] == -77.0364

    def test_multiple_locations(self):
        raw = "1#France#FR####48.8566#2.3522#;1#Germany#DE####52.52#13.405#"
        result = parse_v2_locations(raw)
        assert len(result) == 2

    def test_empty_string(self):
        assert parse_v2_locations("") == []

    def test_too_short_entry_skipped(self):
        raw = "1#France#FR"
        assert parse_v2_locations(raw) == []

    def test_deduplicates_same_location(self):
        raw = (
            "1#Iran#IR#IR##32.0#53.0#IR;"
            "1#Iran#IR#IR##32.0#53.0#IR;"
            "1#United States#US#US##39.828175#-98.5795#US"
        )
        result = parse_v2_locations(raw)
        assert len(result) == 2
        assert result[0]["name"] == "Iran"
        assert result[1]["name"] == "United States"

    def test_deduplicates_same_name_different_coords(self):
        raw = (
            "4#Kalava#IN#IN36##15.5#78.5#;"
            "4#Kalava#IN#IN36##15.500001#78.500001#"
        )
        result = parse_v2_locations(raw)
        assert len(result) == 1
        assert result[0]["name"] == "Kalava"


class TestParseV2Persons:
    def test_basic_persons(self):
        raw = "Joe Biden,100;Volodymyr Zelenskyy,200"
        result = parse_v2_persons(raw)
        assert result == ["Joe Biden", "Volodymyr Zelenskyy"]

    def test_deduplicates(self):
        raw = "Joe Biden,100;Joe Biden,200;Jane Doe,300"
        result = parse_v2_persons(raw)
        assert result == ["Joe Biden", "Jane Doe"]

    def test_empty(self):
        assert parse_v2_persons("") == []


class TestFilterPersonsAgainstLocations:
    def test_removes_location_names_from_persons(self):
        persons = ["Priyanka Chopra", "Los Angeles", "Nick Jonas"]
        locations = [
            {"name": "Los Angeles", "country_code": "US"},
            {"name": "Mumbai", "country_code": "IN"},
        ]
        result = filter_persons_against_locations(persons, locations)
        assert result == ["Priyanka Chopra", "Nick Jonas"]

    def test_case_insensitive_match(self):
        persons = ["los angeles", "Joe Biden"]
        locations = [{"name": "Los Angeles", "country_code": "US"}]
        result = filter_persons_against_locations(persons, locations)
        assert result == ["Joe Biden"]

    def test_no_locations(self):
        persons = ["Joe Biden"]
        result = filter_persons_against_locations(persons, [])
        assert result == ["Joe Biden"]

    def test_no_overlap(self):
        persons = ["Joe Biden"]
        locations = [{"name": "Washington", "country_code": "US"}]
        result = filter_persons_against_locations(persons, locations)
        assert result == ["Joe Biden"]


class TestParseV2Organizations:
    def test_basic_orgs(self):
        raw = "United Nations,50;NATO,150"
        result = parse_v2_organizations(raw)
        assert result == ["United Nations", "NATO"]


class TestParseAllNames:
    def test_basic_names(self):
        raw = "Joe Biden,100;White House,200"
        result = parse_all_names(raw)
        assert result == ["Joe Biden", "White House"]

    def test_empty(self):
        assert parse_all_names("") == []
