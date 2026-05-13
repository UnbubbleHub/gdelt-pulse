"""Tests for SQL filter builder."""

import json

from gdelt_event_pipeline.query.filters import build_filter_clauses
from gdelt_event_pipeline.query.models import SearchFilters


class TestBuildFilterClauses:
    def test_none_filters(self):
        sql, params = build_filter_clauses(None)
        assert sql == ""
        assert params == []

    def test_empty_filters(self):
        sql, params = build_filter_clauses(SearchFilters())
        assert sql == ""
        assert params == []

    def test_location_filter(self):
        sql, params = build_filter_clauses(SearchFilters(locations=["Turkey"]))
        assert "locations @>" in sql
        assert json.loads(params[0]) == [{"name": "Turkey"}]

    def test_multiple_locations(self):
        sql, params = build_filter_clauses(SearchFilters(locations=["Turkey", "Syria"]))
        assert sql.count("locations @>") == 2
        assert len(params) == 2

    def test_persons_filter(self):
        sql, params = build_filter_clauses(SearchFilters(persons=["John Smith"]))
        assert "persons @>" in sql
        assert json.loads(params[0]) == ["John Smith"]

    def test_organizations_filter(self):
        sql, params = build_filter_clauses(SearchFilters(organizations=["NATO"]))
        assert "organizations @>" in sql
        assert json.loads(params[0]) == ["NATO"]

    def test_themes_filter(self):
        sql, params = build_filter_clauses(SearchFilters(themes=["ECON_BANKRUPTCY"]))
        assert "themes @>" in sql
        assert json.loads(params[0]) == [{"theme": "ECON_BANKRUPTCY"}]

    def test_domain_filter(self):
        sql, params = build_filter_clauses(SearchFilters(domains=["bbc.co.uk"]))
        assert "domain = ANY" in sql
        assert "domain LIKE ANY" in sql
        assert params[0] == ["bbc.co.uk"]
        assert params[1] == ["%.bbc.co.uk"]

    def test_domain_filter_matches_subdomains(self):
        # The LIKE pattern is what enables soft subdomain matching:
        # passing "corriere.it" should also match "video.corriere.it".
        _, params = build_filter_clauses(
            SearchFilters(domains=["corriere.it", "repubblica.it"])
        )
        assert params[0] == ["corriere.it", "repubblica.it"]
        assert params[1] == ["%.corriere.it", "%.repubblica.it"]

    def test_source_filter(self):
        sql, params = build_filter_clauses(SearchFilters(sources=["reuters"]))
        assert "canonical_source = ANY" in sql
        assert params[0] == ["reuters"]

    def test_date_from_filter(self):
        from datetime import UTC, datetime

        dt = datetime(2026, 1, 1, tzinfo=UTC)
        sql, params = build_filter_clauses(SearchFilters(date_from=dt))
        assert "gdelt_timestamp >=" in sql
        assert params[0] == dt

    def test_date_to_filter(self):
        from datetime import UTC, datetime

        dt = datetime(2026, 3, 1, tzinfo=UTC)
        sql, params = build_filter_clauses(SearchFilters(date_to=dt))
        assert "gdelt_timestamp <=" in sql
        assert params[0] == dt

    def test_multiple_filters_combine(self):
        filters = SearchFilters(
            locations=["Turkey"],
            domains=["bbc.co.uk"],
        )
        sql, params = build_filter_clauses(filters)
        assert "locations @>" in sql
        assert "domain = ANY" in sql
        assert " AND " in sql
        # 1 location param + 2 domain params (exact list + LIKE patterns)
        assert len(params) == 3

    def test_sql_starts_with_and(self):
        sql, _ = build_filter_clauses(SearchFilters(domains=["bbc.co.uk"]))
        assert sql.strip().startswith("AND")
