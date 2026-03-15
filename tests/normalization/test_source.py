"""Tests for source name normalization."""

from gdelt_event_pipeline.normalization.source import normalize_source


class TestNormalizeSource:
    def test_known_domain_alias(self):
        assert normalize_source(None, "nytimes.com") == "new_york_times"

    def test_source_name_with_domain(self):
        assert normalize_source("bbc.co.uk", None) == "bbc"

    def test_domain_takes_precedence(self):
        assert normalize_source("some random name", "reuters.com") == "reuters"

    def test_unknown_domain_slugifies(self):
        result = normalize_source(None, "somerandomsite.org")
        assert result == "somerandomsite_org"

    def test_unknown_source_name_slugifies(self):
        result = normalize_source("My Local News", None)
        assert result == "my_local_news"

    def test_both_none_returns_none(self):
        assert normalize_source(None, None) is None

    def test_both_empty_returns_none(self):
        assert normalize_source("", "") is None

    def test_www_prefix_stripped_from_source_name(self):
        assert normalize_source("www.bbc.com", None) == "bbc"
