"""Tests for URL canonicalization."""

from gdelt_event_pipeline.normalization.url import canonicalize_url, extract_domain


class TestCanonicalizeUrl:
    def test_strips_tracking_params(self):
        url = "https://example.com/article?id=1&utm_source=twitter&utm_medium=social"
        assert canonicalize_url(url) == "https://example.com/article?id=1"

    def test_removes_www(self):
        url = "https://www.example.com/path"
        assert canonicalize_url(url) == "https://example.com/path"

    def test_lowercases_scheme_and_host(self):
        url = "HTTPS://WWW.EXAMPLE.COM/Path"
        assert canonicalize_url(url) == "https://example.com/Path"

    def test_strips_trailing_slash(self):
        url = "https://example.com/article/"
        assert canonicalize_url(url) == "https://example.com/article"

    def test_keeps_root_slash(self):
        url = "https://example.com/"
        assert canonicalize_url(url) == "https://example.com/"

    def test_removes_fragment(self):
        url = "https://example.com/article#section"
        assert canonicalize_url(url) == "https://example.com/article"

    def test_removes_default_https_port(self):
        url = "https://example.com:443/path"
        assert canonicalize_url(url) == "https://example.com/path"

    def test_removes_default_http_port(self):
        url = "http://example.com:80/path"
        assert canonicalize_url(url) == "http://example.com/path"

    def test_keeps_non_default_port(self):
        url = "https://example.com:8080/path"
        assert canonicalize_url(url) == "https://example.com:8080/path"

    def test_empty_string_returns_empty(self):
        assert canonicalize_url("") == ""

    def test_preserves_meaningful_query_params(self):
        url = "https://example.com/article?id=123&page=2"
        assert canonicalize_url(url) == "https://example.com/article?id=123&page=2"

    def test_strips_fbclid(self):
        url = "https://example.com/article?fbclid=abc123"
        assert canonicalize_url(url) == "https://example.com/article"

    def test_real_world_gdelt_url(self):
        url = "https://www.reuters.com/world/middle-east/story-2024/?utm_source=reddit"
        assert canonicalize_url(url) == "https://reuters.com/world/middle-east/story-2024"


class TestExtractDomain:
    def test_simple_domain(self):
        assert extract_domain("https://example.com/path") == "example.com"

    def test_strips_www(self):
        assert extract_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"

    def test_strips_port(self):
        assert extract_domain("https://example.com:8080/path") == "example.com"

    def test_empty_url(self):
        assert extract_domain("") == ""
