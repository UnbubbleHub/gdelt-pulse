"""Tests for the HTML title scraper."""

from unittest.mock import MagicMock, patch

from gdelt_event_pipeline.ingestion.scraper import (
    _extract_title,
    _fetch_title,
    scrape_titles,
)


# ── _fetch_title extraction logic ──────────────────────────────────────


class TestFetchTitleExtraction:
    """Test title extraction from HTML (mocking urlopen)."""

    def _mock_urlopen(self, html_bytes, charset=None):
        """Return a context-manager mock that yields html_bytes on read()."""
        resp = MagicMock()
        resp.read.return_value = html_bytes
        resp.headers.get_content_charset.return_value = charset
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_basic_title(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<html><head><title>Hello World</title></head></html>"
        )
        assert _fetch_title("http://example.com") == "Hello World"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_title_with_attributes(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b'<html><title lang="en">My Page</title></html>'
        )
        assert _fetch_title("http://example.com") == "My Page"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_title_with_html_entities(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>Tom &amp; Jerry &lt;3&gt;</title>"
        )
        assert _fetch_title("http://example.com") == "Tom & Jerry <3>"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_title_with_apos_entity(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>It&#39;s &apos;fine&apos;</title>"
        )
        assert _fetch_title("http://example.com") == "It's 'fine'"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_title_with_whitespace_collapse(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>  Hello   \n  World  </title>"
        )
        assert _fetch_title("http://example.com") == "Hello World"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_no_title_tag(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<html><body>No title here</body></html>"
        )
        assert _fetch_title("http://example.com") is None

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_empty_title_returns_none(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>   </title>"
        )
        assert _fetch_title("http://example.com") is None

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_network_error_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        assert _fetch_title("http://example.com") is None

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_multiline_title(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>\n  Breaking News:\n  Something happened\n</title>"
        )
        assert _fetch_title("http://example.com") == "Breaking News: Something happened"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_case_insensitive_tag(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<TITLE>Upper Case</TITLE>"
        )
        assert _fetch_title("http://example.com") == "Upper Case"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_latin1_fallback(self, mock_urlopen):
        # \xe9 is 'é' in latin-1 but invalid standalone utf-8
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>Caf\xe9 News</title>"
        )
        assert _fetch_title("http://example.com") == "Café News"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_og_title_fallback(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b'<html><meta property="og:title" content="OG Title Here"></html>'
        )
        assert _fetch_title("http://example.com") == "OG Title Here"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_title_tag_preferred_over_og(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b'<html><title>Real Title</title>'
            b'<meta property="og:title" content="OG Title"></html>'
        )
        assert _fetch_title("http://example.com") == "Real Title"

    @patch("gdelt_event_pipeline.ingestion.scraper.urlopen")
    def test_numeric_html_entity(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            b"<title>Score: 10 &#8211; 5</title>"
        )
        assert _fetch_title("http://example.com") == "Score: 10 \u2013 5"


class TestExtractTitle:
    def test_basic(self):
        assert _extract_title("<title>Hello</title>") == "Hello"

    def test_html_unescape(self):
        assert _extract_title("<title>&mdash; Dash</title>") == "\u2014 Dash"

    def test_og_fallback(self):
        html = '<meta property="og:title" content="Fallback Title">'
        assert _extract_title(html) == "Fallback Title"

    def test_no_title(self):
        assert _extract_title("<html><body>nothing</body></html>") is None


# ── scrape_titles batch logic ──────────────────────────────────────────


class TestScrapeTitles:
    @patch("gdelt_event_pipeline.ingestion.scraper._fetch_title")
    def test_returns_only_successful(self, mock_fetch):
        mock_fetch.side_effect = lambda url, timeout: {
            "http://a.com": "Title A",
            "http://b.com": None,
            "http://c.com": "Title C",
        }.get(url)

        articles = [
            {"id": "id-1", "url": "http://a.com"},
            {"id": "id-2", "url": "http://b.com"},
            {"id": "id-3", "url": "http://c.com"},
        ]
        result = scrape_titles(articles, max_workers=1)

        assert result == {"id-1": "Title A", "id-3": "Title C"}

    @patch("gdelt_event_pipeline.ingestion.scraper._fetch_title")
    def test_empty_list(self, mock_fetch):
        result = scrape_titles([], max_workers=1)
        assert result == {}
        mock_fetch.assert_not_called()

    @patch("gdelt_event_pipeline.ingestion.scraper._fetch_title")
    def test_all_fail(self, mock_fetch):
        mock_fetch.return_value = None
        articles = [
            {"id": "id-1", "url": "http://a.com"},
            {"id": "id-2", "url": "http://b.com"},
        ]
        result = scrape_titles(articles, max_workers=1)
        assert result == {}
