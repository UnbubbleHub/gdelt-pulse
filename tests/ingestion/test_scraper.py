"""Tests for the HTML title scraper."""

from unittest.mock import MagicMock, patch

import httpx

from gdelt_event_pipeline.ingestion.scraper import (
    _extract_title,
    _fetch_title,
    scrape_titles,
)

# ── _fetch_title extraction logic ──────────────────────────────────────


def _make_stream_response(html_bytes: bytes, charset: str | None = None) -> MagicMock:
    """Build a mock that mimics `client.stream(...)` as a context manager.

    The returned object behaves like an httpx.Response: it iterates body bytes
    via `iter_bytes()` and exposes `charset_encoding`.
    """
    resp = MagicMock()
    resp.charset_encoding = charset
    # Yield the body in a single chunk — scraper accumulates and slices to
    # MAX_READ_BYTES, so a single chunk is fine for tests.
    resp.iter_bytes.return_value = iter([html_bytes])

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _make_client(stream_cm: MagicMock | Exception) -> MagicMock:
    """Build a mock httpx.Client whose `.stream(...)` returns/raises as given."""
    client = MagicMock()
    if isinstance(stream_cm, Exception):
        client.stream.side_effect = stream_cm
    else:
        client.stream.return_value = stream_cm
    return client


class TestFetchTitleExtraction:
    """Test title extraction from HTML (mocking httpx.Client.stream)."""

    def test_basic_title(self):
        client = _make_client(
            _make_stream_response(b"<html><head><title>Hello World</title></head></html>")
        )
        assert _fetch_title(client, "http://example.com") == "Hello World"

    def test_title_with_attributes(self):
        client = _make_client(
            _make_stream_response(b'<html><title lang="en">My Page</title></html>')
        )
        assert _fetch_title(client, "http://example.com") == "My Page"

    def test_title_with_html_entities(self):
        client = _make_client(
            _make_stream_response(b"<title>Tom &amp; Jerry &lt;3&gt;</title>")
        )
        assert _fetch_title(client, "http://example.com") == "Tom & Jerry <3>"

    def test_title_with_apos_entity(self):
        client = _make_client(
            _make_stream_response(b"<title>It&#39;s &apos;fine&apos;</title>")
        )
        assert _fetch_title(client, "http://example.com") == "It's 'fine'"

    def test_title_with_whitespace_collapse(self):
        client = _make_client(
            _make_stream_response(b"<title>  Hello   \n  World  </title>")
        )
        assert _fetch_title(client, "http://example.com") == "Hello World"

    def test_no_title_tag(self):
        client = _make_client(
            _make_stream_response(b"<html><body>No title here</body></html>")
        )
        assert _fetch_title(client, "http://example.com") is None

    def test_empty_title_returns_none(self):
        client = _make_client(_make_stream_response(b"<title>   </title>"))
        assert _fetch_title(client, "http://example.com") is None

    def test_network_error_returns_none(self):
        client = _make_client(httpx.ConnectError("Connection refused"))
        assert _fetch_title(client, "http://example.com") is None

    def test_timeout_error_returns_none(self):
        client = _make_client(httpx.ReadTimeout("read timed out"))
        assert _fetch_title(client, "http://example.com") is None

    def test_multiline_title(self):
        client = _make_client(
            _make_stream_response(
                b"<title>\n  Breaking News:\n  Something happened\n</title>"
            )
        )
        assert (
            _fetch_title(client, "http://example.com")
            == "Breaking News: Something happened"
        )

    def test_case_insensitive_tag(self):
        client = _make_client(_make_stream_response(b"<TITLE>Upper Case</TITLE>"))
        assert _fetch_title(client, "http://example.com") == "Upper Case"

    def test_latin1_fallback(self):
        # \xe9 is 'é' in latin-1 but invalid standalone utf-8
        client = _make_client(_make_stream_response(b"<title>Caf\xe9 News</title>"))
        assert _fetch_title(client, "http://example.com") == "Café News"

    def test_og_title_fallback(self):
        client = _make_client(
            _make_stream_response(
                b'<html><meta property="og:title" content="OG Title Here"></html>'
            )
        )
        assert _fetch_title(client, "http://example.com") == "OG Title Here"

    def test_title_tag_preferred_over_og(self):
        client = _make_client(
            _make_stream_response(
                b'<html><title>Real Title</title>'
                b'<meta property="og:title" content="OG Title"></html>'
            )
        )
        assert _fetch_title(client, "http://example.com") == "Real Title"

    def test_numeric_html_entity(self):
        client = _make_client(
            _make_stream_response(b"<title>Score: 10 &#8211; 5</title>")
        )
        assert _fetch_title(client, "http://example.com") == "Score: 10 – 5"

    def test_body_truncated_at_max_read_bytes(self):
        # Hand the scraper many small chunks totalling more than 64KB; the
        # title appears in the first chunk, so the scraper must still find it
        # without exhausting the stream.
        first = b"<title>Trunc Test</title>" + b" " * 4000
        chunks = [first] + [b"X" * 4096 for _ in range(40)]  # ~160 KB total

        resp = MagicMock()
        resp.charset_encoding = None
        resp.iter_bytes.return_value = iter(chunks)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=resp)
        cm.__exit__ = MagicMock(return_value=False)
        client = _make_client(cm)

        assert _fetch_title(client, "http://example.com") == "Trunc Test"


class TestExtractTitle:
    def test_basic(self):
        assert _extract_title("<title>Hello</title>") == "Hello"

    def test_html_unescape(self):
        assert _extract_title("<title>&mdash; Dash</title>") == "— Dash"

    def test_og_fallback(self):
        html = '<meta property="og:title" content="Fallback Title">'
        assert _extract_title(html) == "Fallback Title"

    def test_no_title(self):
        assert _extract_title("<html><body>nothing</body></html>") is None


# ── scrape_titles batch logic ──────────────────────────────────────────


class TestScrapeTitles:
    @patch("gdelt_event_pipeline.ingestion.scraper._fetch_title")
    def test_returns_only_successful(self, mock_fetch):
        mock_fetch.side_effect = lambda client, url: {
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
