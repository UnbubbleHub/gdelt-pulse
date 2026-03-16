"""Tests for the GKG fetcher module."""

import csv
import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from gdelt_event_pipeline.ingestion.gkg_fetcher import (
    GkgFile,
    download_and_parse_gkg,
    fetch_latest_file_list,
    get_latest_gkg_url,
)


def _mock_urlopen(data: bytes):
    """Return a context-manager mock that yields data on read()."""
    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_gkg_zip(rows: list[list[str]], csv_name: str = "test.gkg.csv") -> bytes:
    """Build an in-memory zip containing a tab-separated CSV."""
    buf = io.BytesIO()
    text = io.StringIO()
    writer = csv.writer(text, delimiter="\t")
    for row in rows:
        writer.writerow(row)

    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(csv_name, text.getvalue())

    return buf.getvalue()


# ── fetch_latest_file_list ─────────────────────────────────────────────


class TestFetchLatestFileList:
    SAMPLE_LASTUPDATE = (
        "123456 abc123hash http://data.gdeltproject.org/gdeltv2/20260315120000.export.CSV.zip\n"
        "654321 def456hash http://data.gdeltproject.org/gdeltv2/20260315120000.gkg.csv.zip\n"
        "111111 ghi789hash http://data.gdeltproject.org/gdeltv2/20260315120000.mentions.CSV.zip\n"
    )

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_parses_gkg_entry(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(self.SAMPLE_LASTUPDATE.encode())
        result = fetch_latest_file_list()

        assert len(result) == 1
        assert result[0] == GkgFile(
            size=654321,
            md5="def456hash",
            url="http://data.gdeltproject.org/gdeltv2/20260315120000.gkg.csv.zip",
        )

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_no_gkg_entry(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(
            b"123 abc http://example.com/export.CSV.zip\n"
        )
        result = fetch_latest_file_list()
        assert result == []

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_empty_response(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(b"")
        result = fetch_latest_file_list()
        assert result == []

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_malformed_lines_skipped(self, mock_urlopen):
        data = (
            "short\n"
            "654321 def456hash http://data.gdeltproject.org/20260315.gkg.csv.zip\n"
        )
        mock_urlopen.return_value = _mock_urlopen(data.encode())
        result = fetch_latest_file_list()
        assert len(result) == 1


# ── get_latest_gkg_url ────────────────────────────────────────────────


class TestGetLatestGkgUrl:
    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.fetch_latest_file_list")
    def test_returns_first_url(self, mock_fetch):
        mock_fetch.return_value = [
            GkgFile(size=100, md5="abc", url="http://example.com/a.gkg.csv.zip"),
            GkgFile(size=200, md5="def", url="http://example.com/b.gkg.csv.zip"),
        ]
        assert get_latest_gkg_url() == "http://example.com/a.gkg.csv.zip"

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.fetch_latest_file_list")
    def test_raises_when_empty(self, mock_fetch):
        mock_fetch.return_value = []
        with pytest.raises(RuntimeError, match="No GKG file"):
            get_latest_gkg_url()


# ── download_and_parse_gkg ────────────────────────────────────────────


class TestDownloadAndParseGkg:
    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_parses_rows_from_zip(self, mock_urlopen):
        rows = [
            ["20260315120000-1", "20260315120000", "", "example.com", "http://example.com/a"],
            ["20260315120000-2", "20260315120000", "", "test.com", "http://test.com/b"],
        ]
        zip_bytes = _make_gkg_zip(rows)
        mock_urlopen.return_value = _mock_urlopen(zip_bytes)

        result = download_and_parse_gkg("http://example.com/test.gkg.csv.zip")
        assert len(result) == 2
        assert result[0][0] == "20260315120000-1"
        assert result[1][3] == "test.com"

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_no_csv_in_zip_raises(self, mock_urlopen):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "not a csv")
        mock_urlopen.return_value = _mock_urlopen(buf.getvalue())

        with pytest.raises(RuntimeError, match="No CSV file"):
            download_and_parse_gkg("http://example.com/bad.zip")

    @patch("gdelt_event_pipeline.ingestion.gkg_fetcher.urlopen")
    def test_empty_csv(self, mock_urlopen):
        zip_bytes = _make_gkg_zip([])
        mock_urlopen.return_value = _mock_urlopen(zip_bytes)

        result = download_and_parse_gkg("http://example.com/empty.gkg.csv.zip")
        assert result == []
