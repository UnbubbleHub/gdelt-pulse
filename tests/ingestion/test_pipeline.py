"""Tests for the ingestion pipeline orchestration."""

from unittest.mock import patch

from gdelt_event_pipeline.ingestion.pipeline import run_ingestion, run_title_scraping


def _make_gkg_row(record_id="20260315120000-1", url="http://example.com/article"):
    """Build a minimal 27-column GKG row."""
    row = [""] * 27
    row[0] = record_id
    row[1] = "20260315120000"
    row[3] = "example.com"
    row[4] = url
    return row


# ── run_ingestion ──────────────────────────────────────────────────────


class TestRunIngestion:
    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_basic_flow(self, mock_download, mock_upsert, mock_update_state):
        mock_download.return_value = [
            _make_gkg_row("rec-1", "http://a.com/1"),
            _make_gkg_row("rec-2", "http://b.com/2"),
        ]
        mock_upsert.return_value = {"id": "some-uuid"}

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip")

        assert result.rows_fetched == 2
        assert result.rows_normalized == 2
        assert result.rows_upserted == 2
        assert result.rows_failed == 0
        assert mock_upsert.call_count == 2
        mock_update_state.assert_called_once()

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_dry_run_skips_db(self, mock_download, mock_upsert, mock_update_state):
        mock_download.return_value = [_make_gkg_row()]

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip", dry_run=True)

        assert result.rows_upserted == 1
        mock_upsert.assert_not_called()
        mock_update_state.assert_not_called()

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_skips_unnormalizable_rows(self, mock_download, mock_upsert, mock_update_state):
        # Row too short → normalize_row returns None
        mock_download.return_value = [["short", "row"]]

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip")

        assert result.rows_skipped == 1
        assert result.rows_normalized == 0
        mock_upsert.assert_not_called()

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_deduplicates_within_batch(self, mock_download, mock_upsert, mock_update_state):
        # Same URL appears twice
        mock_download.return_value = [
            _make_gkg_row("rec-1", "http://a.com/article"),
            _make_gkg_row("rec-2", "http://a.com/article"),
        ]
        mock_upsert.return_value = {"id": "some-uuid"}

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip")

        assert result.rows_normalized == 2
        assert result.rows_upserted == 1
        assert result.duplicate_urls == 1
        assert mock_upsert.call_count == 1

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_upsert_failure_counted(self, mock_download, mock_upsert, mock_update_state):
        mock_download.return_value = [_make_gkg_row()]
        mock_upsert.side_effect = Exception("DB error")

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip")

        assert result.rows_failed == 1
        assert result.rows_upserted == 0
        # Checkpoint should NOT be updated since nothing succeeded
        mock_update_state.assert_not_called()

    @patch("gdelt_event_pipeline.ingestion.pipeline.get_latest_gkg_url")
    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_resolves_url_when_not_provided(
        self, mock_download, mock_upsert, mock_update_state, mock_get_url
    ):
        mock_get_url.return_value = "http://example.com/latest.gkg.csv.zip"
        mock_download.return_value = []

        run_ingestion()

        mock_get_url.assert_called_once()
        mock_download.assert_called_once_with("http://example.com/latest.gkg.csv.zip", timeout=30)

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_pipeline_state")
    @patch("gdelt_event_pipeline.ingestion.pipeline.upsert_article")
    @patch("gdelt_event_pipeline.ingestion.pipeline.download_and_parse_gkg")
    def test_empty_fetch_no_checkpoint(self, mock_download, mock_upsert, mock_update_state):
        mock_download.return_value = []

        result = run_ingestion(gkg_url="http://example.com/test.gkg.csv.zip")

        assert result.rows_fetched == 0
        mock_update_state.assert_not_called()


# ── run_title_scraping ─────────────────────────────────────────────────


class TestRunTitleScraping:
    @patch("gdelt_event_pipeline.ingestion.pipeline.update_article_title")
    @patch("gdelt_event_pipeline.ingestion.scraper.scrape_titles")
    @patch("gdelt_event_pipeline.ingestion.pipeline.get_untitled_articles")
    def test_scrapes_and_updates(self, mock_get, mock_scrape, mock_update):
        mock_get.return_value = [
            {"id": "id-1", "url": "http://a.com"},
            {"id": "id-2", "url": "http://b.com"},
        ]
        mock_scrape.return_value = {"id-1": "Title A"}

        attempted, succeeded = run_title_scraping()

        assert attempted == 2
        assert succeeded == 1
        mock_update.assert_called_once_with("id-1", "Title A")

    @patch("gdelt_event_pipeline.ingestion.pipeline.update_article_title")
    @patch("gdelt_event_pipeline.ingestion.scraper.scrape_titles")
    @patch("gdelt_event_pipeline.ingestion.pipeline.get_untitled_articles")
    def test_no_untitled_articles(self, mock_get, mock_scrape, mock_update):
        mock_get.return_value = []

        attempted, succeeded = run_title_scraping()

        assert attempted == 0
        assert succeeded == 0
        mock_scrape.assert_not_called()
