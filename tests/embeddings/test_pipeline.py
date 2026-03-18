"""Tests for the embedding pipeline orchestrator."""

import json
from unittest.mock import MagicMock, patch

from gdelt_event_pipeline.config.settings import EmbeddingSettings
from gdelt_event_pipeline.embeddings.pipeline import run_embedding


def _make_article(article_id, title, themes=None):
    return {
        "id": article_id,
        "title": title,
        "themes": json.dumps(themes) if themes else None,
        "locations": None,
        "persons": None,
        "organizations": None,
    }


class TestRunEmbedding:
    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_basic_flow(self, mock_get, mock_embed, mock_update):
        mock_get.return_value = [
            _make_article("id-1", "Earthquake in Turkey"),
            _make_article("id-2", "Stock market crash"),
        ]
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        settings = EmbeddingSettings()
        result = run_embedding(settings, limit=10)

        assert result.articles_fetched == 2
        assert result.articles_embedded == 2
        assert result.articles_skipped == 0
        assert result.articles_failed == 0

        mock_embed.assert_called_once()
        assert mock_update.call_count == 2

    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_no_articles(self, mock_get, mock_embed, mock_update):
        mock_get.return_value = []

        result = run_embedding()
        assert result.articles_fetched == 0
        assert result.articles_embedded == 0
        mock_embed.assert_not_called()
        mock_update.assert_not_called()

    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_skips_empty_text(self, mock_get, mock_embed, mock_update):
        mock_get.return_value = [
            _make_article("id-1", None),  # no title, no metadata → empty text
            _make_article("id-2", "Valid title"),
        ]
        mock_embed.return_value = [[0.1] * 384]

        result = run_embedding()
        assert result.articles_skipped == 1
        assert result.articles_embedded == 1

        # Only the valid article should be embedded
        texts_arg = mock_embed.call_args[0][0]
        assert len(texts_arg) == 1
        assert "Valid title" in texts_arg[0]

    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_skips_no_title_even_with_metadata(self, mock_get, mock_embed, mock_update):
        """Articles with entities but no title should still be skipped."""
        mock_get.return_value = [
            _make_article("id-1", None, themes=[{"theme": "ECON"}]),
            _make_article("id-2", "Valid title"),
        ]
        mock_embed.return_value = [[0.1] * 384]

        result = run_embedding()
        assert result.articles_skipped == 1
        assert result.articles_embedded == 1

    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_db_failure_counted(self, mock_get, mock_embed, mock_update):
        mock_get.return_value = [_make_article("id-1", "Title")]
        mock_embed.return_value = [[0.1] * 384]
        mock_update.side_effect = RuntimeError("DB error")

        result = run_embedding()
        assert result.articles_failed == 1
        assert result.articles_embedded == 0

    @patch("gdelt_event_pipeline.embeddings.pipeline.update_article_embedding")
    @patch("gdelt_event_pipeline.embeddings.pipeline.embed_texts")
    @patch("gdelt_event_pipeline.embeddings.pipeline.get_unembedded_articles")
    def test_passes_settings_to_embed(self, mock_get, mock_embed, mock_update):
        mock_get.return_value = [_make_article("id-1", "Title")]
        mock_embed.return_value = [[0.1] * 384]

        settings = EmbeddingSettings(
            model_name="custom-model",
            batch_size=16,
            dimension=384,
        )
        run_embedding(settings)

        mock_embed.assert_called_once_with(
            ["Title"],
            model_name="custom-model",
            batch_size=16,
        )
        mock_update.assert_called_once_with("id-1", [0.1] * 384, "custom-model")
