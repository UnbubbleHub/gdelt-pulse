"""Tests for the clustering pipeline orchestrator."""

from datetime import UTC, datetime
from unittest.mock import patch

from gdelt_event_pipeline.clustering.assign import AssignmentResult
from gdelt_event_pipeline.clustering.pipeline import run_clustering


def _make_article(article_id):
    return {
        "id": article_id,
        "title": f"Title {article_id}",
        "embedding": [0.1] * 384,
        "gdelt_timestamp": datetime(2026, 3, 16, tzinfo=UTC),
    }


class TestRunClustering:
    @patch("gdelt_event_pipeline.clustering.pipeline.assign_article")
    @patch("gdelt_event_pipeline.clustering.pipeline.get_unclustered_articles")
    def test_basic_flow(self, mock_get, mock_assign):
        mock_get.return_value = [_make_article("a1"), _make_article("a2"), _make_article("a3")]
        mock_assign.side_effect = [
            AssignmentResult(cluster_id="c1", similarity=0.9, is_new_cluster=False),
            AssignmentResult(cluster_id="c2", similarity=1.0, is_new_cluster=True),
            AssignmentResult(cluster_id="c1", similarity=0.85, is_new_cluster=False),
        ]

        result = run_clustering(limit=10)

        assert result.articles_processed == 3
        assert result.assigned_to_existing == 2
        assert result.new_clusters_created == 1
        assert result.articles_failed == 0

    @patch("gdelt_event_pipeline.clustering.pipeline.assign_article")
    @patch("gdelt_event_pipeline.clustering.pipeline.get_unclustered_articles")
    def test_no_articles(self, mock_get, mock_assign):
        mock_get.return_value = []

        result = run_clustering()
        assert result.articles_processed == 0
        mock_assign.assert_not_called()

    @patch("gdelt_event_pipeline.clustering.pipeline.assign_article")
    @patch("gdelt_event_pipeline.clustering.pipeline.get_unclustered_articles")
    def test_failure_counted(self, mock_get, mock_assign):
        mock_get.return_value = [_make_article("a1")]
        mock_assign.side_effect = RuntimeError("DB error")

        result = run_clustering()
        assert result.articles_failed == 1
        assert result.articles_processed == 0

    @patch("gdelt_event_pipeline.clustering.pipeline.assign_article")
    @patch("gdelt_event_pipeline.clustering.pipeline.get_unclustered_articles")
    def test_skips_articles_without_title(self, mock_get, mock_assign):
        titled = _make_article("a1")
        untitled = _make_article("a2")
        untitled["title"] = None

        mock_get.return_value = [titled, untitled]
        mock_assign.return_value = AssignmentResult(
            cluster_id="c1", similarity=0.9, is_new_cluster=False
        )

        result = run_clustering()

        assert result.articles_processed == 1
        assert result.articles_skipped == 1
        assert mock_assign.call_count == 1

    @patch("gdelt_event_pipeline.clustering.pipeline.assign_article")
    @patch("gdelt_event_pipeline.clustering.pipeline.get_unclustered_articles")
    def test_threshold_passed_through(self, mock_get, mock_assign):
        mock_get.return_value = [_make_article("a1")]
        mock_assign.return_value = AssignmentResult(
            cluster_id="c1", similarity=1.0, is_new_cluster=True
        )

        run_clustering(threshold=0.9)
        mock_assign.assert_called_once_with(mock_get.return_value[0], threshold=0.9)
