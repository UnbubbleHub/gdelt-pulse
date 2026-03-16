"""Tests for single-pass article assignment."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from gdelt_event_pipeline.clustering.assign import assign_article


def _make_article(article_id="art-1", title="Test Title", embedding=None):
    return {
        "id": article_id,
        "title": title,
        "embedding": embedding or [0.1] * 384,
        "gdelt_timestamp": datetime(2026, 3, 16, tzinfo=timezone.utc),
    }


def _make_cluster(cluster_id="clust-1", cosine_distance=0.1, article_count=5, centroid=None):
    return {
        "id": cluster_id,
        "cosine_distance": cosine_distance,
        "article_count": article_count,
        "centroid_embedding": centroid or [0.1] * 384,
    }


class TestAssignArticle:
    @patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
    @patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
    def test_assigns_to_existing_above_threshold(
        self, mock_find, mock_assign, mock_update_centroid
    ):
        # cosine_distance=0.1 → similarity=0.9 → above default 0.75
        mock_find.return_value = [_make_cluster(cosine_distance=0.1)]

        result = assign_article(_make_article())

        assert not result.is_new_cluster
        assert result.cluster_id == "clust-1"
        assert abs(result.similarity - 0.9) < 1e-9
        mock_assign.assert_called_once()
        mock_update_centroid.assert_called_once()

    @patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.create_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
    def test_creates_new_cluster_below_threshold(
        self, mock_find, mock_create, mock_assign
    ):
        # cosine_distance=0.5 → similarity=0.5 → below default 0.75
        mock_find.return_value = [_make_cluster(cosine_distance=0.5)]
        mock_create.return_value = {"id": "new-clust"}

        result = assign_article(_make_article())

        assert result.is_new_cluster
        assert result.cluster_id == "new-clust"
        assert result.similarity == 1.0
        mock_create.assert_called_once()

    @patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.create_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
    def test_creates_new_cluster_when_none_exist(
        self, mock_find, mock_create, mock_assign
    ):
        mock_find.return_value = []
        mock_create.return_value = {"id": "first-clust"}

        result = assign_article(_make_article())

        assert result.is_new_cluster
        assert result.cluster_id == "first-clust"

    @patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
    @patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
    def test_custom_threshold(self, mock_find, mock_assign, mock_update):
        # similarity = 0.9, but threshold = 0.95
        mock_find.return_value = [_make_cluster(cosine_distance=0.1)]

        with patch(
            "gdelt_event_pipeline.clustering.assign.create_cluster"
        ) as mock_create:
            mock_create.return_value = {"id": "strict-clust"}
            result = assign_article(_make_article(), threshold=0.95)

        assert result.is_new_cluster

    @patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
    @patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
    @patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
    def test_centroid_updated_on_assignment(
        self, mock_find, mock_assign, mock_update_centroid
    ):
        centroid = [1.0] * 384
        mock_find.return_value = [
            _make_cluster(cosine_distance=0.05, article_count=1, centroid=centroid)
        ]

        article = _make_article(embedding=[0.0] * 384)
        assign_article(article)

        # New centroid should be average of [1.0]*384 and [0.0]*384 = [0.5]*384
        call_args = mock_update_centroid.call_args
        new_centroid = call_args[0][1]
        assert abs(new_centroid[0] - 0.5) < 1e-9
