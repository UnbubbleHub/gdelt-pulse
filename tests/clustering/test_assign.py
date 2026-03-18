"""Tests for single-pass article assignment with entity-aware scoring."""

from datetime import datetime, timezone
from unittest.mock import patch

from gdelt_event_pipeline.clustering.assign import assign_article


def _make_article(article_id="art-1", title="Test Title", embedding=None,
                  locations=None, persons=None, organizations=None):
    return {
        "id": article_id,
        "title": title,
        "embedding": embedding or [0.1] * 384,
        "gdelt_timestamp": datetime(2026, 3, 16, tzinfo=timezone.utc),
        "locations": locations,
        "persons": persons,
        "organizations": organizations,
    }


def _make_cluster(cluster_id="clust-1", cosine_distance=0.1, article_count=5, centroid=None):
    return {
        "id": cluster_id,
        "cosine_distance": cosine_distance,
        "article_count": article_count,
        "centroid_embedding": centroid or [0.1] * 384,
    }


@patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.get_cluster_entity_sample")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_assigns_to_existing_above_threshold(
    mock_find, mock_sample, mock_assign, mock_update_centroid
):
    mock_find.return_value = [_make_cluster(cosine_distance=0.1)]
    mock_sample.return_value = []  # no entity data

    result = assign_article(_make_article())

    assert not result.is_new_cluster
    assert result.cluster_id == "clust-1"
    mock_assign.assert_called_once()
    mock_update_centroid.assert_called_once()


@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.create_cluster")
@patch("gdelt_event_pipeline.clustering.assign.get_cluster_entity_sample")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_creates_new_cluster_below_threshold(
    mock_find, mock_sample, mock_create, mock_assign
):
    mock_find.return_value = [_make_cluster(cosine_distance=0.5)]
    mock_sample.return_value = []
    mock_create.return_value = {"id": "new-clust"}

    result = assign_article(_make_article())

    assert result.is_new_cluster
    assert result.cluster_id == "new-clust"


@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.create_cluster")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_creates_new_cluster_when_none_exist(mock_find, mock_create, mock_assign):
    mock_find.return_value = []
    mock_create.return_value = {"id": "first-clust"}

    result = assign_article(_make_article())

    assert result.is_new_cluster
    assert result.cluster_id == "first-clust"


@patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.get_cluster_entity_sample")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_entity_overlap_boosts_borderline_match(
    mock_find, mock_sample, mock_assign, mock_update
):
    # cosine similarity = 0.68 (below 0.70 threshold)
    mock_find.return_value = [_make_cluster(cosine_distance=0.32)]

    # But strong entity overlap should push it above threshold
    import json
    mock_sample.return_value = [
        {"locations": json.dumps([{"name": "Berlin"}]), "persons": json.dumps(["Scholz"]),
         "organizations": None}
    ]

    article = _make_article(
        locations=json.dumps([{"name": "Berlin"}]),
        persons=json.dumps(["Scholz"]),
    )
    result = assign_article(article)

    # Entity overlap = 1.0 for both locations and persons
    # Combined = 0.68 + 0.25 * (0.50 + 0.35) = 0.68 + 0.2125 = 0.8925
    assert not result.is_new_cluster


@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.create_cluster")
@patch("gdelt_event_pipeline.clustering.assign.get_cluster_entity_sample")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_picks_best_candidate_across_multiple(
    mock_find, mock_sample, mock_create, mock_assign
):
    # Two candidates: first has closer cosine but no entity match,
    # second has slightly worse cosine but strong entity match
    mock_find.return_value = [
        _make_cluster(cluster_id="close-no-entity", cosine_distance=0.15),
        _make_cluster(cluster_id="far-with-entity", cosine_distance=0.25),
    ]

    import json
    mock_sample.side_effect = [
        # close-no-entity: no matching entities
        [{"locations": json.dumps([{"name": "Tokyo"}]), "persons": None, "organizations": None}],
        # far-with-entity: matching entities
        [{"locations": json.dumps([{"name": "Berlin"}]), "persons": json.dumps(["Scholz"]),
          "organizations": None}],
    ]

    article = _make_article(
        locations=json.dumps([{"name": "Berlin"}]),
        persons=json.dumps(["Scholz"]),
    )

    with patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid"):
        result = assign_article(article)

    # Second candidate should win despite worse cosine
    assert result.cluster_id == "far-with-entity"


@patch("gdelt_event_pipeline.clustering.assign.update_cluster_centroid")
@patch("gdelt_event_pipeline.clustering.assign.assign_article_to_cluster")
@patch("gdelt_event_pipeline.clustering.assign.get_cluster_entity_sample")
@patch("gdelt_event_pipeline.clustering.assign.find_nearest_cluster")
def test_centroid_updated_on_assignment(
    mock_find, mock_sample, mock_assign, mock_update_centroid
):
    centroid = [1.0] * 384
    mock_find.return_value = [
        _make_cluster(cosine_distance=0.05, article_count=1, centroid=centroid)
    ]
    mock_sample.return_value = []

    article = _make_article(embedding=[0.0] * 384)
    assign_article(article)

    new_centroid = mock_update_centroid.call_args[0][1]
    assert abs(new_centroid[0] - 0.5) < 1e-9
