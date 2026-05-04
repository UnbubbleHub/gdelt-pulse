"""Tests for cluster storage operations."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.conftest import make_cluster, make_mock_pool

MODULE = "gdelt_event_pipeline.storage.clusters"


@pytest.fixture
def patch_pool():
    pool = make_mock_pool()
    with patch(f"{MODULE}.get_pool", return_value=pool):
        yield pool


class TestCreateCluster:
    def test_inserts_and_returns_row(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import create_cluster

        expected = make_cluster()
        patch_pool._mock_cur.fetchone.return_value = expected
        result = create_cluster(representative_title="Breaking News")
        assert result == expected
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "INSERT INTO clusters" in sql
        assert "RETURNING *" in sql

    def test_passes_title_and_embedding(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import create_cluster

        patch_pool._mock_cur.fetchone.return_value = make_cluster()
        create_cluster(representative_title="Test", centroid_embedding=[0.1, 0.2])
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == "Test"
        assert params[1] == [0.1, 0.2]


class TestAssignArticleToCluster:
    def test_inserts_membership_and_updates_cluster(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import assign_article_to_cluster

        patch_pool._mock_cur.fetchone.return_value = {"id": "mem-1"}
        result = assign_article_to_cluster("art-1", "clus-1", similarity_score=0.95)
        assert result == {"id": "mem-1"}
        assert patch_pool._mock_cur.execute.call_count == 2

    def test_update_uses_cluster_id_four_times(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import assign_article_to_cluster

        patch_pool._mock_cur.fetchone.return_value = {"id": "mem-1"}
        assign_article_to_cluster("art-1", "clus-1")
        update_call = patch_pool._mock_cur.execute.call_args_list[1]
        params = update_call[0][1]
        assert params == ("clus-1", "clus-1", "clus-1", "clus-1")


class TestFindNearestCluster:
    def test_with_max_age_hours(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import find_nearest_cluster

        patch_pool._mock_cur.fetchall.return_value = []
        find_nearest_cluster([0.1, 0.2], limit=5, max_age_hours=72)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "make_interval" in sql
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ([0.1, 0.2], 72, [0.1, 0.2], 5)

    def test_without_max_age_hours(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import find_nearest_cluster

        patch_pool._mock_cur.fetchall.return_value = []
        find_nearest_cluster([0.1, 0.2], limit=3, max_age_hours=None)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "make_interval" not in sql
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ([0.1, 0.2], [0.1, 0.2], 3)

    def test_returns_list(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import find_nearest_cluster

        clusters = [make_cluster()]
        patch_pool._mock_cur.fetchall.return_value = clusters
        result = find_nearest_cluster([0.1], max_age_hours=None)
        assert result == clusters


class TestGetClusterById:
    def test_returns_cluster_when_found(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_by_id

        expected = make_cluster()
        patch_pool._mock_cur.fetchone.return_value = expected
        assert get_cluster_by_id("some-id") == expected

    def test_returns_none_when_not_found(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_by_id

        patch_pool._mock_cur.fetchone.return_value = None
        assert get_cluster_by_id("missing") is None


class TestGetActiveClusters:
    def test_sort_recent(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_active_clusters

        patch_pool._mock_cur.fetchall.return_value = []
        get_active_clusters(sort="recent")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "last_article_at DESC" in sql

    def test_sort_articles(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_active_clusters

        patch_pool._mock_cur.fetchall.return_value = []
        get_active_clusters(sort="articles")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "article_count DESC" in sql

    def test_sort_oldest(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_active_clusters

        patch_pool._mock_cur.fetchall.return_value = []
        get_active_clusters(sort="oldest")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "first_article_at ASC" in sql

    def test_passes_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_active_clusters

        patch_pool._mock_cur.fetchall.return_value = []
        get_active_clusters(limit=25)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (25,)


class TestGetClusterArticles:
    def test_joins_memberships(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_cluster_articles("clus-1")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "JOIN cluster_memberships" in sql
        assert "similarity_score" in sql


class TestGetClusterEntitySample:
    def test_delegates_to_batch(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_entity_sample

        patch_pool._mock_cur.fetchall.return_value = []
        result = get_cluster_entity_sample("clus-1", limit=3)
        assert result == []


class TestGetClusterEntitySamples:
    def test_empty_list_returns_empty_dict(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_entity_samples

        result = get_cluster_entity_samples([])
        assert result == {}
        patch_pool._mock_cur.execute.assert_not_called()

    def test_uses_lateral_join(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_entity_samples

        patch_pool._mock_cur.fetchall.return_value = [
            {"cluster_id": "c1", "locations": [], "persons": [], "organizations": []},
            {"cluster_id": "c2", "locations": [], "persons": [], "organizations": []},
        ]
        result = get_cluster_entity_samples(["c1", "c2"])
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "LATERAL" in sql
        assert "c1" in result
        assert "c2" in result

    def test_groups_by_cluster_id(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import get_cluster_entity_samples

        patch_pool._mock_cur.fetchall.return_value = [
            {"cluster_id": "c1", "locations": [1], "persons": [], "organizations": []},
            {"cluster_id": "c1", "locations": [2], "persons": [], "organizations": []},
            {"cluster_id": "c2", "locations": [3], "persons": [], "organizations": []},
        ]
        result = get_cluster_entity_samples(["c1", "c2"])
        assert len(result["c1"]) == 2
        assert len(result["c2"]) == 1


class TestUpdateClusterCentroid:
    def test_executes_update(self, patch_pool):
        from gdelt_event_pipeline.storage.clusters import update_cluster_centroid

        update_cluster_centroid("clus-1", [0.1, 0.2, 0.3])
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ([0.1, 0.2, 0.3], "clus-1")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "centroid_embedding" in sql
