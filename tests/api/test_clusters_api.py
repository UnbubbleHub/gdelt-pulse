"""Tests for /api/clusters endpoints."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import make_cluster, make_mock_pool


class TestListClusters:
    def test_returns_list(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_active_clusters",
            return_value=[make_cluster()],
        ):
            resp = client_no_db.get("/api/clusters")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sort_recent(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_active_clusters",
            return_value=[],
        ) as mock:
            client_no_db.get("/api/clusters?sort=recent")
        mock.assert_called_once()
        assert mock.call_args.kwargs["sort"] == "recent"

    def test_sort_articles(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_active_clusters",
            return_value=[],
        ) as mock:
            client_no_db.get("/api/clusters?sort=articles")
        assert mock.call_args.kwargs["sort"] == "articles"

    def test_limit_passed(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_active_clusters",
            return_value=[],
        ) as mock:
            client_no_db.get("/api/clusters?limit=25")
        assert mock.call_args.kwargs["limit"] == 25

    def test_strips_internal_fields(self, client_no_db):
        cluster = make_cluster()
        cluster["centroid_embedding"] = [0.1, 0.2]
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_active_clusters",
            return_value=[cluster],
        ):
            resp = client_no_db.get("/api/clusters")
        assert "centroid_embedding" not in resp.json()[0]

    def test_filtered_query_uses_db(self, client_no_db):
        pool = make_mock_pool(fetchall_return=[make_cluster()])
        with patch("gdelt_event_pipeline.storage.database.get_pool", return_value=pool):
            resp = client_no_db.get("/api/clusters?location=Rome")
        assert resp.status_code == 200
        sql = pool._mock_cur.execute.call_args[0][0]
        assert "ILIKE" in sql


class TestClusterDetail:
    def test_returns_cluster_and_articles(self, client_no_db):
        cluster = make_cluster(id="test-id")
        with (
            patch(
                "gdelt_event_pipeline.api.routers.clusters.get_cluster_by_id",
                return_value=cluster,
            ),
            patch(
                "gdelt_event_pipeline.api.routers.clusters.get_cluster_articles",
                return_value=[],
            ),
        ):
            resp = client_no_db.get("/api/clusters/test-id")
        assert resp.status_code == 200
        data = resp.json()
        assert "cluster" in data
        assert "articles" in data

    def test_returns_404_when_not_found(self, client_no_db):
        with patch(
            "gdelt_event_pipeline.api.routers.clusters.get_cluster_by_id",
            return_value=None,
        ):
            resp = client_no_db.get("/api/clusters/missing-id")
        assert resp.status_code == 404


class TestDisabledEndpoints:
    def test_globe_returns_404(self, client_no_db):
        assert client_no_db.get("/globe").status_code == 404

    def test_polarization_returns_404(self, client_no_db):
        assert client_no_db.get("/polarization").status_code == 404

    def test_gravity_returns_404(self, client_no_db):
        assert client_no_db.get("/gravity").status_code == 404

    def test_api_globe_returns_404(self, client_no_db):
        assert client_no_db.get("/api/globe/clusters").status_code == 404

    def test_api_polarization_returns_404(self, client_no_db):
        assert client_no_db.get("/api/polarization").status_code == 404

    def test_api_asymmetry_returns_404(self, client_no_db):
        assert client_no_db.get("/api/asymmetry").status_code == 404

    def test_api_gravity_returns_404(self, client_no_db):
        assert client_no_db.get("/api/gravity/graph").status_code == 404
