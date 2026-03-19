"""Tests for centroid computation."""

from gdelt_event_pipeline.clustering.centroid import compute_new_centroid


class TestComputeNewCentroid:
    def test_first_update(self):
        # Cluster has 1 article (the seed). Adding a second.
        current = [1.0, 0.0, 0.0]
        new = [0.0, 1.0, 0.0]
        result = compute_new_centroid(current, new, current_count=1)
        assert result == [0.5, 0.5, 0.0]

    def test_running_average_three(self):
        # Cluster has 2 articles, adding a third.
        current = [0.5, 0.5, 0.0]  # average of [1,0,0] and [0,1,0]
        new = [0.0, 0.0, 1.0]
        result = compute_new_centroid(current, new, current_count=2)
        for actual, expected in zip(result, [1 / 3, 1 / 3, 1 / 3], strict=True):
            assert abs(actual - expected) < 1e-9

    def test_identical_vectors(self):
        current = [0.5, 0.5]
        new = [0.5, 0.5]
        result = compute_new_centroid(current, new, current_count=5)
        assert result == [0.5, 0.5]

    def test_high_count_stability(self):
        # With a large count, the new vector barely shifts the centroid.
        current = [1.0, 0.0]
        new = [0.0, 1.0]
        result = compute_new_centroid(current, new, current_count=1000)
        assert result[0] > 0.999
        assert result[1] < 0.001
