"""Tests for Reciprocal Rank Fusion."""

from gdelt_event_pipeline.query.ranking import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_disjoint_lists(self):
        result = reciprocal_rank_fusion(["a", "b"], ["c", "d"], semantic_weight=0.5)
        ids = [doc_id for doc_id, _ in result]
        # All four should appear
        assert set(ids) == {"a", "b", "c", "d"}

    def test_overlapping_results_rank_higher(self):
        result = reciprocal_rank_fusion(
            ["a", "b", "c"],
            ["b", "a", "d"],
            semantic_weight=0.5,
        )
        ids = [doc_id for doc_id, _ in result]
        # "a" and "b" appear in both lists, should be top 2
        assert set(ids[:2]) == {"a", "b"}

    def test_pure_semantic(self):
        result = reciprocal_rank_fusion(
            ["a", "b", "c"],
            ["c", "b", "a"],
            semantic_weight=1.0,
        )
        ids = [doc_id for doc_id, _ in result]
        assert ids == ["a", "b", "c"]

    def test_pure_keyword(self):
        result = reciprocal_rank_fusion(
            ["a", "b", "c"],
            ["c", "b", "a"],
            semantic_weight=0.0,
        )
        ids = [doc_id for doc_id, _ in result]
        assert ids == ["c", "b", "a"]

    def test_empty_inputs(self):
        result = reciprocal_rank_fusion([], [], semantic_weight=0.5)
        assert result == []

    def test_one_empty_list(self):
        result = reciprocal_rank_fusion(["a", "b"], [], semantic_weight=0.5)
        ids = [doc_id for doc_id, _ in result]
        assert ids == ["a", "b"]

    def test_scores_are_positive(self):
        result = reciprocal_rank_fusion(["a", "b"], ["b", "c"], semantic_weight=0.5)
        for _, score in result:
            assert score > 0

    def test_scores_descending(self):
        result = reciprocal_rank_fusion(
            ["a", "b", "c", "d"],
            ["d", "c", "b", "a"],
            semantic_weight=0.5,
        )
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
