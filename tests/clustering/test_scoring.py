"""Tests for entity-aware similarity scoring."""

import json

from gdelt_event_pipeline.clustering.scoring import (
    compute_combined_score,
    compute_entity_overlap,
    extract_entity_sets,
    merge_entity_sets,
)


class TestExtractEntitySets:
    def test_from_json_strings(self):
        article = {
            "locations": json.dumps([{"name": "Berlin"}, {"name": "Paris"}]),
            "persons": json.dumps(["Joe Biden", "Macron"]),
            "organizations": json.dumps(["NATO"]),
        }
        result = extract_entity_sets(article)
        assert result["locations"] == {"berlin", "paris"}
        assert result["persons"] == {"joe biden", "macron"}
        assert result["organizations"] == {"nato"}

    def test_from_parsed_lists(self):
        article = {
            "locations": [{"name": "Berlin"}],
            "persons": ["Joe Biden"],
            "organizations": ["NATO"],
        }
        result = extract_entity_sets(article)
        assert result["locations"] == {"berlin"}
        assert result["persons"] == {"joe biden"}

    def test_empty_article(self):
        result = extract_entity_sets({})
        assert result["locations"] == set()
        assert result["persons"] == set()
        assert result["organizations"] == set()

    def test_none_fields(self):
        result = extract_entity_sets(
            {"locations": None, "persons": None, "organizations": None}
        )
        assert result["locations"] == set()

    def test_locations_without_name_skipped(self):
        article = {
            "locations": json.dumps([{"country_code": "US"}, {"name": "Berlin"}]),
        }
        result = extract_entity_sets(article)
        assert result["locations"] == {"berlin"}


class TestMergeEntitySets:
    def test_merge_two(self):
        a = {"locations": {"berlin"}, "persons": {"biden"}, "organizations": set()}
        b = {"locations": {"paris"}, "persons": {"biden", "macron"}, "organizations": {"nato"}}
        merged = merge_entity_sets([a, b])
        assert merged["locations"] == {"berlin", "paris"}
        assert merged["persons"] == {"biden", "macron"}
        assert merged["organizations"] == {"nato"}

    def test_empty_list(self):
        merged = merge_entity_sets([])
        assert merged["locations"] == set()


class TestComputeEntityOverlap:
    def test_perfect_overlap(self):
        entities = {"locations": {"berlin"}, "persons": {"biden"}, "organizations": {"nato"}}
        score = compute_entity_overlap(entities, entities)
        assert abs(score - 1.0) < 1e-9

    def test_no_overlap(self):
        a = {"locations": {"berlin"}, "persons": {"biden"}, "organizations": {"nato"}}
        b = {"locations": {"tokyo"}, "persons": {"kishida"}, "organizations": {"un"}}
        score = compute_entity_overlap(a, b)
        assert score == 0.0

    def test_partial_overlap(self):
        a = {"locations": {"berlin", "paris"}, "persons": {"biden"}, "organizations": set()}
        b = {"locations": {"berlin"}, "persons": {"biden", "macron"}, "organizations": set()}
        score = compute_entity_overlap(a, b)
        # locations: 1/2 * 0.50 = 0.25
        # persons: 1/2 * 0.35 = 0.175
        # organizations: both empty → skipped
        assert abs(score - 0.425) < 1e-9

    def test_both_empty(self):
        a = {"locations": set(), "persons": set(), "organizations": set()}
        b = {"locations": set(), "persons": set(), "organizations": set()}
        assert compute_entity_overlap(a, b) == 0.0

    def test_one_side_empty_type(self):
        a = {"locations": {"berlin"}, "persons": set(), "organizations": set()}
        b = {"locations": set(), "persons": set(), "organizations": set()}
        # locations: one side empty → no contribution
        assert compute_entity_overlap(a, b) == 0.0


class TestComputeCombinedScore:
    def test_cosine_only(self):
        assert compute_combined_score(0.8, 0.0) == 0.8

    def test_with_entity_bonus(self):
        score = compute_combined_score(0.7, 1.0, entity_weight=0.25)
        assert abs(score - 0.95) < 1e-9

    def test_entity_pushes_above_threshold(self):
        # cosine=0.65 alone is below 0.70 threshold
        # entity_overlap=1.0 adds 0.25 → combined=0.90, above threshold
        combined = compute_combined_score(0.65, 1.0, entity_weight=0.25)
        assert combined >= 0.70
