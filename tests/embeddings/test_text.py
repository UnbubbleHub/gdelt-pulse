"""Tests for embedding text composition."""

import json

from gdelt_event_pipeline.embeddings.text import compose_embedding_text


class TestComposeEmbeddingText:
    def test_title_only(self):
        article = {"title": "Earthquake hits Turkey"}
        result = compose_embedding_text(article)
        assert result == "Earthquake hits Turkey"

    def test_title_with_themes(self):
        article = {
            "title": "Earthquake hits Turkey",
            "themes": json.dumps([{"theme": "NATURAL_DISASTER"}, {"theme": "HUMANITARIAN"}]),
        }
        result = compose_embedding_text(article)
        assert result == "Earthquake hits Turkey. Themes: NATURAL_DISASTER, HUMANITARIAN"

    def test_title_with_locations(self):
        article = {
            "title": "Earthquake hits Turkey",
            "locations": json.dumps([{"name": "Turkey"}, {"name": "Ankara"}]),
        }
        result = compose_embedding_text(article)
        assert result == "Earthquake hits Turkey. Locations: Turkey, Ankara"

    def test_title_with_persons(self):
        article = {
            "title": "Summit meeting",
            "persons": json.dumps(["Joe Biden", "Erdogan"]),
        }
        result = compose_embedding_text(article)
        assert result == "Summit meeting. Persons: Joe Biden, Erdogan"

    def test_title_with_organizations(self):
        article = {
            "title": "NATO meets",
            "organizations": json.dumps(["NATO", "United Nations"]),
        }
        result = compose_embedding_text(article)
        assert result == "NATO meets. Organizations: NATO, United Nations"

    def test_all_fields(self):
        article = {
            "title": "Big event",
            "themes": json.dumps([{"theme": "WAR"}]),
            "locations": json.dumps([{"name": "Ukraine"}]),
            "persons": json.dumps(["Zelenskyy"]),
            "organizations": json.dumps(["NATO"]),
        }
        result = compose_embedding_text(article)
        assert result == (
            "Big event. Themes: WAR. Locations: Ukraine. Persons: Zelenskyy. Organizations: NATO"
        )

    def test_no_title_metadata_only(self):
        article = {
            "title": None,
            "themes": json.dumps([{"theme": "ECONOMY"}]),
        }
        result = compose_embedding_text(article)
        assert result == "Themes: ECONOMY"

    def test_empty_article(self):
        result = compose_embedding_text({})
        assert result == ""

    def test_already_parsed_lists(self):
        article = {
            "title": "Test",
            "persons": ["Alice", "Bob"],
        }
        result = compose_embedding_text(article)
        assert result == "Test. Persons: Alice, Bob"

    def test_none_fields_ignored(self):
        article = {
            "title": "Test",
            "themes": None,
            "locations": None,
            "persons": None,
            "organizations": None,
        }
        result = compose_embedding_text(article)
        assert result == "Test"

    def test_themes_without_theme_key_skipped(self):
        article = {
            "title": "Test",
            "themes": json.dumps([{"score": 100}, {"theme": "WAR"}]),
        }
        result = compose_embedding_text(article)
        assert result == "Test. Themes: WAR"

    def test_locations_without_name_skipped(self):
        article = {
            "title": "Test",
            "locations": json.dumps([{"country_code": "US"}, {"name": "Berlin"}]),
        }
        result = compose_embedding_text(article)
        assert result == "Test. Locations: Berlin"
