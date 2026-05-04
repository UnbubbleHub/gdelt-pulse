"""Tests for article storage operations."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from tests.conftest import make_article, make_mock_pool

MODULE = "gdelt_event_pipeline.storage.articles"


@pytest.fixture
def patch_pool():
    pool = make_mock_pool()
    with patch(f"{MODULE}.get_pool", return_value=pool):
        yield pool


class TestFlattenParams:
    def test_returns_correct_column_count(self):
        from gdelt_event_pipeline.storage.articles import _UPSERT_COLUMNS, _flatten_params

        article = make_article()
        params = _flatten_params(article)
        assert len(params) == len(_UPSERT_COLUMNS) + 2

    def test_appends_gdelt_timestamp_for_seen_at_fields(self):
        from gdelt_event_pipeline.storage.articles import _flatten_params

        ts = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        article = make_article(gdelt_timestamp=ts)
        params = _flatten_params(article)
        assert params[-1] == ts
        assert params[-2] == ts

    def test_columns_in_expected_order(self):
        from gdelt_event_pipeline.storage.articles import _UPSERT_COLUMNS, _flatten_params

        article = make_article()
        params = _flatten_params(article)
        for i, col in enumerate(_UPSERT_COLUMNS):
            assert params[i] == article.get(col)


class TestUpsertArticles:
    def test_empty_list_returns_zero(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import upsert_articles

        assert upsert_articles([]) == 0
        patch_pool._mock_cur.execute.assert_not_called()

    def test_single_article(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import upsert_articles

        patch_pool._mock_cur.fetchall.return_value = [{"id": "abc"}]
        result = upsert_articles([make_article()])
        assert result == 1
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "INSERT INTO articles" in sql
        assert "ON CONFLICT" in sql

    def test_chunking_with_small_chunk_size(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import upsert_articles

        articles = [make_article(canonical_url=f"example.com/{i}") for i in range(5)]
        patch_pool._mock_cur.fetchall.return_value = [{"id": "x"}] * 2
        result = upsert_articles(articles, chunk_size=2)
        assert patch_pool._mock_cur.execute.call_count == 3
        assert result == 6

    def test_sql_contains_correct_placeholder_count(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import _UPSERT_COLUMNS, upsert_articles

        patch_pool._mock_cur.fetchall.return_value = [{"id": "x"}]
        upsert_articles([make_article()])
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        expected_placeholders = len(_UPSERT_COLUMNS) + 2
        assert f"({', '.join(['%s'] * expected_placeholders)})" in sql


class TestUpsertArticle:
    def test_delegates_to_batch(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import upsert_articles

        with patch(f"{MODULE}.upsert_articles", wraps=upsert_articles) as mock:
            from gdelt_event_pipeline.storage.articles import upsert_article

            patch_pool._mock_cur.fetchall.return_value = [{"id": "x"}]
            upsert_article(make_article())
            mock.assert_called_once()
            assert len(mock.call_args[0][0]) == 1


class TestGetArticleByCanonicalUrl:
    def test_returns_article_when_found(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_article_by_canonical_url

        expected = make_article()
        patch_pool._mock_cur.fetchone.return_value = expected
        result = get_article_by_canonical_url("example.com/article")
        assert result == expected
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "canonical_url" in sql

    def test_returns_none_when_not_found(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_article_by_canonical_url

        patch_pool._mock_cur.fetchone.return_value = None
        assert get_article_by_canonical_url("missing") is None


class TestGetRecentArticles:
    def test_returns_list(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_recent_articles

        articles = [make_article(), make_article()]
        patch_pool._mock_cur.fetchall.return_value = articles
        result = get_recent_articles(limit=10)
        assert result == articles
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "title IS NOT NULL" in sql
        assert "ORDER BY gdelt_timestamp DESC" in sql

    def test_passes_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_recent_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_recent_articles(limit=25)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (25,)


class TestGetArticlesSince:
    def test_passes_since_and_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_articles_since

        since = datetime(2026, 4, 1, tzinfo=UTC)
        patch_pool._mock_cur.fetchall.return_value = []
        get_articles_since(since, limit=50)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (since, 50)


class TestGetUnembeddedArticles:
    def test_with_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_unembedded_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_unembedded_articles(limit=100)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "LIMIT" in sql

    def test_without_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_unembedded_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_unembedded_articles()
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "LIMIT" not in sql


class TestGetUnclusteredArticles:
    def test_sql_uses_left_join(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_unclustered_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_unclustered_articles()
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "LEFT JOIN cluster_memberships" in sql
        assert "cm.id IS NULL" in sql

    def test_with_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_unclustered_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_unclustered_articles(limit=50)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "LIMIT" in sql


class TestGetUntitledArticles:
    def test_passes_max_attempts(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_untitled_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_untitled_articles(max_attempts=3)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == 3

    def test_with_limit(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import get_untitled_articles

        patch_pool._mock_cur.fetchall.return_value = []
        get_untitled_articles(limit=20, max_attempts=2)
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == [2, 20]


class TestIncrementScrapeAttempts:
    def test_empty_list_is_noop(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import increment_scrape_attempts

        increment_scrape_attempts([])
        patch_pool._mock_cur.execute.assert_not_called()

    def test_passes_ids_to_any(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import increment_scrape_attempts

        ids = ["id-1", "id-2"]
        increment_scrape_attempts(ids)
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "ANY(%s)" in sql
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (ids,)


class TestUpdateArticleTitle:
    def test_executes_update(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_title

        update_article_title("art-1", "New Title")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "SET title = %s" in sql
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ("New Title", "art-1")


class TestUpdateArticleTitles:
    def test_empty_dict_returns_zero(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_titles

        assert update_article_titles({}) == 0
        patch_pool._mock_cur.execute.assert_not_called()

    def test_batch_update_builds_values_clause(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_titles

        patch_pool._mock_cur.rowcount = 3
        titles = {"id-1": "Title A", "id-2": "Title B", "id-3": "Title C"}
        result = update_article_titles(titles)
        assert result == 3
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "FROM (VALUES" in sql
        assert sql.count("(%s::uuid, %s)") == 3

    def test_params_alternate_id_and_title(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_titles

        patch_pool._mock_cur.rowcount = 1
        update_article_titles({"id-1": "Hello"})
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == ["id-1", "Hello"]


class TestUpdateArticleEmbedding:
    def test_executes_update(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_embedding

        vec = [0.1, 0.2, 0.3]
        update_article_embedding("art-1", vec, "model-v1")
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params == (vec, "model-v1", "art-1")


class TestUpdateArticleEmbeddings:
    def test_empty_list_returns_zero(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_embeddings

        assert update_article_embeddings([], "model-v1") == 0

    def test_batch_update_with_chunking(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_embeddings

        updates = [(f"id-{i}", [0.1, 0.2]) for i in range(5)]
        patch_pool._mock_cur.rowcount = 2
        result = update_article_embeddings(updates, "model-v1", chunk_size=2)
        assert patch_pool._mock_cur.execute.call_count == 3
        assert result == 6

    def test_model_is_first_param(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_embeddings

        patch_pool._mock_cur.rowcount = 1
        update_article_embeddings([("id-1", [0.5])], "my-model")
        params = patch_pool._mock_cur.execute.call_args[0][1]
        assert params[0] == "my-model"

    def test_sql_uses_vector_cast(self, patch_pool):
        from gdelt_event_pipeline.storage.articles import update_article_embeddings

        patch_pool._mock_cur.rowcount = 1
        update_article_embeddings([("id-1", [0.5])], "m")
        sql = patch_pool._mock_cur.execute.call_args[0][0]
        assert "%s::vector" in sql
        assert "%s::uuid" in sql
