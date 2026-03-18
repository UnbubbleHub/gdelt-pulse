"""Embedding pipeline: fetch unembedded articles, embed, store."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gdelt_event_pipeline.config.settings import EmbeddingSettings
from gdelt_event_pipeline.embeddings.embed import embed_texts
from gdelt_event_pipeline.embeddings.text import compose_embedding_text
from gdelt_event_pipeline.storage.articles import (
    get_unembedded_articles,
    update_article_embedding,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Summary of one embedding run."""

    articles_fetched: int = 0
    articles_embedded: int = 0
    articles_skipped: int = 0
    articles_failed: int = 0


def run_embedding(
    settings: EmbeddingSettings | None = None,
    *,
    limit: int | None = None,
) -> EmbeddingResult:
    """Execute one embedding cycle.

    1. Fetch articles with no embedding
    2. Compose embedding text for each
    3. Generate embeddings in batch
    4. Store each embedding back to the database
    """
    if settings is None:
        settings = EmbeddingSettings()

    result = EmbeddingResult()

    articles = get_unembedded_articles(limit=limit)
    result.articles_fetched = len(articles)
    if not articles:
        logger.info("No unembedded articles found")
        return result

    logger.info("Embedding %d articles with %s", len(articles), settings.model_name)

    # Compose texts, tracking which articles produced usable input
    texts: list[str] = []
    valid_articles: list[dict] = []
    for article in articles:
        if not article.get("title"):
            result.articles_skipped += 1
            logger.debug(
                "Skipping article %s: no title", article.get("id"),
            )
            continue
        text = compose_embedding_text(article)
        if not text.strip():
            result.articles_skipped += 1
            continue
        texts.append(text)
        valid_articles.append(article)

    if not texts:
        logger.info("No articles produced usable embedding text")
        return result

    # Embed in one batch call
    vectors = embed_texts(
        texts,
        model_name=settings.model_name,
        batch_size=settings.batch_size,
    )

    # Store results
    for article, vector in zip(valid_articles, vectors):
        try:
            update_article_embedding(
                str(article["id"]),
                vector,
                settings.model_name,
            )
            result.articles_embedded += 1
        except Exception:
            logger.exception(
                "Failed to store embedding for article %s", article.get("id")
            )
            result.articles_failed += 1

    logger.info(
        "Embedding complete: fetched=%d embedded=%d skipped=%d failed=%d",
        result.articles_fetched,
        result.articles_embedded,
        result.articles_skipped,
        result.articles_failed,
    )
    return result
