"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def load_model(model_name: str) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model."""
    global _model
    if _model is None or _model.model_card_data.model_id != model_name:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[list[float]]:
    """Encode a list of texts into embedding vectors.

    Returns a list of float lists, one per input text.
    """
    if not texts:
        return []

    model = load_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()
