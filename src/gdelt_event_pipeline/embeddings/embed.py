"""Embedding generation — sentence-transformers or fastembed backend."""

import logging
import os

logger = logging.getLogger(__name__)

_model = None
_fe_model: dict = {}


def load_model(model_name: str):
    """Load (and cache) a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer  # lazy: not needed at import time

    global _model
    if _model is None or _model.model_card_data.model_id != model_name:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def load_fe_model(model_name: str):
    """Load (and cache) a fastembed TextEmbedding model."""
    from fastembed import TextEmbedding  # lazy: only when backend is configured

    if model_name not in _fe_model:
        logger.info("Loading fastembed model: %s", model_name)
        _fe_model[model_name] = TextEmbedding(model_name)
    return _fe_model[model_name]


def embed_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[list[float]]:
    """Encode a list of texts into embedding vectors.

    Backend is selected by the EMBEDDING_BACKEND env var:
      - "fastembed" or unset  → fastembed.TextEmbedding (default)
      - "sentence-transformers" → SentenceTransformer (requires optional install)
    """
    if not texts:
        return []

    backend = os.environ.get("EMBEDDING_BACKEND", "fastembed")
    if backend == "fastembed":
        fe_model = load_fe_model(model_name)
        return [v.tolist() for v in fe_model.embed(texts)]

    model = load_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()
