"""Embedding generation — sentence-transformers or fastembed backend."""

import logging
import os

logger = logging.getLogger(__name__)

_model = None


def load_model(model_name: str):
    """Load (and cache) a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer  # lazy: not needed at import time

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

    Backend is selected by the EMBEDDING_BACKEND env var:
      - "fastembed"            → fastembed.TextEmbedding (Vercel)
      - "sentence-transformers" or unset → SentenceTransformer (Railway / local)
    """
    if not texts:
        return []

    backend = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers")
    if backend == "fastembed":
        from fastembed import TextEmbedding  # lazy: only when backend is configured
        fe_model = TextEmbedding(model_name)
        return [v.tolist() for v in fe_model.embed(texts)]

    model = load_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()
