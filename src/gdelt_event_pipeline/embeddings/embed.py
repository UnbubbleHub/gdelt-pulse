"""Embedding generation via fastembed."""

import logging

logger = logging.getLogger(__name__)

_fe_model: dict = {}


def load_fe_model(model_name: str):
    """Load (and cache) a fastembed TextEmbedding model."""
    from fastembed import TextEmbedding

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
    """Encode a list of texts into embedding vectors using fastembed."""
    if not texts:
        return []

    fe_model = load_fe_model(model_name)
    return [v.tolist() for v in fe_model.embed(texts, batch_size=batch_size)]
