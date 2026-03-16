"""Tests for embedding generation."""

from gdelt_event_pipeline.embeddings.embed import embed_texts

EMBEDDING_DIM = 384
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TestEmbedTexts:
    def test_empty_list(self):
        assert embed_texts([]) == []

    def test_single_text(self):
        result = embed_texts(["Hello world"], model_name=MODEL)
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in result[0])

    def test_multiple_texts(self):
        texts = ["Earthquake in Turkey", "Stock market crashes", "New iPhone released"]
        result = embed_texts(texts, model_name=MODEL)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == EMBEDDING_DIM

    def test_similar_texts_closer_than_unrelated(self):
        texts = [
            "Earthquake hits Turkey, thousands displaced",
            "Seismic disaster in Turkey leaves many homeless",
            "Apple announces new iPhone pricing strategy",
        ]
        vectors = embed_texts(texts, model_name=MODEL)

        sim_related = _cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = _cosine_similarity(vectors[0], vectors[2])
        assert sim_related > sim_unrelated


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)
