# scripts/compare_embeddings.py
"""
Gate 1 validation: verify fastembed and sentence-transformers produce
compatible vectors for the same model, and that retrieval quality holds.

Run: uv run python scripts/compare_embeddings.py
Requires: DATABASE_URL set to Neon direct DSN, both libraries installed locally.
"""

import math
import os

HEADLINES = [
    "Earthquake strikes Turkey, thousands displaced",
    "Federal Reserve raises interest rates again",
    "Apple unveils new iPhone with AI features",
    "Ukraine ceasefire talks stall in Brussels",
    "Amazon warehouse workers vote to unionize",
    "Scientists discover potential cancer vaccine",
    "Hurricane Milton makes landfall in Florida",
    "Tesla recalls 200,000 vehicles over brake defect",
    "Israel expands ground offensive in Gaza",
    "China launches lunar sample return mission",
    "US Senate passes $60 billion Ukraine aid bill",
    "OpenAI releases GPT-5 with extended context",
    "WHO declares mpox a global health emergency",
    "Boeing 737 Max cleared to fly again in Europe",
    "EU imposes tariffs on Chinese electric vehicles",
    "Wildfire destroys 10,000 acres in California",
    "India overtakes China as world's most populous nation",
    "SpaceX Starship completes first full flight test",
    "UK economy falls into recession for second quarter",
    "Inflation in Argentina hits 200 percent annually",
]

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PASS_THRESHOLD = 0.9999


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def embed_st(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_fe(texts: list[str]) -> list[list[float]]:
    from fastembed import TextEmbedding
    model = TextEmbedding(MODEL)
    return [v.tolist() for v in model.embed(texts)]


def check_vector_compatibility() -> bool:
    print("=== Gate 1a: Vector compatibility ===")
    st_vecs = embed_st(HEADLINES)
    fe_vecs = embed_fe(HEADLINES)

    sims = [cosine_similarity(st, fe) for st, fe in zip(st_vecs, fe_vecs, strict=True)]
    mean_sim = sum(sims) / len(sims)
    min_sim = min(sims)

    print(f"Mean cosine similarity: {mean_sim:.6f}")
    print(f"Min cosine similarity:  {min_sim:.6f}")
    print(f"Threshold:              {PASS_THRESHOLD}")

    passed = min_sim >= PASS_THRESHOLD
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def check_retrieval_quality() -> None:
    print("\n=== Gate 1b: Retrieval quality (manual inspection) ===")
    import psycopg
    from fastembed import TextEmbedding
    from psycopg.rows import dict_row

    db_url = os.environ["DATABASE_URL"]
    model = TextEmbedding(MODEL)

    test_queries = [
        "military conflict Middle East",
        "economic recession inflation",
        "technology artificial intelligence",
        "natural disaster climate",
        "election politics government",
    ]

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        for query in test_queries:
            vec = list(model.embed([query]))[0].tolist()
            rows = conn.execute(
                """
                SELECT title, domain,
                       embedding <=> %s::vector AS distance
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 5
                """,
                (vec, vec),
            ).fetchall()

            print(f"\nQuery: {query!r}")
            for i, r in enumerate(rows, 1):
                print(f"  {i}. [{r['domain']}] {r['title']} (dist={r['distance']:.4f})")

    print("\nInspect results above. Proceed only if results are semantically relevant.")


if __name__ == "__main__":
    compatible = check_vector_compatibility()
    if compatible:
        check_retrieval_quality()
    else:
        print("\nVector compatibility FAILED. Do not enable /api/search. Keep 501.")
