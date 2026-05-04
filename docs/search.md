# Hybrid Search

GDELT Pulse implements a hybrid search system that combines semantic vector similarity with keyword matching, merged using Reciprocal Rank Fusion (RRF).

## How It Works

```
                           query: "earthquake Turkey"
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                    ┌─────▼─────┐         ┌─────▼─────┐
                    │  Semantic │         │  Keyword  │
                    │  Search   │         │  Search   │
                    │           │         │           │
                    │  Embed    │         │  Parse    │
                    │  query →  │         │  query →  │
                    │  HNSW     │         │  tsquery  │
                    │  nearest  │         │  + rank   │
                    │  neighbor │         │           │
                    └─────┬─────┘         └─────┬─────┘
                          │                     │
                          │   ranked lists      │
                          └──────────┬──────────┘
                                     │
                              ┌──────▼──────┐
                              │     RRF     │
                              │  (merge &   │
                              │   re-rank)  │
                              └──────┬──────┘
                                     │
                              final results
```

### 1. Semantic Search

The query is embedded into a 384-dimensional vector using the same model as article embeddings (all-MiniLM-L6-v2 via fastembed). The pgvector HNSW index finds the nearest neighbors by cosine distance.

This handles:
- Paraphrases ("climate change" matches "global warming")
- Concept similarity ("military conflict" matches "armed confrontation")
- Multilingual-ish queries (the model has some cross-lingual transfer)

### 2. Keyword Search

PostgreSQL's full-text search (`websearch_to_tsquery`) matches exact terms against article titles. This uses a GIN index on the `title_tsv` column.

This handles:
- Exact name matches ("Zelensky", "NATO")
- Rare terms that may not have strong semantic neighbors
- Precise acronyms and codes

### 3. Reciprocal Rank Fusion (RRF)

Both search methods return ranked lists. RRF merges them with the formula:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
```

where `k = 60` (standard constant) and the sum is over all ranking lists that include the document.

The `semantic_weight` parameter adjusts the contribution:

| Value | Behavior |
|-------|----------|
| `0.0` | Pure keyword search |
| `0.5` | Equal weight (default) |
| `1.0` | Pure semantic search |

In practice, 0.5--0.7 works well for most queries. Use higher semantic weight for conceptual queries ("countries with rising tensions") and lower for specific entity lookups ("UNHCR Syria report").

## Filters

Filters are applied as SQL WHERE conditions **before** search, reducing the candidate set:

| Filter | SQL Behavior |
|--------|-------------|
| `location=Turkey` | `locations::text ILIKE '%Turkey%'` |
| `person=Macron` | `persons::text ILIKE '%Macron%'` |
| `org=NATO` | `organizations::text ILIKE '%NATO%'` |
| `theme=MILITARY` | `themes::text ILIKE '%MILITARY%'` |
| `domain=bbc.com` | `domain ILIKE '%bbc.com%'` |
| `source=Reuters` | `source_common_name ILIKE '%Reuters%' OR canonical_source ILIKE '%Reuters%'` |
| `date_from=...` | `gdelt_timestamp >= <value>` |
| `date_to=...` | `gdelt_timestamp <= <value>` |

Filters use ILIKE for case-insensitive partial matching. Entity fields (locations, persons, organizations, themes) are stored as PostgreSQL arrays and cast to text for ILIKE matching.

## Cluster Search

When `clusters=true`, the search also queries cluster centroids. Matching clusters are returned separately in the response, ranked by centroid similarity to the query vector.

This is useful for finding events (story clusters) rather than individual articles.

## Performance

- **HNSW index**: pgvector's HNSW index provides approximate nearest neighbor search with sub-millisecond query times for the vector component
- **GIN index**: PostgreSQL's GIN index on `title_tsv` handles full-text queries efficiently
- **Connection pooling**: psycopg 3's connection pool minimizes connection overhead
- **Serverless tuning**: On Vercel, the pool is configured with `min_size=0, max_size=2` for fast cold starts

## Examples

```bash
# Broad conceptual search
curl "/api/search?q=climate+policy+negotiations&semantic_weight=0.8"

# Precise entity lookup
curl "/api/search?q=NATO+summit&semantic_weight=0.3"

# Filtered search
curl "/api/search?q=elections&location=France&date_from=2026-04-01"

# Search with cluster results
curl "/api/search?q=AI+regulation&clusters=true&limit=5"
```
