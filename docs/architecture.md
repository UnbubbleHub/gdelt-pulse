# Architecture

GDELT Pulse is split into two independent deployments sharing a single PostgreSQL database:

- **Pipeline** -- A long-running Docker container (Railway) that ingests, embeds, and clusters articles
- **API** -- A FastAPI application (Vercel) serving search, browse, and auth endpoints

```
                    ┌──────────────────────────────────────────────┐
                    │            Railway (Pipeline)                 │
                    │                                              │
                    │   ┌──────────┐  ┌─────────┐  ┌───────────┐  │
  GDELT GKG ──────►│   │ Ingest   ├─►│  Embed  ├─►│  Cluster  │  │
  (HTTP, 15min)     │   └──────────┘  └─────────┘  └───────────┘  │
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                   ┌───────▼────────┐
                                   │  PostgreSQL    │
                                   │  + pgvector    │
                                   │  (Neon)        │
                                   └───────┬────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │            Vercel (API)                       │
                    │                                              │
                    │   FastAPI ── routers/search.py               │
                    │              routers/clusters.py              │
                    │              routers/articles.py              │
                    │              routers/keys.py                  │
                    │                                              │
                    │   middleware.py ── rate limiting              │
                    │   auth.py ── Clerk JWT verification           │
                    └──────────────────────────────────────────────┘
```

## Pipeline Stages

### 1. Ingestion

**Module:** `gdelt_event_pipeline.ingestion`

Fetches the latest GDELT Global Knowledge Graph (GKG) export every 15 minutes. Each GKG file contains metadata about news articles: URLs, themes, entities (persons, organizations, locations), tone scores, and source information.

The ingestion stage:

1. Downloads and parses the latest GKG ZIP file
2. Normalizes URLs (strips tracking parameters, resolves redirects)
3. Maps source domains to canonical source names
4. Parses structured fields (themes, entities, locations, tone) from GKG's delimited format
5. Deduplicates by canonical URL
6. Upserts articles into PostgreSQL in batches
7. Optionally scrapes titles for articles that don't have one

**State tracking:** The `pipeline_state` table records the last processed GDELT timestamp, so subsequent runs only fetch new data.

### 2. Embedding

**Module:** `gdelt_event_pipeline.embeddings`

Generates 384-dimensional vectors for articles using [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, all-MiniLM-L6-v2).

For each article without an embedding:

1. Composes a text representation from title + themes + entities + source
2. Generates the embedding vector
3. Stores it in the `embedding` column (pgvector `vector(384)` type)

Processes articles in batches. Articles without titles are skipped since titles carry the most semantic weight.

### 3. Clustering

**Module:** `gdelt_event_pipeline.clustering`

Groups related articles into event clusters. Each cluster represents a real-world story (e.g., "EU passes AI regulation", "Earthquake in Turkey").

For each unclustered article:

1. **Candidate selection** -- Finds clusters active within a configurable temporal window (default: 72 hours)
2. **Two-stage scoring:**
   - **Cosine similarity** between the article embedding and each cluster's centroid
   - **Entity overlap** -- Weighted Jaccard similarity across locations, persons, and organizations
3. **Assignment** -- If the best candidate exceeds the similarity threshold (default: 0.75), the article joins that cluster. Otherwise, a new cluster is created.
4. **Centroid update** -- The cluster's centroid is recomputed as the mean of all member embeddings

The temporal window prevents old clusters from absorbing unrelated new articles on the same broad topic.

### 4. Continuous Runner

**Module:** `gdelt_event_pipeline.runner`

Orchestrates all three stages in a loop:

```
while running:
    ingest()          # fetch latest GKG, upsert articles
    scrape_titles()   # fill missing titles via HTTP
    embed()           # generate vectors for new articles
    cluster()         # assign articles to event clusters
    cleanup()         # remove articles that failed title scraping
    sleep(interval)   # default: 900s (15 min)
```

Handles `SIGTERM`/`SIGINT` for graceful shutdown (Railway sends `SIGTERM` on deploy). Includes exponential backoff on failures and per-cycle logging.

## API Layer

**Module:** `gdelt_event_pipeline.api`

The API is a FastAPI application with a modular router structure:

| File | Endpoints | Purpose |
|------|-----------|---------|
| `routers/search.py` | `GET /api/search` | Hybrid search with configurable semantic/keyword balance |
| `routers/clusters.py` | `GET /api/clusters`, `GET /api/clusters/{id}` | Browse and inspect event clusters |
| `routers/articles.py` | `GET /api/articles`, `GET /api/stats` | Browse articles with filters, pipeline stats |
| `routers/keys.py` | `GET/POST/DELETE /api/auth/keys`, `GET /api/auth/config` | API key CRUD (Clerk-gated) |

Supporting modules:

| File | Purpose |
|------|---------|
| `app.py` | FastAPI app initialization, lifespan (pool + schema), CORS, router mounts (~100 lines) |
| `middleware.py` | Per-request rate limiting (IP-based for public, key-based for authenticated) |
| `auth.py` | Clerk JWT verification dependency |

### Rate Limiting

| Client Type | Limit |
|-------------|-------|
| Unauthenticated (by IP) | 30 requests/minute |
| Authenticated (by API key) | 200 requests/minute |

Rate limiting uses Upstash Redis in production (Vercel-compatible HTTP-based Redis) with an in-memory fallback for local development.

## Storage Layer

**Module:** `gdelt_event_pipeline.storage`

All database operations are centralized in the storage layer:

| File | Purpose |
|------|---------|
| `database.py` | Connection pool lifecycle (init, get, close) using psycopg 3 |
| `articles.py` | Article CRUD: single/batch upsert, getters, title/embedding updates |
| `clusters.py` | Cluster CRUD: create, assign, find nearest, entity sample updates |
| `pipeline_state.py` | Get/update the last-processed GDELT timestamp |
| `migrations.py` | Applies SQL schema files on startup (resolves paths for Vercel/Docker/local) |

## Query Layer

**Module:** `gdelt_event_pipeline.query`

Implements hybrid search combining two strategies:

1. **Semantic search** (`vector.py`) -- Embeds the query and finds nearest neighbors via pgvector's HNSW index
2. **Keyword search** (`keyword.py`) -- PostgreSQL full-text search with `websearch_to_tsquery`

Results are merged via **Reciprocal Rank Fusion** (`ranking.py`) with a configurable semantic weight (0.0 = pure keyword, 1.0 = pure semantic, default 0.5).

Filters (`filters.py`) support location, person, organization, theme, domain, source, and date range -- all applied as SQL conditions before search.

See [Search](search.md) for a detailed explanation.

## Database Schema

Four core tables:

```sql
articles          -- Ingested articles with metadata and embeddings
clusters          -- Event clusters with centroids and summaries
cluster_memberships  -- Article ↔ cluster links with similarity scores
pipeline_state    -- Ingestion checkpoint (last GDELT timestamp)
api_keys          -- Per-user API keys (hashed, one active per user)
```

See [Database Design](database_design.md) for the full schema with column definitions and index documentation.

## Design Decisions

**Why split API and pipeline?** The pipeline needs long-running processes and CPU for embedding generation. The API needs fast cold starts for serverless. Splitting them lets each run on infrastructure suited to its workload.

**Why fastembed over sentence-transformers?** fastembed uses ONNX Runtime, which works within Vercel's memory and startup constraints. sentence-transformers requires PyTorch, which exceeds Vercel's 250MB function size limit.

**Why RRF over learned reranking?** RRF is parameter-free (no training data needed), fast (just rank arithmetic), and works surprisingly well for combining heterogeneous retrieval signals. It's the right starting point before investing in a learned reranker.

**Why psycopg 3 over SQLAlchemy?** The query patterns here are mostly hand-tuned SQL for pgvector operations and full-text search. An ORM would add abstraction without reducing complexity. psycopg 3's connection pooling and `dict_row` factory give us what we need.
