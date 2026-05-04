<p align="center">
  <img src="docs/assets/unbubblehub-logo.png" alt="UnbubbleHub" width="80">
</p>

<h1 align="center">gdelt-pulse</h1>

<p align="center">
  <strong>Real-time global news intelligence API powered by GDELT</strong>
</p>

<p align="center">
  <a href="https://github.com/UnbubbleHub/gdelt-pulse/stargazers"><img src="https://img.shields.io/github/stars/UnbubbleHub/gdelt-pulse?style=flat" alt="Stars"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776ab.svg?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/fastapi-0.115+-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="docs/api-reference.md">API Reference</a> &middot;
  <a href="docs/architecture.md">Architecture</a> &middot;
  <a href="docs/deployment.md">Deploy</a> &middot;
  <a href="docs/search.md">Search</a> &middot;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  Part of the <a href="https://github.com/UnbubbleHub">UnbubbleHub</a> open-source ecosystem
</p>

---

An API that continuously ingests the [GDELT](https://www.gdeltproject.org/) global news feed, clusters articles into real-world event stories, and exposes them through a hybrid semantic + keyword search engine. Built for researchers, journalists, and developers who need structured, queryable access to global news events as they happen.

```
GDELT stream  -->  normalize  -->  embed  -->  cluster  -->  search API
   (15 min)        (dedupe)      (384-dim)    (entity-     (vector +
                                               aware)      keyword + RRF)
```

## Features

- **Hybrid search** -- Combines pgvector semantic similarity with PostgreSQL full-text search, merged via Reciprocal Rank Fusion (RRF)
- **Event clustering** -- Groups related articles into story clusters using cosine similarity + entity overlap scoring
- **Rich metadata** -- Every article carries locations, persons, organizations, themes, tone scores, and source attribution from GDELT's Global Knowledge Graph
- **Incremental pipeline** -- Ingestion, embedding, and clustering stages run continuously, picking up where they left off every 15 minutes
- **Developer-friendly** -- OpenAPI docs, per-user API keys, rate limiting, and a clean REST interface

## Quick Start

**Prerequisites:** Python 3.11+, PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector), [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/UnbubbleHub/gdelt-pulse.git
cd gdelt-pulse
uv sync
cp .env.example .env   # edit with your Postgres credentials
```

```bash
# Start the API server
uv run --group api uvicorn gdelt_event_pipeline.api.app:app --reload

# Run the pipeline (ingest → embed → cluster)
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles
uv run python -m gdelt_event_pipeline.embeddings
uv run python -m gdelt_event_pipeline.clustering -v
```

Open `http://<your-host>:8000` for the landing page, or `http://<your-host>:8000/api/docs` for interactive API docs.

> Full setup instructions including database creation, pgvector installation, and configuration: **[Getting Started Guide](docs/getting-started.md)**

## API at a Glance

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=...` | Hybrid semantic + keyword search across articles and clusters |
| `GET /api/clusters` | Browse active event clusters, sorted by recency or size |
| `GET /api/clusters/{id}` | Cluster detail with member articles and similarity scores |
| `GET /api/articles` | Recent articles with filtering by location, person, org, theme, domain, source, date range |
| `GET /api/stats` | Pipeline statistics (article counts, cluster counts, embedding coverage) |

All endpoints accept an optional `X-API-Key` header for higher rate limits (200 req/min vs 30 req/min for unauthenticated).

```bash
# Search for articles about climate policy
curl "https://your-api.vercel.app/api/search?q=climate+policy&limit=10"

# Get the largest active clusters
curl "https://your-api.vercel.app/api/clusters?sort=articles&limit=5"

# Filter articles by location and theme
curl "https://your-api.vercel.app/api/articles?location=Ukraine&theme=MILITARY_CONFLICT"
```

> Full endpoint documentation with all parameters and response schemas: **[API Reference](docs/api-reference.md)**

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │          Railway (Pipeline)              │
                    │                                          │
                    │  ┌──────────┐  ┌─────────┐  ┌────────┐   │
  GDELT GKG ──────► │  │ Ingest   ├─►│  Embed  ├─►│Cluster │   │
  (HTTP, 15min)     │  └──────────┘  └─────────┘  └────────┘   │
                    └──────────────────────┬───────────────────┘
                                          │
                                  ┌───────▼───────┐
                                  │  PostgreSQL   │
                                  │  + pgvector   │
                                  │    (Neon)     │
                                  └───────┬───────┘
                                          │
                    ┌─────────────────────▼───────────────────┐
                    │          Vercel (API)                   │
                    │                                         │
                    │  FastAPI ── search / clusters / articles│
                    │  Clerk auth ── API key management       │
                    │  Rate limiting ── per-IP + per-key      │
                    └─────────────────────────────────────────┘
```

The project is split into two independent deployments sharing one database:

- **API** (Vercel) -- FastAPI serving search, browse, and auth endpoints as a serverless function
- **Pipeline** (Railway) -- Long-running Docker container that ingests, embeds, and clusters articles on a 15-minute cycle

> Deep dive into pipeline stages, data flow, and design decisions: **[Architecture Guide](docs/architecture.md)**

## Project Structure

```
gdelt-pulse/
├── api/index.py                    # Vercel ASGI entry point
├── src/gdelt_event_pipeline/
│   ├── api/                        # FastAPI application
│   │   ├── app.py                  #   Slim core (~100 lines)
│   │   ├── middleware.py           #   Rate limiting + API key validation
│   │   ├── auth.py                 #   Clerk JWT verification
│   │   └── routers/               #   Endpoint modules
│   │       ├── search.py           #     /api/search
│   │       ├── clusters.py         #     /api/clusters
│   │       ├── articles.py         #     /api/articles, /api/stats
│   │       └── keys.py            #     /api/auth/*
│   ├── ingestion/                  # GDELT GKG fetch + parsing
│   ├── normalization/              # URL, source, field normalization
│   ├── embeddings/                 # fastembed vector generation
│   ├── clustering/                 # Entity-aware event clustering
│   ├── query/                      # Hybrid search (vector + keyword + RRF)
│   ├── storage/                    # PostgreSQL operations
│   ├── config/                     # Environment-based settings
│   └── runner.py                   # Continuous pipeline orchestrator
├── sql/                            # Schema migrations
├── tests/                          # 353 tests mirroring src/ structure
└── docs/                           # Documentation
```

## Documentation

| Guide | Description |
|-------|-------------|
| **[Getting Started](docs/getting-started.md)** | Installation, database setup, configuration, first run |
| **[Architecture](docs/architecture.md)** | System design, pipeline stages, data flow, schema |
| **[API Reference](docs/api-reference.md)** | All endpoints, parameters, response formats, examples |
| **[Search](docs/search.md)** | How hybrid search works: vector, keyword, RRF, filters |
| **[Pipeline](docs/pipeline.md)** | Running each stage, continuous runner, tuning |
| **[Deployment](docs/deployment.md)** | Deploying to Vercel + Railway, environment variables |
| **[Database Design](docs/database_design.md)** | Schema design decisions, field documentation |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| API | [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) |
| Database | PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) |
| Search | pgvector HNSW (semantic) + `tsvector` (keyword) + RRF ranking |
| Embeddings | [fastembed](https://github.com/qdrant/fastembed) (all-MiniLM-L6-v2, 384-dim) |
| Auth | [Clerk](https://clerk.com) (JWT via PyJWT) |
| Rate Limiting | In-memory + [Upstash Redis](https://upstash.com/) |
| DB Driver | [psycopg 3](https://www.psycopg.org/psycopg3/) with connection pooling |
| API Hosting | [Vercel](https://vercel.com/) (Python runtime) |
| Pipeline Hosting | [Railway](https://railway.app/) (Docker) |
| Database Hosting | [Neon](https://neon.tech/) (serverless Postgres) |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Testing | pytest (353 tests) |
| Linting | [ruff](https://docs.astral.sh/ruff/) |
| CI | GitHub Actions |

## Contributing

We welcome contributions of all kinds. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines on setup, code style, testing, and the PR workflow.

```bash
# Run the test suite
uv run --group api pytest tests/ -q

# Lint and format check
ruff check . && ruff format . --check
```

## License

MIT -- see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/UnbubbleHub">UnbubbleHub</a></sub>
</p>
