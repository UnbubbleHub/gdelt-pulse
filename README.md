# gdelt-pulse

A backend pipeline for ingesting GDELT-derived news records, clustering them into evolving event entities, and making those events queryable — with an analytics layer for exploring narrative, geography, and source dynamics.

Developed as part of the [UnbubbleHub](https://github.com/UnbubbleHub) open-source ecosystem.

---

## Table of Contents

- [Project Goal](#project-goal)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Database Setup](#database-setup)
  - [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Ingestion](#1-ingestion)
  - [2. Embedding](#2-embedding)
  - [3. Clustering](#3-clustering)
  - [Full Pipeline Run](#full-pipeline-run)
  - [Continuous Runner](#continuous-runner)
- [Deployment](#deployment)
  - [API — Vercel](#api--vercel)
  - [Pipeline — Railway](#pipeline--railway)
- [API & Dashboard](#api--dashboard)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
  - [Hybrid Search](#hybrid-search)
  - [Analytics Views](#analytics-views)
  - [Auth & API Keys](#auth--api-keys)
  - [Dashboard Pages](#dashboard-pages)
- [Utility Scripts](#utility-scripts)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Project Goal

Transform the continuous GDELT news stream into a structured event index with analytics.

GDELT provides structured metadata about global news coverage, but does not directly provide semantic news story clusters. This project builds an incremental event-construction layer on top of GDELT: each **event** represents a group of related articles describing the same real-world development. On top of that index, a set of analytics views expose narrative polarization, geographic attention, source fingerprints, story propagation, and topic velocity.

```
GDELT records  ->  normalized articles  ->  clustered events  ->  hybrid search API  ->  analytics dashboards
```

---

## How It Works

The system has three pipeline stages, an API/visualization layer, and a continuous runner that ties them together:

1. **Ingestion** — Fetches the latest GDELT GKG (Global Knowledge Graph) data, normalizes article metadata (URLs, sources, themes, entities, tone), deduplicates by canonical URL, and stores articles in PostgreSQL.

2. **Embedding** — Takes articles that don't yet have an embedding, composes a text representation from title + metadata, and generates a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (or FastEmbed on Vercel). Vectors are stored in the database via pgvector.

3. **Clustering** — Takes articles with embeddings but no cluster assignment, and matches them to existing event clusters using a two-stage scoring system:
   - **Cosine similarity** between article embedding and cluster centroids
   - **Entity overlap** (locations, persons, organizations) weighted by importance
   - If no cluster is similar enough, a new event cluster is created

Each pipeline stage is incremental — it picks up where it left off, so the pipeline can run continuously or on a schedule.

4. **Continuous Runner** — Orchestrates all three stages in a loop, matching the 15-minute GDELT update cadence. Handles title scraping, cleanup of unresolvable articles, graceful shutdown, and per-cycle logging. Deployed as a long-running Railway service.

5. **API & Analytics** — A FastAPI server exposes REST endpoints for browsing articles and clusters, plus a hybrid search that combines semantic vector similarity (pgvector HNSW) with PostgreSQL full-text search, merged via Reciprocal Rank Fusion (RRF). Multiple interactive analytics views (globe, polarization, gravity map, source DNA, propagation, velocity) provide different lenses on the data. API access is gated by Clerk authentication and per-user API keys with rate limiting.

---

## Architecture

```
┌─────────────────┐
│   GDELT GKG     │
│   (HTTP fetch)  │
└────────┬────────┘
         │
         v
┌──────────────────────────┐
│  1. INGESTION            │  ┐
│  Fetch GKG ZIP > parse > │  │
│  normalize > dedupe >    │  │
│  upsert > checkpoint     │  │  Continuous runner
└─────────┬────────────────┘  │  (Railway — Dockerfile.pipeline)
          │                   │  15-minute cadence
          v                   │
┌──────────────────────────┐  │
│  2. EMBEDDING            │  │
│  Fetch unembedded >      │  │
│  compose text > embed >  │  │
│  store vectors           │  │
└─────────┬────────────────┘  │
          │                   │
          v                   │
┌──────────────────────────┐  │
│  3. CLUSTERING           │  │
│  Fetch unclustered >     │  │
│  score candidates >      │  │
│  assign or create >      │  │
│  update centroids        │  ┘
└─────────┬────────────────┘
          │
          v  PostgreSQL + pgvector (shared)
          │
┌──────────────────────────────────────────────────┐
│  4. API & ANALYTICS (FastAPI — Vercel)            │
│  /api/search       — hybrid search               │
│  /api/articles     — browse articles             │
│  /api/clusters     — browse events               │
│  /api/stats        — dashboard stats             │
│  /api/globe/*      — live/rising/silent events   │
│  /api/polarization — narrative tone analysis     │
│  /api/gravity/*    — country co-mention graph    │
│  /api/asymmetry    — attention asymmetry         │
│  /api/sources/*    — source fingerprints         │
│  /api/propagation/* — story spread timeline      │
│  /api/velocity/*   — topic trend detection       │
│  /api/auth/keys    — API key management (Clerk)  │
│  /                 — SPA + analytics views       │
└──────────────────────────────────────────────────┘
```

**State tracking:** The `pipeline_state` table tracks the last processed GDELT timestamp, enabling incremental re-runs without reprocessing.

**Auth:** Dashboard and API key endpoints are gated by [Clerk](https://clerk.com) JWT. Public API endpoints accept an optional `X-API-Key` header for higher rate limits.

---

## Repository Structure

```
gdelt-pulse/
├── api/
│   └── index.py            # Vercel ASGI entry point (re-exports FastAPI app)
├── src/gdelt_event_pipeline/
│   ├── ingestion/          # GDELT fetching, GKG parsing, title scraping
│   │   ├── gkg_fetcher.py
│   │   ├── pipeline.py
│   │   ├── run.py
│   │   └── scraper.py
│   ├── normalization/      # URL canonicalization, source mapping, GKG field parsing
│   ├── storage/            # PostgreSQL operations (articles, clusters, pipeline state)
│   ├── embeddings/         # Vector embedding generation (sentence-transformers / fastembed)
│   ├── clustering/         # Event clustering with entity-aware scoring
│   ├── config/             # Settings loaded from environment variables
│   ├── query/              # Hybrid search layer (vector, keyword, RRF ranking, filters)
│   ├── api/                # FastAPI server + static analytics SPA
│   │   ├── app.py          # All API endpoints, rate limiting, auth middleware
│   │   ├── auth.py         # Clerk JWT verification dependency
│   │   ├── keys.py         # API key CRUD endpoints (/api/auth/keys)
│   │   └── static/         # Multi-page analytics frontend (vanilla JS)
│   │       ├── index.html      # Home / search dashboard
│   │       ├── globe.html      # 3D NewsGlobe
│   │       ├── polarization.html # Narrative Polarization
│   │       ├── gravity.html    # Geopolitical Gravity Map
│   │       ├── asymmetry.html  # Attention Asymmetry
│   │       ├── sources.html    # Source DNA
│   │       ├── propagation.html # Story Propagation
│   │       ├── velocity.html   # Topic Velocity
│   │       ├── dashboard.html  # API key dashboard (Clerk-gated)
│   │       ├── developers.html # Developer reference
│   │       └── pulse.css       # Shared stylesheet
│   ├── runner.py           # Continuous pipeline orchestrator (used by Railway service)
│   └── utils/              # Shared utilities
├── tests/                  # Mirror structure of src/ with pytest tests
│   ├── api/                # Tests for app.py, auth.py, keys.py
│   ├── clustering/
│   ├── embeddings/
│   ├── ingestion/
│   ├── normalization/
│   ├── query/
│   └── storage/
├── scripts/
│   ├── browse_articles.py  # CLI tool to inspect stored articles
│   └── compare_embeddings.py # Compare embedding similarity scores
├── sql/
│   ├── 001_schema.sql      # Core schema (articles, clusters, memberships, pipeline_state)
│   ├── 002_api_keys.sql    # API key management table
│   └── 002_gravity_views.sql # Materialized views for the gravity map
├── docs/
│   └── database_design.md  # Schema design decisions and field documentation
├── Dockerfile              # API container image
├── Dockerfile.pipeline     # Pipeline runner container image (Railway)
├── docker-compose.yml      # Local dev: API + pipeline + Postgres
├── railway.toml            # Railway deployment config (pipeline runner)
├── vercel.json             # Vercel deployment config (API)
├── pyproject.toml          # Project metadata, dependencies, tool config
└── .env.example            # Template for environment variables
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** with the [pgvector](https://github.com/pgvector/pgvector) extension
- **[uv](https://docs.astral.sh/uv/)** (recommended Python package manager)

#### Installing pgvector

pgvector adds vector similarity search to PostgreSQL. Install it for your platform:

```bash
# macOS (Homebrew)
brew install pgvector

# Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# From source (any platform)
# See https://github.com/pgvector/pgvector#installation
```

### Installation

```bash
# Clone the repository
git clone https://github.com/UnbubbleHub/gdelt-pulse.git
cd gdelt-pulse

# Install dependencies (including dev tools)
uv sync

# Copy the environment template
cp .env.example .env
```

### Windows users

If you're using Windows Command Prompt, you must define the PostgreSQL user directly from the terminal (CMD).

For the current terminal session only:
```cmd
set PGUSER=postgres
```

To make it persistent for future terminals:
```cmd
setx PGUSER postgres
```

### Database Setup

1. **Create the database:**

```bash
createdb gdelt_pulse
```

2. **Enable required extensions and create the schema:**

```bash
psql -d gdelt_pulse -f sql/001_schema.sql
psql -d gdelt_pulse -f sql/002_api_keys.sql
psql -d gdelt_pulse -f sql/002_gravity_views.sql
```

`001_schema.sql` creates four core tables (`articles`, `clusters`, `cluster_memberships`, `pipeline_state`) with all required indexes, including HNSW indexes for vector similarity search. `002_api_keys.sql` adds the per-user API key table. `002_gravity_views.sql` creates materialized views used by the gravity map. See [docs/database_design.md](docs/database_design.md) for full schema documentation.

> **Note:** The API server auto-applies `001_schema.sql` and `002_api_keys.sql` on first start if the tables don't exist, so you only need to run them manually for local development.

3. **Verify the setup:**

```bash
psql -d gdelt_pulse -c "\dt"
```

You should see all tables listed.

### Configuration

Edit `.env` with your PostgreSQL credentials:

```bash
# PostgreSQL connection
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=your_password
PGDATABASE=gdelt_pulse

# Or use a single connection URL (takes precedence)
# DATABASE_URL=postgresql://user:pass@host:5432/gdelt_pulse
```

Optional embedding settings (defaults work out of the box):

```bash
# Embedding backend: sentence-transformers (local) or fastembed (Vercel-compatible)
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=64
```

Auth and rate limiting (required for API key features):

```bash
# Clerk authentication (https://clerk.com)
CLERK_JWKS_URL=https://<your-clerk-domain>/.well-known/jwks.json
CLERK_PUBLISHABLE_KEY=pk_test_...

# Upstash Redis for distributed rate limiting (optional — falls back to in-memory)
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
```

> **Note:** The first time you run the embedding stage with `sentence-transformers`, the model (~80 MB) will be downloaded automatically.

---

## Running the Pipeline

Each stage is a standalone Python module. Run them in order from the project root.

### 1. Ingestion

Fetches the latest GDELT GKG data, normalizes articles, and stores them.

```bash
# Fetch the latest GKG file
uv run python -m gdelt_event_pipeline.ingestion

# Fetch a specific GKG file by URL
uv run python -m gdelt_event_pipeline.ingestion --url <gkg_file_url>

# Preview without writing to the database
uv run python -m gdelt_event_pipeline.ingestion --dry-run

# Also scrape titles for articles missing them
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles

# Scrape titles only (no new ingestion)
uv run python -m gdelt_event_pipeline.ingestion --scrape-only

# Verbose output
uv run python -m gdelt_event_pipeline.ingestion -v
```

### 2. Embedding

Generates vector embeddings for articles that don't have one yet.

```bash
uv run python -m gdelt_event_pipeline.embeddings
```

Processes up to 500 articles per run. Articles without a title are skipped.

### 3. Clustering

Assigns embedded articles to event clusters.

```bash
# Run with default settings (threshold=0.75, 72h temporal window)
uv run python -m gdelt_event_pipeline.clustering

# Adjust similarity threshold (lower = more permissive matching)
uv run python -m gdelt_event_pipeline.clustering --threshold 0.70

# Custom temporal window (only match clusters active within N hours)
uv run python -m gdelt_event_pipeline.clustering --window 48

# Disable temporal window (match all active clusters)
uv run python -m gdelt_event_pipeline.clustering --window 0

# Process more articles per run
uv run python -m gdelt_event_pipeline.clustering --limit 1000

# Verbose output (shows per-article assignment details)
uv run python -m gdelt_event_pipeline.clustering -v
```

By default, only clusters that received an article within the last 72 hours are considered as candidates. This prevents old clusters from absorbing unrelated new articles on the same topic. The window is configurable via `--window` or the `CLUSTER_WINDOW_HOURS` environment variable.

### Full Pipeline Run

Run all three stages in sequence:

```bash
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles && \
uv run python -m gdelt_event_pipeline.embeddings && \
uv run python -m gdelt_event_pipeline.clustering -v
```

### Continuous Runner

For production use, the runner module executes all pipeline stages in a loop, sleeping between cycles to match the 15-minute GDELT update cadence:

```bash
uv run python -m gdelt_event_pipeline.runner

# Override the cycle interval (seconds)
PIPELINE_INTERVAL=600 uv run python -m gdelt_event_pipeline.runner
```

The runner handles graceful shutdown on `SIGTERM`/`SIGINT`, automatically initializes the schema on fresh databases, and cleans up articles that failed title scraping after one attempt.

---

## Deployment

The project is split into two independent deployments that share the same PostgreSQL database.

### API — Vercel

The FastAPI application is deployed to Vercel via `api/index.py`. `vercel.json` routes all traffic through this entry point and bundles `src/` and `sql/` into the function.

```bash
vercel deploy --prod
```

Set the following environment variables in the Vercel dashboard:
- `DATABASE_URL` — connection string to your hosted Postgres instance
- `EMBEDDING_BACKEND=fastembed` — lighter embedding backend compatible with Vercel's runtime
- `CLERK_JWKS_URL`, `CLERK_PUBLISHABLE_KEY` — Clerk auth
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` — distributed rate limiting

### Pipeline — Railway

The continuous runner is deployed to Railway using `Dockerfile.pipeline`. Railway's Postgres plugin variables (`DATABASE_URL` etc.) are linked to the service automatically.

`railway.toml` configures the build to use `Dockerfile.pipeline` and sets `restartPolicyType = "always"` so the runner restarts on failure.

---

## API & Dashboard

Once the pipeline has populated the database, you can explore the data through the API and dashboard.

### Running the API

```bash
# Start the API server (development mode with auto-reload)
uv run python -m gdelt_event_pipeline.api.app

# Or via uvicorn directly
uv run uvicorn gdelt_event_pipeline.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) to access the dashboard.
Interactive API docs are available at [http://localhost:8000/api/docs](http://localhost:8000/api/docs).

### API Endpoints

**Core**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | Dashboard statistics (total articles, clusters, largest cluster, etc.) |
| `GET` | `/api/articles?limit=50` | Recent articles with titles (max 200) |
| `GET` | `/api/clusters?limit=100&sort=recent` | Active clusters, sortable by `recent`, `articles`, or `oldest` |
| `GET` | `/api/clusters/{id}` | Cluster detail with member articles and similarity scores |
| `GET` | `/api/search?q=...` | Hybrid semantic + keyword search |

**Analytics**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/globe/clusters` | Top clusters with coordinates for the 3D globe (modes: `live`, `rising`, `silent`) |
| `GET` | `/api/polarization` | Most narratively polarized story clusters (tone stddev across sources) |
| `GET` | `/api/polarization/{id}` | Per-source tone breakdown for a single cluster |
| `GET` | `/api/gravity/graph` | Country co-mention graph (nodes + weighted edges) |
| `GET` | `/api/gravity/country/{code}` | Detail view for a single country: connections, clusters, tone |
| `GET` | `/api/asymmetry` | Attention asymmetry: coverage volume vs crisis intensity by country |
| `GET` | `/api/sources/fingerprints` | Per-source fingerprints: article count, tone, top themes, top countries |
| `GET` | `/api/sources/{domain}/detail` | Full fingerprint for a single source domain |
| `GET` | `/api/propagation/stories` | Stories suitable for propagation analysis (multi-source clusters) |
| `GET` | `/api/propagation/{id}` | Chronological timeline of how a story spread across sources |
| `GET` | `/api/velocity/topics` | Rising and declining GDELT themes over a configurable time window |
| `GET` | `/api/velocity/timeline` | Hourly article count for a specific theme |

**Auth & Keys** (Clerk JWT required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/auth/keys` | Check whether the authenticated user has an active API key |
| `POST` | `/api/auth/keys` | Create (or rotate) an API key for the authenticated user |
| `DELETE` | `/api/auth/keys` | Revoke the user's active API key |
| `GET` | `/api/auth/config` | Return public Clerk configuration for the frontend |

### Hybrid Search

The `/api/search` endpoint combines two search strategies and merges results using **Reciprocal Rank Fusion (RRF)**:

1. **Semantic search** — Embeds the query with the same model (all-MiniLM-L6-v2 / fastembed) and finds nearest neighbors via pgvector HNSW index
2. **Keyword search** — Uses PostgreSQL full-text search (`websearch_to_tsquery`) on article titles

**Search parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | *(required)* | Search query text |
| `limit` | 20 | Max results (1–100) |
| `semantic_weight` | 0.5 | Balance between semantic (1.0) and keyword (0.0) search |
| `clusters` | false | Also search cluster centroids |
| `location` | — | Filter by location names (comma-separated) |
| `person` | — | Filter by person names |
| `org` | — | Filter by organizations |
| `theme` | — | Filter by GDELT theme codes |
| `domain` | — | Filter by source domain |
| `source` | — | Filter by source name |
| `date_from` | — | Start date (ISO format) |
| `date_to` | — | End date (ISO format) |

Example:

```bash
curl "http://localhost:8000/api/search?q=earthquake+Turkey&semantic_weight=0.7&location=Turkey&limit=10"
```

Public endpoints are rate-limited to **30 requests/minute** per IP. Requests authenticated with an `X-API-Key` header are allowed **200 requests/minute** per key.

### Analytics Views

Each analytics view is a standalone page backed by its own API namespace:

| Page | URL | What it shows |
|------|-----|---------------|
| **NewsGlobe** | `/globe` | 3D globe with live/rising/silent event clusters pinned to their primary location |
| **Narrative Polarization** | `/polarization` | Story clusters ranked by tone variance across sources |
| **Geopolitical Gravity Map** | `/gravity` | Force-directed graph of country co-mention relationships |
| **Attention Asymmetry** | `/asymmetry` | World map comparing coverage volume to crisis-theme intensity by country |
| **Source DNA** | `/sources` | Per-source fingerprints: tone, top themes, top countries |
| **Story Propagation** | `/propagation` | Chronological timeline of how a multi-source story spread |
| **Topic Velocity** | `/velocity` | Rising and declining GDELT themes with configurable lookback windows |

### Auth & API Keys

Authentication is powered by [Clerk](https://clerk.com). Dashboard and API key management pages require a Clerk session in the browser.

API keys (prefixed `gdp_`) are stored hashed (SHA-256) — the full key is only returned at creation time. Each user can have one active key at a time; creating a new key automatically revokes the previous one.

### Dashboard Pages

| Page | URL | Notes |
|------|-----|-------|
| Home / Search | `/` | Hybrid search with weight slider and filter panel |
| API Dashboard | `/dashboard` | Clerk-gated — generate and revoke API keys |
| Developer Reference | `/developers` | API documentation and quickstart |

---

## Utility Scripts

**Browse articles** — Display stored articles with metadata:

```bash
uv run python scripts/browse_articles.py
```

**Compare embeddings** — Compare cosine similarity scores between articles:

```bash
uv run python scripts/compare_embeddings.py
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Run tests for a specific module
uv run pytest tests/normalization/
uv run pytest tests/clustering/
uv run pytest tests/api/

# Run with verbose output
uv run pytest -v
```

Lint and format checks:

```bash
uv run ruff check .
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Database | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) |
| Search | Hybrid: pgvector HNSW (semantic) + PostgreSQL full-text search (keyword) + RRF |
| API | [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) |
| Frontend | Multi-page analytics app (vanilla JavaScript, no framework) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2, 384-dim) / [fastembed](https://github.com/qdrant/fastembed) on Vercel |
| Auth | [Clerk](https://clerk.com) (JWT verification via PyJWT) |
| Rate limiting | In-memory fallback + [Upstash Redis](https://upstash.com/) for distributed deployments |
| DB driver | [psycopg 3](https://www.psycopg.org/psycopg3/) + connection pooling |
| API deployment | [Vercel](https://vercel.com/) (Python runtime) |
| Pipeline deployment | [Railway](https://railway.app/) (Docker) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Testing | pytest |
| Linting | ruff |

---

## Roadmap

Phase 1 — reliable pipeline + search API:

- [x] Incremental GDELT ingestion with checkpointing
- [x] Article normalization and deduplication
- [x] Vector embedding generation
- [x] Entity-aware event clustering
- [x] Hybrid search layer (semantic + keyword + RRF)
- [x] REST API with FastAPI
- [x] Interactive search dashboard (SPA)
- [x] Continuous pipeline runner (Railway)

Phase 2 — analytics layer + developer platform:

- [x] 3D NewsGlobe (live/rising/silent modes)
- [x] Narrative Polarization view
- [x] Geopolitical Gravity Map (country co-mention graph)
- [x] Attention Asymmetry view
- [x] Source DNA fingerprints
- [x] Story Propagation timeline
- [x] Topic Velocity tracking
- [x] Clerk authentication + API key management
- [x] Rate limiting (per-IP and per-key)
- [x] Vercel + Railway split deployment

Later phases may add:

- Event analytics and trend detection improvements
- Coverage and source diversity scoring
- Narrative comparison across sources

See [open issues](https://github.com/UnbubbleHub/gdelt-pulse/issues) for specific improvements being tracked.

---

## Contributing

We welcome contributions! This is an open-source project and there are many ways to help.

**Getting involved:**

- **Pick up an issue** — Check [open issues](https://github.com/UnbubbleHub/gdelt-pulse/issues) 
- **Suggest improvements** — Have ideas for better clustering algorithms, new features, or architectural changes? Open a [Discussion](https://github.com/UnbubbleHub/gdelt-pulse/discussions) to talk it through
- **Report bugs** — Found something broken? Open an issue with steps to reproduce

**Workflow:**

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check .`)
5. Open a pull request against `main`

**Code style:**

- Ruff is configured in `pyproject.toml` (line length 100, Python 3.11 target)
- Run `uv run ruff check .` before committing
- Write tests for new functionality — test files mirror the `src/` structure under `tests/`

---

## License

This project is part of the [UnbubbleHub](https://github.com/UnbubbleHub) ecosystem.
