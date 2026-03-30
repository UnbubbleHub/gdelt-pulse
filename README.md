# gdelt-pulse

A backend pipeline for ingesting GDELT-derived news records, clustering them into evolving event entities, and making those events queryable.

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
- [API & Dashboard](#api--dashboard)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
  - [Hybrid Search](#hybrid-search)
  - [Dashboard](#dashboard)
- [Utility Scripts](#utility-scripts)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Project Goal

Transform the continuous GDELT news stream into a structured event index.

GDELT provides structured metadata about global news coverage, but does not directly provide semantic news story clusters. This project builds an incremental event-construction layer on top of GDELT: each **event** represents a group of related articles describing the same real-world development.

```
GDELT records  ->  normalized articles  ->  clustered events  ->  hybrid search API  ->  interactive dashboard
```

---

## How It Works

The system has three pipeline stages plus an API/visualization layer:

1. **Ingestion** — Fetches the latest GDELT GKG (Global Knowledge Graph) data, normalizes article metadata (URLs, sources, themes, entities), deduplicates by canonical URL, and stores articles in PostgreSQL.

2. **Embedding** — Takes articles that don't yet have an embedding, composes a text representation from title + metadata, and generates a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Vectors are stored in the database via pgvector.

3. **Clustering** — Takes articles with embeddings but no cluster assignment, and matches them to existing event clusters using a two-stage scoring system:
   - **Cosine similarity** between article embedding and cluster centroids
   - **Entity overlap** (locations, persons, organizations) weighted by importance
   - If no cluster is similar enough, a new event cluster is created

Each pipeline stage is incremental — it picks up where it left off, so the pipeline can run continuously or on a schedule.

4. **API & Search** — A FastAPI server exposes REST endpoints for browsing articles and clusters, plus a hybrid search that combines semantic vector similarity (pgvector HNSW) with PostgreSQL full-text search, merged via Reciprocal Rank Fusion (RRF). An interactive single-page dashboard provides search with configurable semantic/keyword weighting and multi-dimensional filtering.

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
│  1. INGESTION            │
│  Fetch GKG ZIP > parse > │
│  normalize > dedupe >    │
│  upsert > checkpoint     │
└─────────┬────────────────┘
          │
          v
┌──────────────────────────┐
│  2. EMBEDDING            │
│  Fetch unembedded >      │
│  compose text > embed >  │
│  store vectors           │
└─────────┬────────────────┘
          │
          v
┌──────────────────────────┐
│  3. CLUSTERING           │
│  Fetch unclustered >     │
│  score candidates >      │
│  assign or create >      │
│  update centroids        │
└─────────┬────────────────┘
          │
          v
┌──────────────────────────────────┐
│  4. API & DASHBOARD (FastAPI)    │
│  /api/search  — hybrid search   │
│  /api/articles — browse articles │
│  /api/clusters — browse events   │
│  /api/stats   — dashboard stats  │
│  /           — SPA dashboard     │
└──────────────────────────────────┘
```

**State tracking:** The `pipeline_state` table tracks the last processed GDELT timestamp, enabling incremental re-runs without reprocessing.

---

## Repository Structure

```
gdelt-pulse/
├── src/gdelt_event_pipeline/
│   ├── ingestion/          # GDELT fetching, GKG parsing, title scraping
│   ├── normalization/      # URL canonicalization, source mapping, GKG field parsing
│   ├── storage/            # PostgreSQL operations (articles, clusters, pipeline state)
│   ├── embeddings/         # Vector embedding generation (sentence-transformers)
│   ├── clustering/         # Event clustering with entity-aware scoring
│   ├── config/             # Settings loaded from environment variables
│   ├── query/              # Hybrid search layer (vector, keyword, RRF ranking, filters)
│   ├── api/                # FastAPI server + static dashboard SPA
│   │   ├── app.py          # API endpoints and server entry point
│   │   └── static/         # Single-page dashboard (vanilla JS)
│   ├── pipeline/           # (planned) Unified pipeline orchestration
│   └── utils/              # Shared utilities
├── tests/                  # Mirror structure of src/ with pytest tests
├── scripts/
│   ├── browse_articles.py  # CLI tool to inspect stored articles
│   └── cluster_viewer.py   # Generates interactive HTML cluster report
├── sql/
│   └── 001_schema.sql      # Database schema (PostgreSQL + pgvector)
├── docs/
│   └── database_design.md  # Schema design decisions and field documentation
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
```

This creates four tables (`articles`, `clusters`, `cluster_memberships`, `pipeline_state`) with all required indexes, including HNSW indexes for vector similarity search. See [docs/database_design.md](docs/database_design.md) for full schema documentation.

3. **Verify the setup:**

```bash
psql -d gdelt_pulse -c "\dt"
```

You should see all four tables listed.

### Configuration

Edit `.env` with your PostgreSQL credentials:

```bash
# PostgreSQL connection
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=your_password
PGDATABASE=gdelt_pulse
```

Optional embedding settings (defaults work out of the box):

```bash
# Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=64
```

> **Note:** The first time you run the embedding stage, the model (~80 MB) will be downloaded automatically by sentence-transformers.

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
# Run with default settings (threshold=0.75, limit=500)
uv run python -m gdelt_event_pipeline.clustering

# Adjust similarity threshold (lower = more permissive matching)
uv run python -m gdelt_event_pipeline.clustering --threshold 0.70

# Process more articles per run
uv run python -m gdelt_event_pipeline.clustering --limit 1000

# Verbose output (shows per-article assignment details)
uv run python -m gdelt_event_pipeline.clustering -v
```

### Full Pipeline Run

Run all three stages in sequence:

```bash
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles && \
uv run python -m gdelt_event_pipeline.embeddings && \
uv run python -m gdelt_event_pipeline.clustering -v
```

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

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the dashboard SPA |
| `GET` | `/api/stats` | Dashboard statistics (total articles, clusters, largest cluster, etc.) |
| `GET` | `/api/articles?limit=50` | Recent articles with titles (max 200) |
| `GET` | `/api/clusters?limit=100&sort=recent` | Active clusters, sortable by `recent`, `articles`, or `oldest` |
| `GET` | `/api/clusters/{id}` | Cluster detail with member articles and similarity scores |
| `GET` | `/api/search?q=...` | Hybrid semantic + keyword search (see below) |

### Hybrid Search

The `/api/search` endpoint combines two search strategies and merges results using **Reciprocal Rank Fusion (RRF)**:

1. **Semantic search** — Embeds the query with the same model (all-MiniLM-L6-v2) and finds nearest neighbors via pgvector HNSW index
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

### Dashboard

The dashboard is a single-page application (vanilla JS, no framework dependencies) with four views:

- **Home** — Overview stats and quick access to recent articles and top clusters
- **Search** — Full hybrid search with a semantic/keyword weight slider, collapsible filter panel, and tabbed results (articles + clusters)
- **Articles** — Browsable list of recent articles with metadata tags
- **Clusters** — Sortable list of active event clusters; click any cluster to open a detail modal with all member articles and their similarity scores

---

## Utility Scripts

**Browse articles** — Display stored articles with metadata:

```bash
uv run python scripts/browse_articles.py
```

**Cluster viewer** — Generate an interactive HTML report of event clusters:

```bash
uv run python scripts/cluster_viewer.py
# Opens cluster_report.html in your browser
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Run tests for a specific module
uv run pytest tests/normalization/
uv run pytest tests/clustering/

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
| Frontend | Single-page app (vanilla JavaScript, no framework) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2, 384-dim) |
| DB driver | [psycopg 3](https://www.psycopg.org/psycopg3/) + connection pooling |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Testing | pytest |
| Linting | ruff |

---

## Roadmap

Phase 1 (current) — reliable pipeline + search API:

- [x] Incremental GDELT ingestion with checkpointing
- [x] Article normalization and deduplication
- [x] Vector embedding generation
- [x] Entity-aware event clustering
- [x] Hybrid search layer (semantic + keyword + RRF)
- [x] REST API with FastAPI
- [x] Interactive dashboard (SPA)
- [ ] Unified pipeline orchestration

Later phases may add:

- Event analytics and trend detection
- Coverage and source diversity analysis
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