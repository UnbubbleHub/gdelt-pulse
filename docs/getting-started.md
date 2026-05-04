# Getting Started

This guide walks through setting up GDELT Pulse for local development, from prerequisites through your first pipeline run.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for `match` statements and type union syntax |
| PostgreSQL | 15+ | With the [pgvector](https://github.com/pgvector/pgvector) extension |
| uv | Latest | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |

### Installing pgvector

pgvector adds vector similarity search to PostgreSQL. It's required for semantic search and embedding storage.

```bash
# macOS (Homebrew)
brew install pgvector

# Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# From source (any platform)
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make && make install
```

See the [pgvector installation docs](https://github.com/pgvector/pgvector#installation) for other platforms.

## Installation

```bash
# Clone the repository
git clone https://github.com/UnbubbleHub/gdelt-pulse.git
cd gdelt-pulse

# Install all dependencies (core + dev + API)
uv sync

# Copy the environment template
cp .env.example .env
```

## Database Setup

### 1. Create the database

```bash
createdb gdelt_pulse
```

### 2. Apply the schema

```bash
psql -d gdelt_pulse -f sql/001_schema.sql
psql -d gdelt_pulse -f sql/002_api_keys.sql
```

`001_schema.sql` creates four core tables:

| Table | Purpose |
|-------|---------|
| `articles` | Ingested articles with metadata, embeddings, and full-text search index |
| `clusters` | Event clusters with centroids and entity summaries |
| `cluster_memberships` | Article-to-cluster assignments with similarity scores |
| `pipeline_state` | Tracks the last processed GDELT timestamp for incremental ingestion |

`002_api_keys.sql` adds the `api_keys` table for per-user key management.

> **Note:** The API server auto-applies both schema files on startup if tables don't exist. Manual application is only needed for standalone pipeline use or database inspection.

### 3. Verify

```bash
psql -d gdelt_pulse -c "\dt"
```

You should see `articles`, `clusters`, `cluster_memberships`, `pipeline_state`, and `api_keys`.

See [Database Design](database_design.md) for full schema documentation and design rationale.

## Configuration

Edit `.env` with your PostgreSQL credentials:

```bash
# Option A: Individual connection parameters
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=your_password
PGDATABASE=gdelt_pulse

# Option B: Connection URL (takes precedence over individual params)
# DATABASE_URL=postgresql://user:pass@localhost:5432/gdelt_pulse
```

### Optional settings

```bash
# Pipeline cycle interval in seconds (default: 900 = 15 minutes)
PIPELINE_INTERVAL=900

# Embedding backend (fastembed is the only supported option)
EMBEDDING_BACKEND=fastembed
```

### Auth settings (required for API key features)

```bash
# Clerk authentication (https://clerk.com)
CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
CLERK_JWKS_URL=https://<your-clerk-domain>/.well-known/jwks.json

# Upstash Redis for distributed rate limiting (optional)
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
```

## First Run

### Start the API server

```bash
uv run --group api uvicorn gdelt_event_pipeline.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://<your-host>:8000` for the landing page, or `http://<your-host>:8000/api/docs` for interactive Swagger docs.

### Run the pipeline

Run each stage individually to see what happens at each step:

```bash
# 1. Ingest the latest GDELT GKG file
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles -v

# 2. Generate embeddings for new articles
uv run python -m gdelt_event_pipeline.embeddings

# 3. Cluster embedded articles into events
uv run python -m gdelt_event_pipeline.clustering -v
```

Or run all three in sequence:

```bash
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles && \
uv run python -m gdelt_event_pipeline.embeddings && \
uv run python -m gdelt_event_pipeline.clustering -v
```

After the pipeline runs, search and browse endpoints will return data.

### Run tests

```bash
uv run --group api pytest tests/ -q
```

You should see all 353 tests pass. Tests use mocked database connections and don't require a running PostgreSQL instance.

## Windows Users

If you're using Windows Command Prompt, set the PostgreSQL user explicitly:

```cmd
# Current session only
set PGUSER=postgres

# Persistent across sessions
setx PGUSER postgres
```

## Next Steps

- **[Pipeline Guide](pipeline.md)** -- Configure and run the continuous pipeline
- **[API Reference](api-reference.md)** -- Explore all available endpoints
- **[Deployment Guide](deployment.md)** -- Deploy to Vercel and Railway
- **[Architecture](architecture.md)** -- Understand the system design
