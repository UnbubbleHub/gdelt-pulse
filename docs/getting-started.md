# Getting Started

This guide walks through setting up GDELT Pulse for local development, from prerequisites through your first pipeline run.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for `match` statements and type union syntax |
| uv | Latest | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |
| PostgreSQL 15+ with pgvector | — | Pick one option below |

You don't need to install Postgres locally. Pick whichever DB option matches your workflow:

- **Neon (recommended)** — free cloud Postgres, identical to production. Requires only an internet connection.
- **Docker Compose** — containerized Postgres + pgvector. Works fully offline.
- **Local Postgres** — Homebrew, Postgres.app, or a system install if you already have one.

## Installation

```bash
git clone https://github.com/UnbubbleHub/gdelt-pulse.git
cd gdelt-pulse

uv sync                # install all dependencies (core + dev + API)
cp .env.example .env   # configure DATABASE_URL — see below
```

## Database Setup

The API server **auto-applies the schema** on startup (`001_schema.sql` + `002_api_keys.sql`) if tables don't exist. You don't need to run any `psql` or `createdb` commands manually for any of the three options below.

The schema creates these tables:

| Table | Purpose |
|-------|---------|
| `articles` | Ingested articles with metadata, embeddings, and full-text search index |
| `clusters` | Event clusters with centroids and entity summaries |
| `cluster_memberships` | Article-to-cluster assignments with similarity scores |
| `pipeline_state` | Tracks the last processed GDELT timestamp for incremental ingestion |
| `api_keys` | Per-user API key management |

See [Database Design](database_design.md) for full schema documentation.

### Option 1 — Neon (recommended)

1. Create a free project at [neon.tech](https://neon.tech/)
2. Copy the connection string from the dashboard (looks like `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require`)
3. Set it as `DATABASE_URL` in `.env`

pgvector is pre-installed on Neon — the schema enables the extension automatically.

### Option 2 — Docker Compose

```bash
docker compose up -d db
```

This starts a `pgvector/pgvector:pg16` container with fixed local-only credentials (`gdelt`/`gdelt`/`gdelt_pulse`) bound to `127.0.0.1:5432`. Set `DATABASE_URL` in `.env` to:

```bash
DATABASE_URL=postgresql://gdelt:gdelt@localhost:5432/gdelt_pulse
```

### Option 3 — Local Postgres install

If you already have Postgres running, just create a database and install pgvector:

```bash
createdb gdelt_pulse
psql -d gdelt_pulse -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

To install pgvector:

```bash
brew install pgvector                       # macOS
sudo apt install postgresql-15-pgvector     # Ubuntu/Debian
```

Then point `DATABASE_URL` at it:

```bash
DATABASE_URL=postgresql://your_user:your_password@localhost:5432/gdelt_pulse
```

## Configuration

The only required setting is `DATABASE_URL`. See `.env.example` for the full list of optional variables (pipeline interval, retention, Clerk auth for the dashboard, Upstash Redis for rate limiting).

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

## Next Steps

- **[Pipeline Guide](pipeline.md)** -- Configure and run the continuous pipeline
- **[API Reference](api-reference.md)** -- Explore all available endpoints
- **[Deployment Guide](deployment.md)** -- Deploy to Vercel and Railway
- **[Architecture](architecture.md)** -- Understand the system design
