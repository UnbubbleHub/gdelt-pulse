# Deployment Guide

GDELT Pulse is designed as a split deployment: the API runs on **Vercel** (serverless) and the pipeline runs on **Railway** (Docker). Both connect to the same **PostgreSQL + pgvector** database (we use [Neon](https://neon.tech/)).

```
┌───────────┐     ┌──────────────────┐     ┌───────────┐
│  Vercel   │────►│   Neon (Postgres │◄────│  Railway  │
│  (API)    │     │   + pgvector)    │     │ (Pipeline)│
└───────────┘     └──────────────────┘     └───────────┘
```

## Database (Neon)

### Setup

1. Create a project at [neon.tech](https://neon.tech/)
2. Enable the pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Schema is applied automatically on first API startup via `storage/migrations.py`

### Connection

Neon provides a connection string in the format:

```
postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
```

This is set as `DATABASE_URL` in both Vercel and Railway.

---

## API -- Vercel

### How it works

- `api/index.py` is the Vercel entry point -- it re-exports the FastAPI `app` object
- `vercel.json` routes all requests through this single function
- `vercel.json` bundles `src/**` and `sql/**` via `includeFiles`
- The FastAPI lifespan handler initializes the DB pool and applies schema on cold start

### Deploy

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to production
vercel --prod
```

Or connect your GitHub repository in the Vercel dashboard for automatic deploys on push to `main`.

### Environment Variables

Set these in the Vercel dashboard (Settings > Environment Variables):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Neon connection string |
| `CLERK_PUBLISHABLE_KEY` | Yes | Clerk frontend key (for dashboard auth widget) |
| `CLERK_SECRET_KEY` | Yes | Clerk backend key (for JWT verification) |
| `CLERK_JWKS_URL` | Yes | Clerk JWKS endpoint for key rotation |
| `UPSTASH_REDIS_REST_URL` | Recommended | Upstash Redis URL for distributed rate limiting |
| `UPSTASH_REDIS_REST_TOKEN` | Recommended | Upstash Redis auth token |

If Neon is added as a Vercel integration, `DATABASE_URL` is injected automatically.

Without Upstash Redis, rate limiting falls back to in-memory counters (resets on each cold start -- fine for low traffic, not reliable at scale).

### Verify

After deploying, check:

- `GET /` -- Landing page loads
- `GET /api/docs` -- Swagger UI renders with all endpoints
- `GET /api/stats` -- Returns JSON with article/cluster counts
- `GET /api/clusters` -- Returns cluster data (empty array is fine pre-pipeline)

---

## Pipeline -- Railway

### How it works

- `Dockerfile.pipeline` builds the pipeline image
- `railway.toml` configures Railway to use this Dockerfile with `restartPolicyType = "always"`
- The runner module (`gdelt_event_pipeline.runner`) loops through ingest → embed → cluster every 15 minutes
- Graceful shutdown on `SIGTERM` (sent by Railway during deploys)

### Deploy

1. Create a new service in Railway
2. Connect your GitHub repository
3. Set the root directory if needed, or let Railway detect `Dockerfile.pipeline` via `railway.toml`
4. Railway auto-deploys on push to `main`

### Environment Variables

Set these in the Railway service settings:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Neon connection string (same as Vercel) |
| `EMBEDDING_BACKEND` | Yes | Set to `fastembed` |
| `PIPELINE_INTERVAL` | No | Cycle interval in seconds (default: 900) |
| `RETENTION_HOURS` | No | Delete articles older than this (default: 168 = 7 days; 0 disables) |

### Monitoring

Railway provides logs for the running service. The pipeline logs each cycle:

```
[2026-05-01 12:00:01] Starting pipeline cycle
[2026-05-01 12:00:03] Ingested 47 articles (12 new, 35 updated)
[2026-05-01 12:00:15] Embedded 12 articles
[2026-05-01 12:00:18] Clustered 12 articles (8 assigned, 4 new clusters)
[2026-05-01 12:00:18] Cycle complete. Next run in 900s
```

---

## Post-Deploy Checklist

After both services are running:

| Check | Command / URL | Expected |
|-------|---------------|----------|
| Landing page | `GET /` | HTML renders |
| Swagger docs | `GET /api/docs` | Interactive docs with 4 endpoint groups |
| Stats | `GET /api/stats` | JSON with counters |
| Clusters | `GET /api/clusters` | JSON array (may be empty initially) |
| Search | `GET /api/search?q=test` | Search results or `501` if no embeddings yet |
| Articles | `GET /api/articles` | JSON array |
| Dashboard | `GET /dashboard` | Clerk auth widget loads |
| Developers | `GET /developers` | Reference page renders |
| Pipeline logs | Railway dashboard | Cycle logs appearing every 15 min |

---

## Local Development

The recommended local setup is documented in [Getting Started](getting-started.md). It supports three database options (Neon, Docker Compose, local Postgres) all driven by a single `DATABASE_URL` env var that mirrors production.

For an offline workflow, `docker-compose.yml` ships a `pgvector/pgvector:pg16` container with fixed local creds bound to `127.0.0.1:5432`:

```bash
docker compose up -d db
# DATABASE_URL=postgresql://gdelt:gdelt@localhost:5432/gdelt_pulse
```

Then run the API and pipeline natively:

```bash
# API
uv run uvicorn gdelt_event_pipeline.api.app:app --reload

# Pipeline (in another terminal)
uv run python -m gdelt_event_pipeline.runner
```

---

## Troubleshooting

**Vercel function timeout:** Vercel's free tier has a 10-second function timeout. Embedding-heavy search queries may exceed this. The API uses a minimal connection pool (`min_size=0, max_size=2`) on Vercel to handle cold starts efficiently.

**Schema not applied:** If tables are missing, the API applies `001_schema.sql` and `002_api_keys.sql` on startup. Check Vercel function logs for migration errors. The `sql/` directory must be included in the Vercel bundle (verified via `vercel.json` `includeFiles`).

**Pipeline not picking up new data:** Check `pipeline_state` in the database -- it tracks the last processed GDELT timestamp. If stuck, update or delete the row to re-trigger ingestion.

**Rate limiting not working:** Without Upstash Redis, rate limits reset on each Vercel cold start. Add `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` for persistent, distributed rate limiting.
