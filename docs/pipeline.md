# Pipeline Guide

The pipeline transforms raw GDELT data into searchable, clustered events through three stages. Each stage is incremental -- it picks up where it left off, so the pipeline can run on a schedule or continuously.

## Overview

```
GDELT GKG feed (every 15 min)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Ingestion  │────►│  Embedding   │────►│  Clustering  │
│              │     │              │     │              │
│  Fetch GKG   │     │  Generate    │     │  Score &     │
│  Parse       │     │  384-dim     │     │  assign to   │
│  Normalize   │     │  vectors     │     │  event       │
│  Deduplicate │     │  (fastembed) │     │  clusters    │
│  Upsert      │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Running Individual Stages

### Ingestion

```bash
# Fetch the latest GKG file
uv run python -m gdelt_event_pipeline.ingestion

# With title scraping (recommended)
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles

# Scrape titles only (no new ingestion)
uv run python -m gdelt_event_pipeline.ingestion --scrape-only

# Fetch a specific GKG file by URL
uv run python -m gdelt_event_pipeline.ingestion --url <gkg_file_url>

# Preview without writing to the database
uv run python -m gdelt_event_pipeline.ingestion --dry-run

# Verbose output
uv run python -m gdelt_event_pipeline.ingestion -v
```

What happens during ingestion:

1. Downloads the latest GDELT GKG ZIP file (or resumes from the last checkpoint)
2. Parses the tab-delimited GKG format into structured article records
3. Normalizes URLs (strips tracking parameters, resolves canonical URLs)
4. Maps source domains to human-readable source names
5. Parses delimited fields: themes, locations, persons, organizations, tone
6. Deduplicates by canonical URL
7. Upserts articles into PostgreSQL in batches
8. Updates the `pipeline_state` checkpoint

### Embedding

```bash
uv run python -m gdelt_event_pipeline.embeddings
```

For each article without an embedding:

1. Composes a text representation: `title | themes | entities | source`
2. Generates a 384-dimensional vector using fastembed (all-MiniLM-L6-v2)
3. Stores the vector in the `embedding` column

Articles without titles are skipped (titles carry the most semantic signal). Processes in batches.

### Clustering

```bash
# Default settings
uv run python -m gdelt_event_pipeline.clustering

# Adjust similarity threshold (lower = more permissive matching)
uv run python -m gdelt_event_pipeline.clustering --threshold 0.70

# Custom temporal window (only match clusters active within N hours)
uv run python -m gdelt_event_pipeline.clustering --window 48

# Disable temporal window (consider all active clusters)
uv run python -m gdelt_event_pipeline.clustering --window 0

# Process more articles per run
uv run python -m gdelt_event_pipeline.clustering --limit 1000

# Verbose output (per-article assignment details)
uv run python -m gdelt_event_pipeline.clustering -v
```

For each unclustered article:

1. Loads candidate clusters (active within the temporal window)
2. Scores each candidate using:
   - **Cosine similarity** between article embedding and cluster centroid
   - **Entity overlap** (weighted Jaccard across locations, persons, organizations)
3. If the best score exceeds the threshold: assigns the article to that cluster
4. Otherwise: creates a new cluster seeded with this article
5. Recomputes the cluster centroid

### Full Pipeline

Run all three stages in sequence:

```bash
uv run python -m gdelt_event_pipeline.ingestion --scrape-titles && \
uv run python -m gdelt_event_pipeline.embeddings && \
uv run python -m gdelt_event_pipeline.clustering -v
```

## Continuous Runner

For production, the runner module orchestrates all stages in a loop:

```bash
uv run python -m gdelt_event_pipeline.runner
```

Each cycle:

1. Ingests the latest GKG data
2. Scrapes missing titles
3. Generates embeddings for new articles
4. Clusters embedded articles
5. Cleans up articles that failed title scraping
6. Sleeps for the configured interval (default: 900 seconds)

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PIPELINE_INTERVAL` | 900 | Seconds between cycles (matches GDELT's 15-min cadence) |
| `EMBEDDING_BACKEND` | fastembed | Embedding engine (fastembed is the only supported option) |
| `CLUSTER_WINDOW_HOURS` | 72 | Only consider clusters active within this window |
| `RETENTION_HOURS` | 72 | Delete articles older than this (hours). Set to 0 to disable |

### Graceful Shutdown

The runner handles `SIGTERM` and `SIGINT` signals. When received:

1. The current cycle is allowed to finish
2. The database pool is closed
3. The process exits cleanly

Railway sends `SIGTERM` during deploys, so in-progress cycles complete before the new version starts.

### Error Handling

If a cycle fails:

- The error is logged with full traceback
- The pipeline waits with exponential backoff before retrying
- A single stage failure doesn't corrupt the overall pipeline state (each stage writes its own checkpoints)

## Tuning

### Clustering Threshold

The `--threshold` parameter (default: 0.75) controls how similar an article must be to a cluster to join it:

- **0.80+** -- Tight clusters, more events created, fewer false merges
- **0.75** -- Balanced (default)
- **0.70** -- Looser clusters, fewer events, risk of merging distinct stories

### Temporal Window

The `--window` parameter (default: 72 hours) limits which clusters are considered as candidates:

- **24h** -- Very recent events only (fast, but misses multi-day stories)
- **72h** -- Three-day window (default, good for most news cycles)
- **168h** -- One-week window (for slower-developing stories)
- **0** -- No window (consider all active clusters -- slower, risk of old cluster absorption)

### Batch Sizes

Embedding and clustering process articles in batches. For large backlogs:

```bash
# Cluster more articles per run
uv run python -m gdelt_event_pipeline.clustering --limit 2000
```
