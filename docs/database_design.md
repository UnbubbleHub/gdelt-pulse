# Database Design

PostgreSQL + pgvector schema for gdelt-pulse.

## Design Decisions

- **One record per URL.** If the same article appears in multiple GKG dumps, we upsert and merge metadata. `first_seen_at` keeps the earliest GDELT timestamp; `last_seen_at` tracks the latest observation.
- **JSONB for semi-structured GKG fields.** Locations, persons, organizations, and tone are parsed from raw GKG strings into structured JSONB on ingestion. No raw comma-separated strings stored.
- **Separate cluster_memberships table.** Article-to-cluster assignment is explicit and traceable, with similarity score and assignment method recorded. An article can theoretically belong to multiple clusters.
- **raw_payload is mandatory.** Stored as JSONB for every article. Invaluable for debugging parsing bugs during early pipeline development.
- **Title stored, full text deferred.** Title is extracted and stored for embedding input. Full article content may be added in a later phase.
- **Cluster lifecycle is a simple boolean.** `is_active` on clusters, not a status enum. Sufficient for phase 1.

## Tables

### articles

Main input table. One record per unique canonical URL.

| Column | Type | Notes |
|---|---|---|
| id | UUID | Primary key |
| gkg_record_id | TEXT UNIQUE | Original GKG row identity |
| gdelt_timestamp | TIMESTAMPTZ | Timestamp from GDELT, used for time-ordering |
| url | TEXT | Original DocumentIdentifier (not really needed) |
| canonical_url | TEXT UNIQUE | Normalized URL, primary dedupe key |
| domain | TEXT | Extracted from canonical_url |
| source_common_name | TEXT | Raw source name from GDELT |
| canonical_source | TEXT | Normalized source identity |
| title | TEXT | Article title for embedding input |
| themes | JSONB | Parsed GKG themes with scores |
| locations | JSONB | Structured location data |
| organizations | JSONB | Extracted organization names |
| persons | JSONB | Extracted person names |
| all_names | JSONB | All extracted names |
| tone | JSONB | Parsed tone metrics (tone, positive, negative, polarity, activity_ref_density) |
| embedding | vector | Article embedding for similarity search |
| embedding_model | TEXT | Model used to generate the embedding |
| raw_payload | JSONB | Original GKG row for debugging |
| first_seen_at | TIMESTAMPTZ | Earliest GDELT observation |
| last_seen_at | TIMESTAMPTZ | Latest GDELT observation |
| created_at | TIMESTAMPTZ | Row creation time |
| updated_at | TIMESTAMPTZ | Row last modified time |

### clusters

Each cluster represents one real-world event entity.

| Column | Type | Notes |
|---|---|---|
| id | UUID | Primary key |
| representative_title | TEXT | Nullable initially; derived from top themes or first article |
| summary | TEXT | Optional event summary |
| centroid_embedding | vector | Cluster centroid for nearest-cluster assignment |
| article_count | INTEGER | Denormalized count of member articles |
| first_article_at | TIMESTAMPTZ | Timestamp of earliest member article |
| last_article_at | TIMESTAMPTZ | Timestamp of latest member article |
| is_active | BOOLEAN | Whether the cluster is still accepting new articles |
| created_at | TIMESTAMPTZ | Row creation time |
| updated_at | TIMESTAMPTZ | Row last modified time |

### cluster_memberships

Links articles to clusters. Keeps assignment explicit and traceable.

| Column | Type | Notes |
|---|---|---|
| id | UUID | Primary key |
| article_id | UUID | FK -> articles.id |
| cluster_id | UUID | FK -> clusters.id |
| similarity_score | DOUBLE PRECISION | Cosine similarity at time of assignment |
| assignment_method | TEXT | How the article was assigned (e.g. "nearest_centroid", "new_cluster") |
| assigned_at | TIMESTAMPTZ | When the assignment was made |
| created_at | TIMESTAMPTZ | Row creation time |

Unique constraint on (article_id, cluster_id).

### pipeline_state

Tracks incremental ingestion progress per source.

| Column | Type | Notes |
|---|---|---|
| id | UUID | Primary key |
| source_name | TEXT UNIQUE | Source identifier, e.g. "gdelt_gkg" |
| last_processed_timestamp | TIMESTAMPTZ | Most recent GDELT timestamp processed |
| last_processed_record_id | TEXT | Most recent GKG record ID processed |
| last_successful_run_at | TIMESTAMPTZ | When the pipeline last completed successfully |
| updated_at | TIMESTAMPTZ | Row last modified time |

Seeded with a single row for `gdelt_gkg`.

## Indexes

### B-tree

- `articles.gdelt_timestamp` — time-range queries
- `articles.domain` — filter by source domain
- `articles.canonical_source` — filter by normalized source
- `articles.created_at` — recent ingestion queries
- `clusters.is_active` — filter active clusters
- `clusters.last_article_at` — sort/filter clusters by recency
- `cluster_memberships.article_id` — look up clusters for an article
- `cluster_memberships.cluster_id` — look up articles in a cluster

### GIN

- `articles.themes` — JSONB containment queries on themes
- `articles.locations` — JSONB containment queries on locations
- `articles.organizations` — JSONB containment queries on organizations
- `articles.persons` — JSONB containment queries on persons

### HNSW (pgvector)

- `articles.embedding` — approximate nearest neighbor search (cosine)
- `clusters.centroid_embedding` — nearest cluster lookup (cosine)

## Upsert Strategy

On ingestion, use `INSERT ... ON CONFLICT (canonical_url) DO UPDATE`:

- Merge metadata fields if the new observation is richer (more themes, more persons, etc.)
- Keep `first_seen_at` as the minimum of existing and new `gdelt_timestamp`
- Update `last_seen_at` to the new `gdelt_timestamp`
- Update `updated_at` to `now()`

## JSONB Field Formats

### themes

```json
[
  {"theme": "ARMEDCONFLICT", "score": 2056},
  {"theme": "EPU_CATS_NATIONAL_SECURITY", "score": 2056}
]
```

### tone

```json
{
  "tone": -1.19,
  "positive_score": 2.08,
  "negative_score": 3.27,
  "polarity": 5.35,
  "activity_ref_density": 17.26
}
```

### locations / persons / organizations

Parsed from GKG into arrays of structured objects. Exact shape depends on what the GKG fields provide per record.
