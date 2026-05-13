# API Reference

Base URL: `https://<your-deployment>.vercel.app`

Interactive Swagger docs are available at `/api/docs`.

## Authentication

Public endpoints require no authentication but are rate-limited to **30 requests/minute** per IP.

For higher limits (**200 requests/minute**), pass an API key:

```
X-API-Key: gdp_your_api_key_here
```

API keys are managed through the `/api/auth/keys` endpoints (requires Clerk authentication).

---

## Search

### `GET /api/search`

Hybrid semantic + keyword search across articles and optionally clusters. Results are ranked using Reciprocal Rank Fusion (RRF).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | *required* | Search query text |
| `limit` | integer | 20 | Max results (1--100) |
| `semantic_weight` | float | 0.5 | Balance: 0.0 = pure keyword, 1.0 = pure semantic |
| `clusters` | boolean | false | Also search cluster centroids |
| `location` | string | -- | Filter by location (comma-separated for multiple) |
| `person` | string | -- | Filter by person name |
| `org` | string | -- | Filter by organization |
| `theme` | string | -- | Filter by GDELT theme code |
| `domain` | string | -- | Filter by source domain (comma-separated). Soft match: `corriere.it` also matches `video.corriere.it`. |
| `source` | string | -- | Filter by canonical source slug (comma-separated) |
| `date_from` | datetime | -- | Start date (ISO 8601) |
| `date_to` | datetime | -- | End date (ISO 8601) |

**Example:**

```bash
curl "/api/search?q=earthquake+Turkey&semantic_weight=0.7&location=Turkey&limit=10"
```

**Response:**

```json
{
  "articles": [
    {
      "url": "https://example.com/article",
      "title": "Major earthquake strikes southeastern Turkey",
      "gdelt_timestamp": "2026-05-01T12:00:00",
      "domain": "example.com",
      "source_common_name": "Example News",
      "locations": ["Turkey", "Gaziantep"],
      "persons": [],
      "organizations": ["AFAD"],
      "themes": ["NATURAL_DISASTER", "HUMANITARIAN_AID"],
      "tone": -3.2,
      "score": 0.847
    }
  ],
  "clusters": []
}
```

See [Search](search.md) for details on how hybrid search works internally.

---

## Clusters

### `GET /api/clusters`

List active event clusters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Max clusters to return (1--500) |
| `sort` | string | `recent` | Sort order: `recent`, `articles`, or `oldest` |
| `location` | string | -- | Filter by location (comma-separated) |
| `person` | string | -- | Filter by person (comma-separated) |
| `org` | string | -- | Filter by organization (comma-separated) |
| `theme` | string | -- | Filter by GDELT theme (comma-separated) |
| `domain` | string | -- | Filter by source domain (comma-separated). Soft match: `corriere.it` also matches `video.corriere.it`. |
| `source` | string | -- | Filter by canonical source slug (comma-separated) |
| `date_from` | datetime | -- | Start date (ISO 8601) |
| `date_to` | datetime | -- | End date (ISO 8601) |

Filters share the same semantics as `/api/search` (JSONB containment for entities, exact / soft-domain match for domain) — applied to article-level fields, then clusters are returned for any matching member.

**Example:**

```bash
curl "/api/clusters?sort=articles&limit=5"
```

**Response:**

```json
[
  {
    "id": 42,
    "label": "EU AI Act implementation",
    "article_count": 187,
    "first_seen": "2026-04-28T08:00:00",
    "last_updated": "2026-05-01T14:30:00",
    "top_locations": ["Brussels", "Europe"],
    "top_persons": ["Thierry Breton"],
    "top_organizations": ["European Commission", "EU Parliament"],
    "top_themes": ["AI_REGULATION", "TECHNOLOGY"],
    "avg_tone": -0.8
  }
]
```

### `GET /api/clusters/{id}`

Get a single cluster with its member articles and similarity scores.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | integer | *required* | Cluster ID (path parameter) |

**Response:** Cluster metadata plus a `members` array of articles with their `similarity_score` to the cluster centroid.

Returns `404` if the cluster does not exist or is not active.

---

## Articles

### `GET /api/articles`

List recent articles, newest first, with optional filters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Max articles to return (1--200) |
| `location` | string | -- | Filter by location (comma-separated) |
| `person` | string | -- | Filter by person (comma-separated) |
| `org` | string | -- | Filter by organization (comma-separated) |
| `theme` | string | -- | Filter by GDELT theme (comma-separated) |
| `domain` | string | -- | Filter by source domain (comma-separated). Soft match: `corriere.it` also matches `video.corriere.it`. |
| `source` | string | -- | Filter by canonical source slug (comma-separated) |
| `date_from` | datetime | -- | Start date (ISO 8601) |
| `date_to` | datetime | -- | End date (ISO 8601) |

Filter semantics are shared with `/api/search` and `/api/clusters` (JSONB containment for entities, exact + subdomain match for `domain`, exact match for `source`).

**Example:**

```bash
curl "/api/articles?location=Ukraine&theme=MILITARY_CONFLICT&limit=20"
```

### `GET /api/stats`

Pipeline statistics.

**Response:**

```json
{
  "total_articles": 45230,
  "titled_articles": 38100,
  "embedded_articles": 37500,
  "total_clusters": 1240,
  "largest_cluster": 412,
  "total_memberships": 36800
}
```

---

## Auth & API Keys

All `/api/auth/keys` endpoints require a valid Clerk JWT in the `Authorization` header.

### `GET /api/auth/keys`

Check if the authenticated user has an active API key.

**Response:**

```json
{
  "has_key": true,
  "created_at": "2026-04-15T10:00:00",
  "prefix": "gdp_abc1..."
}
```

### `POST /api/auth/keys`

Create or rotate an API key. If the user already has an active key, it is revoked and a new one is issued.

**Response:**

```json
{
  "api_key": "gdp_abc123def456...",
  "message": "Store this key securely. It will not be shown again."
}
```

### `DELETE /api/auth/keys`

Revoke the user's active API key.

### `GET /api/auth/config`

Returns public Clerk configuration for the frontend. No authentication required.

```json
{
  "clerk_publishable_key": "pk_test_..."
}
```

---

## Common Patterns

### Filtering

Most list endpoints accept the same filter parameters. Filters use ILIKE matching, so partial strings work:

```bash
# Articles mentioning any location containing "york"
/api/articles?location=york

# Clusters about military themes from BBC
/api/clusters?theme=MILITARY&domain=bbc.com
```

Multiple values for the same field are not supported -- use the first value only.

### Date Ranges

Use ISO 8601 format for date filters:

```bash
/api/articles?date_from=2026-04-01T00:00:00&date_to=2026-04-30T23:59:59
```

### Rate Limiting

Rate limit headers are included in every response:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Requests allowed per window |
| `X-RateLimit-Remaining` | Requests remaining |
| `X-RateLimit-Reset` | Seconds until the window resets |

When the limit is exceeded, the API returns `429 Too Many Requests`.

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `400` | Bad request (invalid parameters) |
| `404` | Resource not found |
| `422` | Validation error (FastAPI parameter validation) |
| `429` | Rate limit exceeded |
| `501` | Feature not available (e.g., search when no embedding model is loaded) |

Error body:

```json
{
  "detail": "Description of the error"
}
```
