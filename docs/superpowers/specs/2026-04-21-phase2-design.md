# API / Pipeline Split — Phase 2 Design

**Date:** 2026-04-21
**Branch:** `phase2-infra`
**Scope:** Neon migration, fastembed swap, Upstash Redis rate limiting, API key auth.

---

## Goal

Phase 1 split the API (Vercel) from the pipeline (Railway) and deferred four infrastructure improvements. Phase 2 delivers them:

| Item | Problem solved |
|---|---|
| Neon migration | Railway Postgres is a poor fit for Vercel cold starts; Neon's pooled endpoint (pgBouncer) handles ephemeral connections cleanly |
| fastembed swap | `sentence-transformers` (~500MB with PyTorch) exceeds Vercel's limit; `fastembed` (ONNX, smaller) may fit |
| Upstash Redis rate limiting | In-memory `_rate_limit_store` is per-instance on Vercel — limits are not shared across function instances |
| API key auth | The API is now public-facing; add a minimal global-key gate before broader exposure |

---

## Architecture

### Before Phase 2

```
Vercel (API)          Railway (pipeline)
    └─ psycopg ──────────────┘
         └── Railway Postgres (shared)

Rate limiting: in-memory dict (per-instance, not shared)
Auth: none
/api/search: 501 (sentence-transformers not installed on Vercel)
```

### After Phase 2

```
Vercel (API)          Railway (pipeline)
    └─ psycopg (pooled DSN) ──┘
         └── Neon Postgres (shared)
    └─ upstash-redis (HTTP) ── Upstash Redis

Rate limiting: Upstash Redis sliding window (shared across instances)
Auth: X-API-Key header checked against API_KEY env var
/api/search: live — if both fastembed gates pass (see Section 3)
```

### Files changed

| File | Change |
|---|---|
| `src/gdelt_event_pipeline/api/app.py` | Upstash Redis middleware, auth dependency, updated search guard |
| `src/gdelt_event_pipeline/embeddings/embed.py` | fastembed branch controlled by `EMBEDDING_BACKEND` env var |
| `requirements.txt` | Add `fastembed`, `upstash-redis` (pinned) — only after gates pass |
| `pyproject.toml` | Unchanged — pipeline keeps `sentence-transformers` |

### New files

| File | Purpose |
|---|---|
| `scripts/compare_embeddings.py` | One-off validation script — not deployed |

### Env vars

| Var | Where set | Purpose |
|---|---|---|
| `DATABASE_URL` | Railway + Vercel | Updated to Neon DSN (direct for Railway, pooled for Vercel) |
| `EMBEDDING_BACKEND` | Vercel only | `fastembed` to enable fastembed; defaults to `sentence-transformers` |
| `UPSTASH_REDIS_REST_URL` | Vercel only | Upstash Redis HTTP endpoint |
| `UPSTASH_REDIS_REST_TOKEN` | Vercel only | Upstash Redis auth token |
| `API_KEY` | Vercel only | Global API key (MVP); if unset, auth is disabled |

---

## Section 1 — Neon migration

### Provision

1. Create a Neon project
2. **Enable `pgvector` extension before restoring the dump:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Neon provides two DSNs per branch — direct and pooled

### Connection strings

- **Railway pipeline** → Neon **direct** DSN as `DATABASE_URL`. Uses `psycopg_pool` with `min_size=2`, persistent connections appropriate for a long-running worker.
- **Vercel API** → Neon **pooled** DSN (pgBouncer endpoint) as `DATABASE_URL`. The API already sets `min_size=0` when `VERCEL=1`, so pgBouncer handles external pooling.

### Migration steps (manual, not in code)

1. `pg_dump` Railway Postgres → `.sql` file (use `--no-owner --no-acl`)
2. Enable `pgvector` on Neon (see above)
3. `psql` restore to Neon
4. Verify row counts on critical tables: `articles`, `clusters`, `cluster_memberships`, `mv_country_comentions`, `mv_country_stats`
5. Update `DATABASE_URL` in both Railway and Vercel env vars (direct DSN for Railway, pooled DSN for Vercel)
6. Redeploy both services

### Code changes

None. `settings.py` already reads `DATABASE_URL` → `db.url` → `db.dsn`. The pool init in `app.py` already handles serverless sizing via `VERCEL=1`.

---

## Section 2 — fastembed swap

### Two independent gates

`/api/search` stays at 501 until **both** gates pass independently. They can be evaluated in any order.

#### Gate 1 — Vector compatibility (`scripts/compare_embeddings.py`)

- Embeds ~20 representative news-headline strings using both `sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` and `fastembed.TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")`
- Computes cosine similarity between each pair of output vectors; prints mean and min
- **Pass threshold:** mean cosine similarity ≥ 0.9999 (effectively identical vectors)
- Also runs 5–10 test queries against the live Neon DB using fastembed-produced query vectors; prints the top-5 articles per query for manual retrieval-quality inspection
- If either check fails: keep 501, document the outcome, do not add `fastembed` to `requirements.txt`

#### Gate 2 — Vercel bundle size

- Verify that `fastembed` + ONNX model files fit within Vercel's 500MB deployment limit
- Method: test-deploy to a Vercel preview branch with `fastembed` added to `requirements.txt`; check deployment size in the Vercel dashboard
- If bundle exceeds 500MB: keep 501 regardless of Gate 1 outcome

### `embed.py` changes

Backend is controlled by `EMBEDDING_BACKEND` env var, not inferred from `VERCEL`:

```python
import os

def embed_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[list[float]]:
    if not texts:
        return []

    backend = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers")
    if backend == "fastembed":
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name)
        return [v.tolist() for v in model.embed(texts)]

    # sentence-transformers path (existing, unchanged)
    model = load_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()
```

### `app.py` search guard

`_SEARCH_AVAILABLE` checks the configured backend:

```python
_backend = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers")
try:
    if _backend == "fastembed":
        import fastembed as _fe_check  # noqa: F401
    else:
        import sentence_transformers as _st_check  # noqa: F401
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False
```

### `requirements.txt`

Add `fastembed` (pinned exact version from local install) only after both gates pass.

---

## Section 3 — Upstash Redis rate limiting

### Goal

Replace the per-instance `_rate_limit_store: dict[str, list[float]]` with a Redis-backed sliding window shared across all Vercel function instances.

### Library

`upstash-redis` Python SDK — uses HTTP, no persistent TCP socket, zero connection overhead on serverless cold starts.

### Implementation

```python
# app.py — module level
from upstash_redis import Redis as UpstashRedis

_redis: UpstashRedis | None = None

def _get_redis() -> UpstashRedis | None:
    global _redis
    if _redis is None:
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            _redis = UpstashRedis(url=url, token=token)
    return _redis
```

Sliding window in the rate limit middleware (key: `ratelimit:{client_ip}`):

```python
redis = _get_redis()
if redis is None:
    # Fallback: in-memory dict (local dev / Railway — no Upstash credentials)
    # existing _rate_limit_store logic unchanged
    ...
else:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window_start)
    pipe.zcard(key)
    pipe.zadd(key, {f"{now}-{os.urandom(4).hex()}": now})
    pipe.expire(key, RATE_LIMIT_WINDOW)
    results = pipe.execute()
    count = results[1]
    if count >= RATE_LIMIT_MAX:
        return Response(..., status_code=429)
```

### Fallback

If `UPSTASH_REDIS_REST_URL` is not set, `_get_redis()` returns `None` and the middleware falls back to the existing in-memory dict. Local dev works without Upstash credentials.

### Rate limits

Unchanged from Phase 1: 30 requests per 60-second sliding window, per client IP, on all `/api/*` paths.

### `requirements.txt`

Add `upstash-redis` (pinned).

---

## Section 4 — API key auth

### Goal

Require a valid `X-API-Key` header on all `/api/*` endpoints. MVP global-key implementation — not per-user, no revocation, no DB involvement.

### Implementation

```python
# app.py
from fastapi.security import APIKeyHeader
from fastapi import Security

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def _verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if not expected:
        return  # API_KEY not set → auth disabled (local dev)
    if key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
```

Applied via `dependencies=[Depends(_verify_api_key)]` on every `@app.get("/api/...")` route. Static page routes (`/`, `/globe`, `/polarization`, etc.) are not protected.

### Fallback

If `API_KEY` is not set, the dependency is a no-op. Local dev and the Railway pipeline (which does not call the API) are unaffected.

### Out of scope

Per-key rate limits, key rotation, multiple keys, provisioning UI — all deferred.

---

## Manual steps (not in code)

1. **Neon project setup:** create project, enable `pgvector`, run migration (see Section 1)
2. **Upstash Redis:** create an Upstash Redis database, copy REST URL and token to Vercel env vars
3. **Vercel env vars:** set `EMBEDDING_BACKEND=fastembed`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, `API_KEY`, update `DATABASE_URL` to Neon pooled DSN
4. **Railway env vars:** update `DATABASE_URL` to Neon direct DSN, set `EMBEDDING_BACKEND=sentence-transformers` (or leave unset)
5. **Gate validation:** run `scripts/compare_embeddings.py` locally, test-deploy to Vercel preview for bundle size check

---

## Out of scope

- Landing page redesign (deferred from Phase 1)
- Per-user API keys with storage, rotation, revocation
- Re-embedding existing articles (only needed if fastembed Gate 1 fails and is later reconsidered)
- Enable `/docs`, `/openapi.json`
