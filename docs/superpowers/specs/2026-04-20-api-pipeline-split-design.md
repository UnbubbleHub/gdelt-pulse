# API / Pipeline Split — Phase 1 Design

**Date:** 2026-04-20
**Branch:** `split-api-pipeline-vercel-railway`
**Scope:** Structural split only. No DB migration, no API keys, no landing page (Phase 2).

---

## Goal

Decouple the two processes that currently run in one Railway container:

| Process | What it does | Target runtime |
|---|---|---|
| Pipeline (`runner.py`) | Ingest GDELT → scrape titles → embed → cluster, loops every 15 min | Railway — long-running worker |
| API (`app.py`) | Serve HTTP requests against the database | Vercel — stateless serverless functions |

Both continue to use the same PostgreSQL database (Railway Postgres, unchanged in Phase 1).

---

## Deployment topology

```
Railway service (pipeline)
  └── Dockerfile.pipeline
        CMD: uv run python -m gdelt_event_pipeline.runner
        Deps: full pyproject.toml (sentence-transformers stays here)

Vercel project (api)
  └── api/index.py          ← ASGI entry point
      requirements.txt      ← API-only deps, NO sentence-transformers
      vercel.json           ← routes + Python 3.11 runtime config
        Routes: /* → api/index.py

Shared
  └── PostgreSQL on Railway (DATABASE_URL injected into both)
```

---

## Files changed

### New files

| File | Purpose |
|---|---|
| `Dockerfile.pipeline` | Railway image — pipeline process only |
| `api/index.py` | Vercel ASGI entry — adds `src/` to `sys.path`, imports `app` |
| `vercel.json` | Vercel build + routing config |
| `requirements.txt` | Vercel deps — excludes `sentence-transformers` |

### Modified files

| File | Change |
|---|---|
| `railway.toml` | `dockerfilePath = "Dockerfile.pipeline"`, healthcheck removed (no HTTP port) |
| `src/gdelt_event_pipeline/embeddings/embed.py` | Move `from sentence_transformers import SentenceTransformer` inside `load_model()` (lazy import) so the module loads without the package present |
| `src/gdelt_event_pipeline/api/app.py` | (1) Detect `sentence_transformers` availability at startup; `/api/search` returns HTTP 501 when unavailable. (2) Use `min_size=0, max_size=2` for the connection pool when `VERCEL=1` env var is set. |

### Unchanged

- `Dockerfile` — kept for local `docker-compose` dev
- `docker-compose.yml` — unchanged (local dev runs both pipeline + API together)
- `pyproject.toml` — unchanged (Railway/uv uses this; includes sentence-transformers)
- All pipeline code (`runner.py`, `ingestion/`, `embeddings/`, `clustering/`)
- All API endpoints except `/api/search`

---

## Detail: lazy import fix

`embed.py` currently imports `sentence_transformers` at module level. Because `app.py` → `query/search.py` → `embeddings/embed.py` all use top-level imports, Vercel fails on startup with `ModuleNotFoundError`.

Fix: move the import inside the function that uses it.

```python
# embeddings/embed.py — before
from sentence_transformers import SentenceTransformer   # module-level, breaks Vercel

# after
def load_model(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer  # lazy, only when called
    ...
```

`from __future__ import annotations` is already present in `embed.py`, so the type annotation `SentenceTransformer | None` on `_model` remains a string at runtime — no change needed there.

---

## Detail: search 501 guard in app.py

At module load, check whether `sentence_transformers` is importable:

```python
try:
    import sentence_transformers as _st_check  # noqa: F401
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False
```

At the top of the `/api/search` endpoint:

```python
if not _SEARCH_AVAILABLE:
    raise HTTPException(
        status_code=501,
        detail="Semantic search is not available in this deployment.",
    )
```

This makes the 501 explicit and prevents silent misbehaviour.

---

## Detail: connection pool on Vercel

`psycopg_pool.ConnectionPool(min_size=2)` keeps 2 persistent connections alive per process. On Vercel, each function instance creates its own pool; many concurrent instances exhaust the DB connection limit.

Fix: detect Vercel at startup and use `min_size=0` (no idle connections held open):

```python
import os
_is_serverless = bool(os.environ.get("VERCEL"))
init_pool(settings.db,
          min_size=0 if _is_serverless else 2,
          max_size=2 if _is_serverless else 10)
```

Vercel automatically sets `VERCEL=1` in all its runtime environments.

---

## Detail: Vercel entrypoint and PYTHONPATH

The package lives under `src/`. Rather than relying on `PYTHONPATH` being set externally, `api/index.py` adds it programmatically — explicit and portable:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gdelt_event_pipeline.api.app import app  # noqa: E402 — intentional late import
```

Vercel looks for the `app` name in the module and uses it as the ASGI handler.

`vercel.json` must specify Python 3.11 explicitly (Vercel defaults to 3.9 if unset):

```json
{
  "functions": {
    "api/index.py": { "runtime": "python3.11" }
  },
  "routes": [
    { "src": "/(.*)", "dest": "/api/index.py" }
  ]
}
```

---

## Detail: static files on Vercel

`app.py` mounts static files via:
```python
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

Because we use `sys.path.insert` (source directory, not an installed wheel), `Path(__file__)` resolves to `src/gdelt_event_pipeline/api/app.py` and `static/` is its sibling. Vercel clones the full repo, so the directory exists at runtime. No change required.

---

## Detail: `_ensure_schema()` on Vercel

`app.py`'s lifespan calls `_ensure_schema()`. It resolves `sql/001_schema.sql` as `Path(__file__).resolve().parents[3] / "sql"` which, from `src/gdelt_event_pipeline/api/app.py`, is the repo root — present on Vercel. However, since the DB schema already exists from the Railway deployment, the early-exit `SELECT EXISTS(...)` check returns `true` immediately and the function is a no-op. No change required.

---

## Detail: requirements.txt (Vercel)

Vercel installs this file. It must list all API runtime dependencies and explicitly exclude `sentence-transformers` (and therefore `torch`, which is pulled transitively).

```
psycopg[binary]>=3.2.0
psycopg-pool>=3.2.0
pgvector>=0.3.0
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
python-dotenv>=1.0.0
```

Shown with `>=` for readability. The implementation step must pin exact versions sourced from `uv.lock` to guarantee reproducibility across Railway and Vercel builds.

`sentence-transformers` is not listed → not installed → `/api/search` returns 501.
`fastembed` is not listed → deferred to Phase 2 (after compatibility validation).

---

## Detail: railway.toml

Remove the healthcheck — the pipeline process has no HTTP port and Railway will error if it tries to connect.

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile.pipeline"

[deploy]
restartPolicyType = "always"
```

---

## Known limitation: rate limiting

The in-memory `_rate_limit_store` (a `defaultdict(list)`) is per-process. On Vercel, each function instance has its own memory; limits are not shared across instances. This means the rate limiter is ineffective (not incorrectly strict, just porous).

For Phase 1 this is acceptable — the API is not yet public. Phase 2 replaces this with Upstash Redis.

No code change in Phase 1 for rate limiting.

---

## Manual steps (not in code)

These are required before go-live but are not in the implementation plan:

1. **Vercel project**: create project, link to repo, set env vars:
   - `DATABASE_URL` — Railway Postgres public URL
   - `CORS_ORIGINS` — Vercel deployment URL(s)
   - `VERCEL=1` is set automatically by Vercel; no manual action needed

2. **Railway service**: after merge, update the service to point to `Dockerfile.pipeline`. The existing Railway service keeps its DATABASE_URL and other pipeline env vars.

3. **DNS** (after Vercel deployment succeeds): add `gdelt-pulse.unbubblehub.org` CNAME pointing to the Vercel deployment URL in the unbubblehub zone.

---

## Out of scope (Phase 2)

- Neon migration
- `fastembed` swap and `/api/search` re-enablement (requires: compatibility validation script, ONNX model bundling in Vercel build)
- Upstash rate limiting
- API key auth
- Landing page redesign
- Enable `/docs`, `/openapi.json`
- Globe UI removal
