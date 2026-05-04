# API / Pipeline Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the single Railway container into a Railway pipeline worker (`Dockerfile.pipeline`) and a Vercel serverless API (`api/index.py`), with `/api/search` returning HTTP 501 on Vercel until fastembed is validated (Phase 2).

**Architecture:** `sentence_transformers` import in `embeddings/embed.py` is made lazy so `app.py` loads without it. A module-level flag `_SEARCH_AVAILABLE` gates the search endpoint. The connection pool is sized down (`min_size=0`) when `VERCEL=1` is set to avoid exhausting Railway Postgres connections across serverless instances.

**Tech Stack:** FastAPI, psycopg3, uv, Docker (Railway), Vercel Python Runtime (`@vercel/python`), pytest/monkeypatch.

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/gdelt_event_pipeline/embeddings/embed.py` | Lazy-import `sentence_transformers` inside `load_model()` |
| Modify | `src/gdelt_event_pipeline/api/app.py` | Add `_SEARCH_AVAILABLE` flag + 501 guard; add serverless pool sizing |
| Create | `tests/api/__init__.py` | Test package marker |
| Create | `tests/api/test_app.py` | Tests for 501 guard and pool sizing |
| Create | `requirements.txt` | Vercel deps — pinned, no `sentence-transformers` |
| Create | `api/__init__.py` | Python package marker for Vercel entry directory |
| Create | `api/index.py` | Vercel ASGI entry — adds `src/` to `sys.path`, exports `app` |
| Create | `vercel.json` | Vercel build config: Python 3.11, routes `/*` → `api/index.py` |
| Create | `Dockerfile.pipeline` | Railway image — pipeline only, no uvicorn |
| Modify | `railway.toml` | Point to `Dockerfile.pipeline`; remove healthcheck |

---

## Task 1: Create the feature branch

**Files:** none (git only)

- [ ] **Create and switch to the feature branch**

  ```bash
  git checkout -b split-api-pipeline-vercel-railway
  ```

- [ ] **Verify you are on the right branch**

  ```bash
  git branch --show-current
  ```

  Expected output: `split-api-pipeline-vercel-railway`

---

## Task 2: Fix lazy import in `embed.py`

**Files:**
- Modify: `src/gdelt_event_pipeline/embeddings/embed.py`
- Test: `tests/embeddings/test_embed.py`

The current module-level `from sentence_transformers import SentenceTransformer` causes `app.py` to fail on Vercel startup with `ModuleNotFoundError`. Moving it inside `load_model()` makes the module safe to import without the package installed.

- [ ] **Write the failing test** — append this class to `tests/embeddings/test_embed.py`:

  ```python
  class TestLazyImport:
      def test_module_loads_without_sentence_transformers(self, monkeypatch):
          """embed.py must not import sentence_transformers at module load time.
          After the lazy-import fix, importing the module with sentence_transformers
          blocked must succeed without raising ModuleNotFoundError."""
          import sys

          # Block sentence_transformers from being importable
          monkeypatch.setitem(sys.modules, "sentence_transformers", None)
          # Remove the cached embed module so Python re-imports it fresh
          monkeypatch.delitem(
              sys.modules, "gdelt_event_pipeline.embeddings.embed", raising=False
          )

          # Must NOT raise ModuleNotFoundError / ImportError
          import gdelt_event_pipeline.embeddings.embed  # noqa: F401
  ```

- [ ] **Run the test — confirm it fails**

  ```bash
  uv run pytest tests/embeddings/test_embed.py::TestLazyImport -v
  ```

  Expected: `FAILED` — `ModuleNotFoundError: No module named 'sentence_transformers'`

- [ ] **Apply the fix to `embed.py`**

  Open `src/gdelt_event_pipeline/embeddings/embed.py`. The file currently starts with:

  ```python
  """Embedding generation using sentence-transformers."""

  from __future__ import annotations

  import logging

  from sentence_transformers import SentenceTransformer

  logger = logging.getLogger(__name__)

  _model: SentenceTransformer | None = None


  def load_model(model_name: str) -> SentenceTransformer:
      """Load (and cache) a sentence-transformers model."""
      global _model
      if _model is None or _model.model_card_data.model_id != model_name:
          logger.info("Loading embedding model: %s", model_name)
          _model = SentenceTransformer(model_name)
      return _model
  ```

  Replace it with (remove the module-level import, add it inside `load_model`):

  ```python
  """Embedding generation using sentence-transformers."""

  from __future__ import annotations

  import logging

  logger = logging.getLogger(__name__)

  _model = None


  def load_model(model_name: str):
      """Load (and cache) a sentence-transformers model."""
      from sentence_transformers import SentenceTransformer  # lazy: not needed at import time

      global _model
      if _model is None or _model.model_card_data.model_id != model_name:
          logger.info("Loading embedding model: %s", model_name)
          _model = SentenceTransformer(model_name)
      return _model
  ```

  The `_model` annotation changes from `SentenceTransformer | None` to bare `= None` because `SentenceTransformer` is no longer imported at module scope. `from __future__ import annotations` was only needed to defer evaluation of the annotation — removing the annotation removes the dependency on it.

- [ ] **Run the full embeddings test suite — confirm everything passes**

  ```bash
  uv run pytest tests/embeddings/ -v
  ```

  Expected: all tests `PASSED` (the existing `test_single_text`, `test_multiple_texts` etc. are unaffected because `load_model` still works when `sentence_transformers` is installed).

- [ ] **Commit**

  ```bash
  git add src/gdelt_event_pipeline/embeddings/embed.py tests/embeddings/test_embed.py
  git commit -m "fix: make sentence_transformers import lazy in embed.py"
  ```

---

## Task 3: Add 501 guard and serverless pool sizing to `app.py`

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_app.py`

Two separate changes to `app.py`:
1. `_SEARCH_AVAILABLE` flag + 501 guard on `/api/search`
2. Detect `VERCEL=1` in the lifespan and use `min_size=0, max_size=2`

- [ ] **Create the test package marker**

  Create `tests/api/__init__.py` as an empty file.

- [ ] **Write the failing tests** — create `tests/api/test_app.py`:

  ```python
  """Tests for app.py behaviour specific to the Vercel deployment split."""

  import pytest
  from unittest.mock import patch
  from fastapi.testclient import TestClient
  import gdelt_event_pipeline.api.app as app_module


  @pytest.fixture
  def client_no_db():
      """TestClient with DB lifecycle mocked out so no real connection is needed."""
      with patch("gdelt_event_pipeline.api.app.init_pool"), \
           patch("gdelt_event_pipeline.api.app.close_pool"), \
           patch("gdelt_event_pipeline.api.app._ensure_schema"):
          with TestClient(app_module.app) as client:
              yield client


  class TestSearchGuard:
      def test_search_returns_501_when_search_unavailable(self, client_no_db, monkeypatch):
          """When _SEARCH_AVAILABLE is False the endpoint must return HTTP 501."""
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
          response = client_no_db.get("/api/search?q=test")
          assert response.status_code == 501
          assert "not available" in response.json()["detail"].lower()

      def test_search_available_true_when_sentence_transformers_installed(self):
          """In the dev / Railway environment sentence_transformers is installed
          so the flag must be True."""
          assert app_module._SEARCH_AVAILABLE is True


  class TestServerlessPoolSizing:
      def test_pool_uses_serverless_sizes_when_vercel_env_set(self, monkeypatch):
          """When VERCEL=1 the lifespan must call init_pool with min_size=0, max_size=2."""
          monkeypatch.setenv("VERCEL", "1")
          with patch("gdelt_event_pipeline.api.app.init_pool") as mock_init, \
               patch("gdelt_event_pipeline.api.app.close_pool"), \
               patch("gdelt_event_pipeline.api.app._ensure_schema"):
              with TestClient(app_module.app):
                  pass
          mock_init.assert_called_once()
          assert mock_init.call_args.kwargs["min_size"] == 0
          assert mock_init.call_args.kwargs["max_size"] == 2

      def test_pool_uses_standard_sizes_without_vercel_env(self, monkeypatch):
          """Without VERCEL env the lifespan must call init_pool with min_size=2, max_size=10."""
          monkeypatch.delenv("VERCEL", raising=False)
          with patch("gdelt_event_pipeline.api.app.init_pool") as mock_init, \
               patch("gdelt_event_pipeline.api.app.close_pool"), \
               patch("gdelt_event_pipeline.api.app._ensure_schema"):
              with TestClient(app_module.app):
                  pass
          mock_init.assert_called_once()
          assert mock_init.call_args.kwargs["min_size"] == 2
          assert mock_init.call_args.kwargs["max_size"] == 10
  ```

- [ ] **Run the tests — confirm they fail**

  ```bash
  uv run pytest tests/api/test_app.py -v
  ```

  Expected: all 4 tests `FAILED`:
  - `test_search_returns_501_when_search_unavailable` — `_SEARCH_AVAILABLE` attribute not found
  - `test_search_available_true_when_sentence_transformers_installed` — same
  - `test_pool_uses_serverless_sizes_when_vercel_env_set` — `init_pool` called without kwargs
  - `test_pool_uses_standard_sizes_without_vercel_env` — same

- [ ] **Apply change 1 — add `_SEARCH_AVAILABLE` flag**

  In `src/gdelt_event_pipeline/api/app.py`, find the block of module-level imports (after all the `from gdelt_event_pipeline...` imports, before `STATIC_DIR`). Add these lines immediately before `STATIC_DIR = ...`:

  ```python
  import os

  # Detect whether sentence_transformers is available at startup.
  # On Vercel it is not installed (see requirements.txt), so /api/search returns 501.
  try:
      import sentence_transformers as _st_check  # noqa: F401
      _SEARCH_AVAILABLE = True
  except ImportError:
      _SEARCH_AVAILABLE = False
  ```

  Note: `import os` is not currently in `app.py` — it must be added here (it is also needed for the pool sizing change below).

- [ ] **Apply change 2 — guard the search endpoint**

  In `src/gdelt_event_pipeline/api/app.py`, find the `/api/search` endpoint function:

  ```python
  @app.get("/api/search", response_model=SearchResponse)
  def search(
      q: str = Query(..., description="Search query text"),
      ...
  ):
      """Hybrid semantic + keyword search over articles and clusters."""
      filters = SearchFilters(
  ```

  Add the guard as the **first two lines** of the function body (before `filters = SearchFilters(...)`):

  ```python
      if not _SEARCH_AVAILABLE:
          raise HTTPException(
              status_code=501,
              detail="Semantic search is not available in this deployment.",
          )
  ```

- [ ] **Apply change 3 — serverless pool sizing in the lifespan**

  In `src/gdelt_event_pipeline/api/app.py`, find the `lifespan` function:

  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
      settings = get_settings()
      init_pool(settings.db)
      _ensure_schema()
      yield
      close_pool()
  ```

  Replace the `init_pool(settings.db)` call with:

  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
      settings = get_settings()
      _is_serverless = bool(os.environ.get("VERCEL"))
      init_pool(
          settings.db,
          min_size=0 if _is_serverless else 2,
          max_size=2 if _is_serverless else 10,
      )
      _ensure_schema()
      yield
      close_pool()
  ```

- [ ] **Run the tests — confirm they all pass**

  ```bash
  uv run pytest tests/api/test_app.py -v
  ```

  Expected: all 4 tests `PASSED`.

- [ ] **Run the full test suite to confirm no regressions**

  ```bash
  uv run pytest -v
  ```

  Expected: all tests `PASSED`.

- [ ] **Commit**

  ```bash
  git add src/gdelt_event_pipeline/api/app.py tests/api/__init__.py tests/api/test_app.py
  git commit -m "feat: add search 501 guard and serverless pool sizing for Vercel"
  ```

---

## Task 4: Create `requirements.txt` (Vercel dependencies)

**Files:**
- Create: `requirements.txt`

Pinned exact versions from `uv.lock`. Does not include `sentence-transformers` or `torch`.

- [ ] **Create `requirements.txt` at the repo root**

  ```
  # Vercel API dependencies — pinned from uv.lock
  # sentence-transformers is intentionally excluded: /api/search returns 501 on Vercel (Phase 2)
  psycopg[binary]==3.3.3
  psycopg-pool==3.3.0
  pgvector==0.4.2
  fastapi==0.135.2
  uvicorn[standard]==0.42.0
  python-dotenv==1.2.2
  starlette==1.0.0
  pydantic==2.12.5
  pydantic-core==2.41.5
  anyio==4.12.1
  h11==0.16.0
  click==8.3.1
  typing-extensions==4.15.0
  annotated-types==0.7.0
  httptools==0.7.1
  uvloop==0.22.1
  watchfiles==1.1.1
  websockets==16.0
  idna==3.11
  certifi==2026.2.25
  ```

- [ ] **Verify the file lists no sentence-transformers or torch**

  ```bash
  grep -i "sentence\|torch" requirements.txt
  ```

  Expected: no output (no matches).

- [ ] **Commit**

  ```bash
  git add requirements.txt
  git commit -m "feat: add requirements.txt with pinned Vercel-only dependencies"
  ```

---

## Task 5: Create `api/index.py` (Vercel entry point)

**Files:**
- Create: `api/__init__.py`
- Create: `api/index.py`

Vercel's Python runtime looks for an `app` ASGI object in the module at the path specified by `vercel.json`. The `api/` directory must be a Python package.

- [ ] **Create `api/__init__.py`** as an empty file

- [ ] **Create `api/index.py`**

  ```python
  """Vercel ASGI entry point.

  Adds src/ to sys.path so the gdelt_event_pipeline package is importable
  without a package install step, then re-exports the FastAPI app object
  that Vercel's runtime uses as the ASGI handler.
  """

  import sys
  from pathlib import Path

  # src/ is not on sys.path in Vercel's runtime environment.
  # Insert it so `from gdelt_event_pipeline...` imports resolve correctly.
  sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

  from gdelt_event_pipeline.api.app import app  # noqa: E402 — intentional late import

  __all__ = ["app"]
  ```

- [ ] **Verify the import resolves locally**

  ```bash
  uv run python -c "import sys; sys.path.insert(0, 'src'); from gdelt_event_pipeline.api.app import app; print(app.title)"
  ```

  Expected output: `GDELT Pulse API`

- [ ] **Commit**

  ```bash
  git add api/__init__.py api/index.py
  git commit -m "feat: add Vercel ASGI entry point api/index.py"
  ```

---

## Task 6: Create `vercel.json`

**Files:**
- Create: `vercel.json`

Routes all requests to `api/index.py`. Specifies Python 3.11 explicitly — Vercel defaults to 3.9 if omitted.

- [ ] **Create `vercel.json` at the repo root**

  ```json
  {
    "functions": {
      "api/index.py": {
        "runtime": "python3.11",
        "maxDuration": 30
      }
    },
    "routes": [
      { "src": "/(.*)", "dest": "/api/index.py" }
    ]
  }
  ```

  `maxDuration: 30` sets a 30-second function timeout (Vercel Hobby default is 10s; some analytics queries are heavy).

- [ ] **Verify the JSON is valid**

  ```bash
  python -m json.tool vercel.json > /dev/null && echo "valid"
  ```

  Expected: `valid`

- [ ] **Commit**

  ```bash
  git add vercel.json
  git commit -m "feat: add vercel.json with Python 3.11 runtime and routing"
  ```

---

## Task 7: Create `Dockerfile.pipeline`

**Files:**
- Create: `Dockerfile.pipeline`

Identical to the existing `Dockerfile` except: no `EXPOSE`, and `CMD` runs only `runner.py` (no uvicorn).

- [ ] **Create `Dockerfile.pipeline`**

  ```dockerfile
  FROM python:3.11-slim AS base

  # Prevent Python from buffering stdout/stderr (important for Docker logs)
  ENV PYTHONUNBUFFERED=1

  # Install uv for fast dependency management
  COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

  WORKDIR /app

  # Install dependencies first (layer caching)
  COPY pyproject.toml uv.lock README.md ./
  RUN uv sync --frozen --no-dev --no-install-project

  # Copy source code and schema
  COPY src/ src/
  COPY sql/ sql/
  RUN uv sync --frozen --no-dev

  # Download the embedding model at build time so it's baked into the image
  # (avoids downloading ~90MB on every container start)
  RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

  # Pipeline has no HTTP port — no EXPOSE

  CMD ["uv", "run", "python", "-m", "gdelt_event_pipeline.runner"]
  ```

- [ ] **Confirm the diff between the two Dockerfiles is minimal**

  ```bash
  diff Dockerfile Dockerfile.pipeline
  ```

  Expected: only the `EXPOSE 8000` line and the `CMD` line differ.

- [ ] **Commit**

  ```bash
  git add Dockerfile.pipeline
  git commit -m "feat: add Dockerfile.pipeline for Railway pipeline-only service"
  ```

---

## Task 8: Update `railway.toml`

**Files:**
- Modify: `railway.toml`

Point Railway at the new pipeline Dockerfile and remove the HTTP healthcheck — the pipeline process has no HTTP port so Railway would fail trying to reach `/api/stats`.

- [ ] **Replace the content of `railway.toml`**

  Current content:
  ```toml
  [build]
  builder = "dockerfile"
  dockerfilePath = "Dockerfile"

  [deploy]
  healthcheckPath = "/api/stats"
  healthcheckTimeout = 300
  restartPolicyType = "always"
  ```

  New content:
  ```toml
  [build]
  builder = "dockerfile"
  dockerfilePath = "Dockerfile.pipeline"

  [deploy]
  restartPolicyType = "always"
  ```

- [ ] **Verify the TOML is valid**

  ```bash
  python -c "import tomllib; tomllib.load(open('railway.toml','rb')); print('valid')"
  ```

  Expected: `valid`

- [ ] **Commit**

  ```bash
  git add railway.toml
  git commit -m "feat: update railway.toml to use Dockerfile.pipeline, remove HTTP healthcheck"
  ```

---

## Task 9: End-to-end smoke test

**Files:** none (verification only)

Verify the full Vercel import chain works: `api/index.py` → `src/gdelt_event_pipeline/api/app.py` imports without `sentence_transformers`, `_SEARCH_AVAILABLE` is `False` when blocked, and all existing tests still pass.

- [ ] **Run the full test suite**

  ```bash
  uv run pytest -v
  ```

  Expected: all tests `PASSED`. The `tests/embeddings/test_embed.py::TestLazyImport` and all `tests/api/test_app.py` tests must be green.

- [ ] **Verify the Vercel entry import chain resolves**

  ```bash
  uv run python -c "
  import importlib.util
  spec = importlib.util.spec_from_file_location('api.index', 'api/index.py')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  assert hasattr(mod, 'app'), 'app object not exported from api/index.py'
  print(f'OK: api/index.py exports app={mod.app.title!r}')
  "
  ```

  Expected output: `OK: api/index.py exports app='GDELT Pulse API'`

- [ ] **Simulate the Vercel environment: block sentence_transformers and verify the flag**

  ```bash
  uv run python -c "
  import sys
  sys.modules['sentence_transformers'] = None
  # Clear any cached gdelt_event_pipeline modules
  to_del = [k for k in sys.modules if 'gdelt_event_pipeline' in k]
  for k in to_del:
      del sys.modules[k]
  sys.path.insert(0, 'src')
  import gdelt_event_pipeline.api.app as m
  assert m._SEARCH_AVAILABLE is False, f'Expected False, got {m._SEARCH_AVAILABLE}'
  print('OK: _SEARCH_AVAILABLE is False when sentence_transformers is absent')
  "
  ```

  Expected output: `OK: _SEARCH_AVAILABLE is False when sentence_transformers is absent`

- [ ] **Verify the existing Dockerfile still builds** (optional, run only if Docker is available)

  ```bash
  docker build -f Dockerfile -t gdelt-pulse-api-local --no-cache .
  ```

  Expected: build succeeds. This confirms the local dev / docker-compose path is not broken.

- [ ] **Final commit summary check**

  ```bash
  git log --oneline -8
  ```

  Expected to see (newest first):
  ```
  feat: update railway.toml to use Dockerfile.pipeline, remove HTTP healthcheck
  feat: add Dockerfile.pipeline for Railway pipeline-only service
  feat: add vercel.json with Python 3.11 runtime and routing
  feat: add Vercel ASGI entry point api/index.py
  feat: add requirements.txt with pinned Vercel-only dependencies
  feat: add search 501 guard and serverless pool sizing for Vercel
  fix: make sentence_transformers import lazy in embed.py
  ```

---

## Deploy reference (manual steps — not in this plan)

Once the branch is merged:

**Railway (pipeline):**
```
# In the Railway dashboard: update service → Dockerfile Path → Dockerfile.pipeline
# The service must have: DATABASE_URL, PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE, PIPELINE_INTERVAL
# Remove CORS_ORIGINS if it was only used by the API
```

**Vercel (API):**
```
# vercel --prod
# Or: connect repo in Vercel dashboard, set Framework Preset = Other
# Required env vars: DATABASE_URL, CORS_ORIGINS
# VERCEL=1 is injected automatically
```

**DNS:**
```
# After Vercel deployment succeeds, add to unbubblehub DNS zone:
# gdelt-pulse.unbubblehub.org  CNAME  <vercel-deployment>.vercel.app
# Then add the custom domain in Vercel project settings
```
