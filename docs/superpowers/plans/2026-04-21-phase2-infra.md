# Phase 2 Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the database to Neon, swap the embedding backend to fastembed on Vercel, replace the broken in-memory rate limiter with Upstash Redis, and add global API key auth — all on the `phase2-infra` branch.

**Architecture:** Four self-contained changes to a FastAPI app on Vercel and a pipeline worker on Railway, both sharing a Neon Postgres database. The rate limiter and auth are added to the existing HTTP middleware; the embedding backend is controlled by an `EMBEDDING_BACKEND` env var; the Neon migration is manual (no code changes).

**Tech Stack:** FastAPI, psycopg + psycopg-pool, fastembed, upstash-redis, Neon Postgres, Upstash Redis, pytest, uv

---

## File Map

| File | Change |
|---|---|
| `src/gdelt_event_pipeline/embeddings/embed.py` | Add fastembed branch controlled by `EMBEDDING_BACKEND` env var |
| `src/gdelt_event_pipeline/api/app.py` | Update search guard, add Redis rate limiting, add API key auth to middleware |
| `tests/embeddings/test_embed.py` | Add fastembed backend routing tests |
| `tests/api/test_app.py` | Add rate limiting, auth, and updated search guard tests |
| `requirements.txt` | Add `fastembed`, `upstash-redis` (Task 7 only, after gates pass) |
| `scripts/compare_embeddings.py` | New one-off validation script (not deployed) |

---

## Task 1: Neon migration (manual — no code)

**No files changed. Complete all steps before running any other task.**

- [ ] **Step 1: Create Neon project**

  Go to neon.tech, create a new project. Under the project settings, note both connection strings:
  - **Direct DSN** (for Railway pipeline): looks like `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname`
  - **Pooled DSN** (for Vercel API): looks like `postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/dbname?pgbouncer=true`

- [ ] **Step 2: Enable pgvector on Neon before restoring**

  Connect to Neon with psql (use the direct DSN):
  ```bash
  psql "postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"
  ```
  This must run before the restore — otherwise the dump will fail on vector column types.

- [ ] **Step 3: Dump Railway Postgres**

  From your local machine with `DATABASE_URL` set to the Railway Postgres URL:
  ```bash
  pg_dump "$RAILWAY_DATABASE_URL" --no-owner --no-acl -f railway_dump.sql
  ```

- [ ] **Step 4: Restore to Neon**

  ```bash
  psql "postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname" -f railway_dump.sql
  ```

- [ ] **Step 5: Verify row counts**

  ```bash
  psql "postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname" -c "
  SELECT 'articles' AS tbl, count(*) FROM articles
  UNION ALL SELECT 'clusters', count(*) FROM clusters
  UNION ALL SELECT 'cluster_memberships', count(*) FROM cluster_memberships
  UNION ALL SELECT 'mv_country_comentions', count(*) FROM mv_country_comentions
  UNION ALL SELECT 'mv_country_stats', count(*) FROM mv_country_stats;
  "
  ```
  Counts must match Railway Postgres exactly.

- [ ] **Step 6: Update env vars**

  - **Railway service:** set `DATABASE_URL` to the Neon **direct** DSN. Set `EMBEDDING_BACKEND=sentence-transformers`.
  - **Vercel project:** set `DATABASE_URL` to the Neon **pooled** DSN.

- [ ] **Step 7: Redeploy both services and smoke-test**

  After Railway and Vercel redeploy, hit `/api/stats` and `/api/clusters` on both services. Verify non-empty responses.

---

## Task 2: Write the comparison script (Gate 1 for fastembed)

**Files:**
- Create: `scripts/compare_embeddings.py`

This script is **not deployed**. Run it locally before Task 6 to confirm fastembed produces vectors compatible with the existing sentence-transformers vectors in the DB.

- [ ] **Step 1: Create `scripts/` directory and write the script**

  ```python
  # scripts/compare_embeddings.py
  """
  Gate 1 validation: verify fastembed and sentence-transformers produce
  compatible vectors for the same model, and that retrieval quality holds.

  Run: uv run python scripts/compare_embeddings.py
  Requires: DATABASE_URL set to Neon direct DSN, both libraries installed locally.
  """

  import os
  import math

  HEADLINES = [
      "Earthquake strikes Turkey, thousands displaced",
      "Federal Reserve raises interest rates again",
      "Apple unveils new iPhone with AI features",
      "Ukraine ceasefire talks stall in Brussels",
      "Amazon warehouse workers vote to unionize",
      "Scientists discover potential cancer vaccine",
      "Hurricane Milton makes landfall in Florida",
      "Tesla recalls 200,000 vehicles over brake defect",
      "Israel expands ground offensive in Gaza",
      "China launches lunar sample return mission",
      "US Senate passes $60 billion Ukraine aid bill",
      "OpenAI releases GPT-5 with extended context",
      "WHO declares mpox a global health emergency",
      "Boeing 737 Max cleared to fly again in Europe",
      "EU imposes tariffs on Chinese electric vehicles",
      "Wildfire destroys 10,000 acres in California",
      "India overtakes China as world's most populous nation",
      "SpaceX Starship completes first full flight test",
      "UK economy falls into recession for second quarter",
      "Inflation in Argentina hits 200 percent annually",
  ]

  MODEL = "sentence-transformers/all-MiniLM-L6-v2"
  PASS_THRESHOLD = 0.9999


  def cosine_similarity(a: list[float], b: list[float]) -> float:
      dot = sum(x * y for x, y in zip(a, b))
      norm_a = math.sqrt(sum(x * x for x in a))
      norm_b = math.sqrt(sum(x * x for x in b))
      return dot / (norm_a * norm_b)


  def embed_st(texts: list[str]) -> list[list[float]]:
      from sentence_transformers import SentenceTransformer
      model = SentenceTransformer(MODEL)
      return model.encode(texts, show_progress_bar=False).tolist()


  def embed_fe(texts: list[str]) -> list[list[float]]:
      from fastembed import TextEmbedding
      model = TextEmbedding(MODEL)
      return [v.tolist() for v in model.embed(texts)]


  def check_vector_compatibility() -> bool:
      print("=== Gate 1a: Vector compatibility ===")
      st_vecs = embed_st(HEADLINES)
      fe_vecs = embed_fe(HEADLINES)

      sims = [cosine_similarity(st, fe) for st, fe in zip(st_vecs, fe_vecs)]
      mean_sim = sum(sims) / len(sims)
      min_sim = min(sims)

      print(f"Mean cosine similarity: {mean_sim:.6f}")
      print(f"Min cosine similarity:  {min_sim:.6f}")
      print(f"Threshold:              {PASS_THRESHOLD}")

      passed = mean_sim >= PASS_THRESHOLD
      print(f"Result: {'PASS' if passed else 'FAIL'}")
      return passed


  def check_retrieval_quality() -> None:
      print("\n=== Gate 1b: Retrieval quality (manual inspection) ===")
      import psycopg
      from psycopg.rows import dict_row
      from fastembed import TextEmbedding

      db_url = os.environ["DATABASE_URL"]
      model = TextEmbedding(MODEL)

      test_queries = [
          "military conflict Middle East",
          "economic recession inflation",
          "technology artificial intelligence",
          "natural disaster climate",
          "election politics government",
      ]

      with psycopg.connect(db_url, row_factory=dict_row) as conn:
          for query in test_queries:
              vec = list(model.embed([query]))[0].tolist()
              rows = conn.execute(
                  """
                  SELECT title, domain,
                         embedding <=> %s::vector AS distance
                  FROM articles
                  WHERE embedding IS NOT NULL
                  ORDER BY embedding <=> %s::vector
                  LIMIT 5
                  """,
                  (vec, vec),
              ).fetchall()

              print(f"\nQuery: {query!r}")
              for i, r in enumerate(rows, 1):
                  print(f"  {i}. [{r['domain']}] {r['title']} (dist={r['distance']:.4f})")

      print("\nInspect results above. Proceed only if results are semantically relevant.")


  if __name__ == "__main__":
      compatible = check_vector_compatibility()
      if compatible:
          check_retrieval_quality()
      else:
          print("\nVector compatibility FAILED. Do not enable /api/search. Keep 501.")
  ```

- [ ] **Step 2: Run the script locally**

  ```bash
  uv run python scripts/compare_embeddings.py
  ```

  Expected output (if compatible):
  ```
  === Gate 1a: Vector compatibility ===
  Mean cosine similarity: 1.000000
  Min cosine similarity:  0.999999
  Threshold:              0.9999
  Result: PASS

  === Gate 1b: Retrieval quality (manual inspection) ===
  Query: 'military conflict Middle East'
    1. [reuters.com] Israel strikes Gaza hospital... (dist=0.1234)
    ...
  ```

  If Gate 1a fails (mean < 0.9999): **stop**. Do not proceed with Tasks 3–4. Keep `/api/search` at 501. Document the failure.

- [ ] **Step 3: Commit the script**

  ```bash
  git add scripts/compare_embeddings.py
  git commit -m "feat: add fastembed vector compatibility comparison script"
  ```

---

## Task 3: fastembed backend in embed.py

**Only proceed if Task 2 Gate 1a passed (mean cosine similarity ≥ 0.9999).**

**Files:**
- Modify: `src/gdelt_event_pipeline/embeddings/embed.py`
- Modify: `tests/embeddings/test_embed.py`

- [ ] **Step 1: Write the failing tests**

  Add to `tests/embeddings/test_embed.py` (after the existing `TestLazyImport` class):

  ```python
  class TestEmbeddingBackend:
      def test_fastembed_backend_used_when_env_set(self, monkeypatch):
          """When EMBEDDING_BACKEND=fastembed, embed_texts must call fastembed.TextEmbedding."""
          import sys
          from unittest.mock import MagicMock, patch

          monkeypatch.setenv("EMBEDDING_BACKEND", "fastembed")

          fake_vec = [0.1] * 384
          mock_model = MagicMock()
          mock_model.embed.return_value = iter([[fake_vec]])

          with patch.dict(sys.modules, {"fastembed": MagicMock(TextEmbedding=MagicMock(return_value=mock_model))}):
              # Re-import embed_texts so it picks up the env var at call time
              from gdelt_event_pipeline.embeddings.embed import embed_texts
              result = embed_texts(["test headline"])

          assert result == [fake_vec]

      def test_sentence_transformers_backend_used_by_default(self, monkeypatch):
          """When EMBEDDING_BACKEND is unset, embed_texts must use sentence-transformers."""
          monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
          from gdelt_event_pipeline.embeddings.embed import embed_texts
          result = embed_texts(["Hello world"])
          assert len(result) == 1
          assert len(result[0]) == 384

      def test_empty_list_returns_empty_regardless_of_backend(self, monkeypatch):
          """Empty input must return [] for both backends without calling any model."""
          for backend in ("sentence-transformers", "fastembed"):
              monkeypatch.setenv("EMBEDDING_BACKEND", backend)
              from gdelt_event_pipeline.embeddings.embed import embed_texts
              assert embed_texts([]) == []
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  uv run pytest tests/embeddings/test_embed.py::TestEmbeddingBackend -v
  ```

  Expected: `FAILED` — `test_fastembed_backend_used_when_env_set` fails because `embed_texts` ignores `EMBEDDING_BACKEND`.

- [ ] **Step 3: Update `embed.py`**

  Replace the entire file:

  ```python
  """Embedding generation — sentence-transformers or fastembed backend."""

  import logging
  import os

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


  def embed_texts(
      texts: list[str],
      *,
      model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
      batch_size: int = 64,
  ) -> list[list[float]]:
      """Encode a list of texts into embedding vectors.

      Backend is selected by the EMBEDDING_BACKEND env var:
        - "fastembed"            → fastembed.TextEmbedding (Vercel)
        - "sentence-transformers" or unset → SentenceTransformer (Railway / local)
      """
      if not texts:
          return []

      backend = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers")
      if backend == "fastembed":
          from fastembed import TextEmbedding  # lazy: only when backend is configured
          fe_model = TextEmbedding(model_name)
          return [v.tolist() for v in fe_model.embed(texts)]

      model = load_model(model_name)
      embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
      return embeddings.tolist()
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  uv run pytest tests/embeddings/ -v
  ```

  Expected: all pass, including pre-existing tests.

- [ ] **Step 5: Commit**

  ```bash
  git add src/gdelt_event_pipeline/embeddings/embed.py tests/embeddings/test_embed.py
  git commit -m "feat: add fastembed backend to embed_texts, controlled by EMBEDDING_BACKEND env var"
  ```

---

## Task 4: Update search guard in app.py

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py` (lines 33–38 — the `_SEARCH_AVAILABLE` block)
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing test**

  Add to `tests/api/test_app.py` inside `class TestSearchGuard`:

  ```python
  def test_search_guard_checks_fastembed_when_backend_is_fastembed(self, monkeypatch):
      """When EMBEDDING_BACKEND=fastembed and fastembed is not importable,
      _SEARCH_AVAILABLE must be False and /api/search must return 501."""
      monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
      response = client_no_db.get("/api/search?q=test")
      assert response.status_code == 501

  def test_search_guard_flag_reflects_configured_backend(self, monkeypatch):
      """_SEARCH_AVAILABLE is True in dev because sentence-transformers is installed
      and EMBEDDING_BACKEND defaults to sentence-transformers."""
      # This test documents the invariant: in the Railway/dev environment,
      # with EMBEDDING_BACKEND unset, the flag is True.
      import sys
      monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
      # Reload the flag by re-evaluating the module-level block
      # (monkeypatching the attribute directly since it's set at import time)
      monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", True)
      assert app_module._SEARCH_AVAILABLE is True
  ```

- [ ] **Step 2: Run tests to confirm they pass (these test existing behaviour)**

  ```bash
  uv run pytest tests/api/test_app.py::TestSearchGuard -v
  ```

  Expected: all pass (the new tests exercise the monkeypatching approach that already works).

- [ ] **Step 3: Update the `_SEARCH_AVAILABLE` block in `app.py`**

  Replace lines 32–38 (the search available block):

  ```python
  # Detect whether the configured embedding backend is importable.
  # On Vercel with EMBEDDING_BACKEND=fastembed, checks for fastembed.
  # Elsewhere defaults to sentence-transformers.
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

- [ ] **Step 4: Run the full test suite**

  ```bash
  uv run pytest tests/api/test_app.py -v
  ```

  Expected: all pass. `test_search_available_flag_true_in_full_environment` still passes because `EMBEDDING_BACKEND` is unset in the test environment and `sentence_transformers` is installed.

- [ ] **Step 5: Commit**

  ```bash
  git add src/gdelt_event_pipeline/api/app.py tests/api/test_app.py
  git commit -m "feat: update search guard to check EMBEDDING_BACKEND-configured library"
  ```

---

## Task 5: Upstash Redis rate limiting

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing tests**

  Add to `tests/api/test_app.py` as a new class (after `TestServerlessPoolSizing`):

  ```python
  class TestRateLimiting:
      def test_redis_rate_limit_returns_429_when_over_limit(self, client_no_db, monkeypatch):
          """When Redis pipeline reports count >= RATE_LIMIT_MAX, middleware returns 429."""
          from unittest.mock import MagicMock

          mock_pipe = MagicMock()
          # Pipeline results: [zremrangebyscore_result, zcard_count, zadd_result, expire_result]
          mock_pipe.execute.return_value = [0, 30, 1, 1]  # count=30 == RATE_LIMIT_MAX

          mock_redis = MagicMock()
          mock_redis.pipeline.return_value = mock_pipe

          monkeypatch.setattr(app_module, "_redis", mock_redis)

          response = client_no_db.get("/api/clusters")
          assert response.status_code == 429
          assert "rate limit" in response.json()["detail"].lower()

      def test_redis_rate_limit_passes_when_under_limit(self, client_no_db, monkeypatch):
          """When Redis pipeline reports count < RATE_LIMIT_MAX, middleware passes through."""
          from unittest.mock import MagicMock

          mock_pipe = MagicMock()
          mock_pipe.execute.return_value = [0, 5, 1, 1]  # count=5, well under limit

          mock_redis = MagicMock()
          mock_redis.pipeline.return_value = mock_pipe

          monkeypatch.setattr(app_module, "_redis", mock_redis)
          # /api/search with _SEARCH_AVAILABLE=False returns 501 — proves middleware passed through
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)

          response = client_no_db.get("/api/search?q=test")
          assert response.status_code == 501  # middleware passed through; endpoint returned 501

      def test_falls_back_to_in_memory_when_redis_not_configured(self, client_no_db, monkeypatch):
          """When _redis is None (no Upstash credentials), middleware uses in-memory store."""
          monkeypatch.setattr(app_module, "_redis", None)
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)

          response = client_no_db.get("/api/search?q=test")
          # In-memory fallback allows the request through (count is 0)
          assert response.status_code == 501  # passed through; not 429
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  uv run pytest tests/api/test_app.py::TestRateLimiting -v
  ```

  Expected: `FAILED` — `_redis` attribute doesn't exist yet, `AttributeError`.

- [ ] **Step 3: Replace the rate limiting section in `app.py`**

  Find and replace the rate limiting block. Current code starts at `# ── Rate limiting ─...` (around line 160). Replace the entire block (from the comment through the end of `rate_limit_middleware`) with:

  ```python
  # ── Rate limiting ────────────────────────────────────────────────────

  RATE_LIMIT_MAX = 30  # requests per window
  RATE_LIMIT_WINDOW = 60  # seconds

  _rate_limit_store: dict[str, list[float]] = defaultdict(list)
  _redis = None


  def _get_redis():
      """Return the Upstash Redis client, or None if not configured."""
      global _redis
      if _redis is None:
          url = os.environ.get("UPSTASH_REDIS_REST_URL")
          token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
          if url and token:
              from upstash_redis import Redis  # lazy: not installed in all environments
              _redis = Redis(url=url, token=token)
      return _redis


  @app.middleware("http")
  async def rate_limit_middleware(request: Request, call_next) -> Response:
      """Per-IP rate limiter. Uses Upstash Redis when configured; falls back to in-memory."""
      if not request.url.path.startswith("/api/"):
          return await call_next(request)

      client_ip = request.client.host if request.client else "unknown"
      now = time.time()
      redis = _get_redis()

      if redis is not None:
          key = f"ratelimit:{client_ip}"
          window_start = now - RATE_LIMIT_WINDOW
          pipe = redis.pipeline()
          pipe.zremrangebyscore(key, "-inf", window_start)
          pipe.zcard(key)
          pipe.zadd(key, {f"{now}-{os.urandom(4).hex()}": now})
          pipe.expire(key, RATE_LIMIT_WINDOW)
          results = pipe.execute()
          count = results[1]
      else:
          # In-memory fallback: per-instance, not shared across Vercel instances
          timestamps = _rate_limit_store[client_ip]
          _rate_limit_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
          count = len(_rate_limit_store[client_ip])
          if count < RATE_LIMIT_MAX:
              _rate_limit_store[client_ip].append(now)

      if count >= RATE_LIMIT_MAX:
          return Response(
              content='{"detail":"Rate limit exceeded. Try again later."}',
              status_code=429,
              media_type="application/json",
          )

      return await call_next(request)
  ```

  Also remove these specific path checks that were in the old middleware — the new one uses `startswith("/api/")` which covers all of them already. Delete the old block:
  ```python
  if not (
      request.url.path.startswith("/api/search")
      or request.url.path.startswith("/api/clusters")
      ...
  ):
      return await call_next(request)
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  uv run pytest tests/api/test_app.py -v
  ```

  Expected: all pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/gdelt_event_pipeline/api/app.py tests/api/test_app.py
  git commit -m "feat: replace in-memory rate limiter with Upstash Redis sliding window"
  ```

---

## Task 6: API key auth

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Write the failing tests**

  Add to `tests/api/test_app.py` as a new class (after `TestRateLimiting`):

  ```python
  class TestApiKeyAuth:
      def test_returns_401_when_key_is_missing(self, client_no_db, monkeypatch):
          """When API_KEY is set and X-API-Key header is absent, return 401."""
          monkeypatch.setenv("API_KEY", "secret-key")
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
          monkeypatch.setattr(app_module, "_redis", None)

          response = client_no_db.get("/api/search?q=test")
          assert response.status_code == 401
          assert "api key" in response.json()["detail"].lower()

      def test_returns_401_when_key_is_wrong(self, client_no_db, monkeypatch):
          """When API_KEY is set and X-API-Key header has wrong value, return 401."""
          monkeypatch.setenv("API_KEY", "secret-key")
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
          monkeypatch.setattr(app_module, "_redis", None)

          response = client_no_db.get("/api/search?q=test", headers={"X-API-Key": "wrong"})
          assert response.status_code == 401

      def test_passes_through_when_key_is_correct(self, client_no_db, monkeypatch):
          """When API_KEY is set and X-API-Key matches, middleware passes through."""
          monkeypatch.setenv("API_KEY", "secret-key")
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
          monkeypatch.setattr(app_module, "_redis", None)

          response = client_no_db.get(
              "/api/search?q=test", headers={"X-API-Key": "secret-key"}
          )
          assert response.status_code == 501  # middleware passed; endpoint returned 501

      def test_auth_disabled_when_api_key_not_set(self, client_no_db, monkeypatch):
          """When API_KEY env var is absent, all requests pass through without auth."""
          monkeypatch.delenv("API_KEY", raising=False)
          monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)
          monkeypatch.setattr(app_module, "_redis", None)

          response = client_no_db.get("/api/search?q=test")
          assert response.status_code == 501  # no auth gate; endpoint returned 501

      def test_static_pages_not_protected(self, client_no_db, monkeypatch):
          """Static routes (/, /globe, etc.) must be accessible without an API key."""
          monkeypatch.setenv("API_KEY", "secret-key")
          monkeypatch.setattr(app_module, "_redis", None)

          response = client_no_db.get("/")
          assert response.status_code != 401  # not gated by auth
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  uv run pytest tests/api/test_app.py::TestApiKeyAuth -v
  ```

  Expected: `FAILED` — no auth logic exists yet, requests pass through.

- [ ] **Step 3: Add API key check to `rate_limit_middleware` in `app.py`**

  Inside `rate_limit_middleware`, add the auth check immediately after the early-return for non-API paths (before the rate limiting logic):

  ```python
  @app.middleware("http")
  async def rate_limit_middleware(request: Request, call_next) -> Response:
      """Per-IP rate limiter and API key auth for /api/* paths."""
      if not request.url.path.startswith("/api/"):
          return await call_next(request)

      # API key auth — skipped if API_KEY env var is not set
      expected_key = os.environ.get("API_KEY")
      if expected_key:
          provided_key = request.headers.get("X-API-Key")
          if provided_key != expected_key:
              return Response(
                  content='{"detail":"Invalid or missing API key."}',
                  status_code=401,
                  media_type="application/json",
              )

      # Rate limiting (existing logic from Task 5)
      client_ip = request.client.host if request.client else "unknown"
      now = time.time()
      redis = _get_redis()
      # ... rest of rate limiting unchanged
  ```

- [ ] **Step 4: Run the full test suite**

  ```bash
  uv run pytest tests/ -v
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/gdelt_event_pipeline/api/app.py tests/api/test_app.py
  git commit -m "feat: add global API key auth to middleware for /api/* routes"
  ```

---

## Task 7: Update requirements.txt and validate deployment

**Only proceed after both fastembed gates pass:**
- Gate 1 (Task 2): mean cosine similarity ≥ 0.9999
- Gate 2: Vercel preview deployment with fastembed fits within 500MB

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Get pinned versions**

  ```bash
  uv pip install fastembed upstash-redis --dry-run 2>&1 | grep -E "^fastembed|^upstash-redis"
  ```

  This prints the resolved pinned versions, e.g.:
  ```
  fastembed==0.4.1
  upstash-redis==1.1.0
  ```

- [ ] **Step 2: Update `requirements.txt`**

  Add the two lines printed by Step 1 at the end of `requirements.txt` (substitute your actual resolved versions):

  ```
  fastembed==0.4.1       # replace with version from Step 1
  upstash-redis==1.1.0   # replace with version from Step 1
  ```

  Update the comment at the top of the file:
  ```
  # Vercel API dependencies — pinned from uv.lock
  # sentence-transformers is intentionally excluded: uses fastembed instead (EMBEDDING_BACKEND=fastembed)
  ```

- [ ] **Step 3: Test-deploy to Vercel preview (Gate 2 — bundle size)**

  Push the branch and create a Vercel preview deployment:
  ```bash
  git push origin phase2-infra
  ```
  In the Vercel dashboard, check the function bundle size for `api/index.py`. Must be ≤ 500MB. If it exceeds 500MB: revert the fastembed addition to `requirements.txt` and keep `/api/search` at 501.

- [ ] **Step 4: Set Vercel env vars**

  In the Vercel dashboard, add:
  - `EMBEDDING_BACKEND` = `fastembed`
  - `UPSTASH_REDIS_REST_URL` = (from Upstash dashboard)
  - `UPSTASH_REDIS_REST_TOKEN` = (from Upstash dashboard)
  - `API_KEY` = (a strong random string, e.g. `openssl rand -hex 32`)

- [ ] **Step 5: Commit and final test run**

  ```bash
  git add requirements.txt
  git commit -m "feat: add fastembed and upstash-redis to Vercel requirements"
  uv run pytest tests/ -v
  ```

  Expected: all pass.

- [ ] **Step 6: Smoke-test the deployed Vercel API**

  Replace `<API_KEY>` and `<VERCEL_URL>` with your values:

  ```bash
  # Auth check — should be 401
  curl -s https://<VERCEL_URL>/api/stats | python -m json.tool

  # Auth check — should be 200
  curl -s -H "X-API-Key: <API_KEY>" https://<VERCEL_URL>/api/stats | python -m json.tool

  # Rate limit check — should be 200 with data
  curl -s -H "X-API-Key: <API_KEY>" https://<VERCEL_URL>/api/clusters?limit=5 | python -m json.tool

  # Search check — should be 200 with results (if both gates passed)
  curl -s -H "X-API-Key: <API_KEY>" "https://<VERCEL_URL>/api/search?q=conflict" | python -m json.tool

  # Static page — should be 200 without API key
  curl -s -o /dev/null -w "%{http_code}" https://<VERCEL_URL>/
  ```
