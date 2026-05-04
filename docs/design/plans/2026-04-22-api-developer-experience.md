# API Developer Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Swagger UI at `/api/docs`, open CORS to `*`, add a developer reference page at `/developers`, and link it from the main dashboard nav.

**Architecture:** Three targeted edits to `app.py` (Swagger, CORS, new route), one new static HTML file, and one line added to `index.html`. No new modules, no schema changes, no JS frameworks.

**Tech Stack:** FastAPI (built-in Swagger/OpenAPI support), vanilla HTML/CSS, pytest + FastAPI TestClient.

---

## File Map

| File | Action |
|------|--------|
| `tests/api/test_app.py` | Add tests for new routes and CORS header |
| `src/gdelt_event_pipeline/api/app.py` | Enable Swagger UI, change CORS to `*`, add `/developers` route, remove `_cors_origins` |
| `src/gdelt_event_pipeline/api/static/developers.html` | New — developer reference page |
| `src/gdelt_event_pipeline/api/static/index.html` | Add "Developers" link to features grid |

---

## Task 1: Tests for new routes and CORS

**Files:**
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Add tests to `tests/api/test_app.py`**

Append this new class after the existing `TestApiKeyAuth` class:

```python
class TestDeveloperExperience:
    def test_swagger_ui_route_returns_200(self, client_no_db):
        """GET /api/docs must return 200 (Swagger UI enabled)."""
        response = client_no_db.get("/api/docs")
        assert response.status_code == 200

    def test_openapi_json_route_returns_200(self, client_no_db):
        """GET /api/openapi.json must return valid OpenAPI JSON."""
        response = client_no_db.get("/api/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_developers_page_returns_200(self, client_no_db):
        """GET /developers must return 200 (static HTML served)."""
        response = client_no_db.get("/developers")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_cors_allows_any_origin(self, client_no_db):
        """CORS preflight must respond with Access-Control-Allow-Origin: * for /api/ paths."""
        response = client_no_db.options(
            "/api/stats",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "*"
```

- [ ] **Step 2: Run the new tests to confirm they all fail**

```bash
uv run pytest tests/api/test_app.py::TestDeveloperExperience -v
```

Expected: 4 FAILs — `404` for `/api/docs`, `404` for `/api/openapi.json`, `404` for `/developers`, and CORS test fails because origins are locked.

---

## Task 2: Update `app.py`

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`

- [ ] **Step 1: Enable Swagger UI and OpenAPI JSON**

In `app.py`, find the `FastAPI(...)` constructor (around line 144) and replace:

```python
app = FastAPI(
    title="GDELT Pulse API",
    description="Hybrid semantic + keyword search over GDELT news events.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
```

with:

```python
app = FastAPI(
    title="GDELT Pulse API",
    description="Hybrid semantic + keyword search over GDELT news events.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
```

- [ ] **Step 2: Open CORS to `*`**

Find these lines (around line 154-162):

```python
_settings = get_settings()
_cors_origins = _settings.api.cors_origins or ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

Replace with:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

The `_settings` and `_cors_origins` lines are now unused — delete them both.

- [ ] **Step 3: Add the `/developers` route**

Add this route after the existing `/velocity` route (around line 289) and before the `# ── Endpoints` comment:

```python
@app.get("/developers", include_in_schema=False)
def developers_page():
    """Serve the developer reference page."""
    return FileResponse(STATIC_DIR / "developers.html")
```

- [ ] **Step 4: Run the four new tests — expect 3 to pass, 1 still failing**

```bash
uv run pytest tests/api/test_app.py::TestDeveloperExperience -v
```

Expected:
- `test_swagger_ui_route_returns_200` — PASS
- `test_openapi_json_route_returns_200` — PASS
- `test_developers_page_returns_200` — FAIL (developers.html doesn't exist yet)
- `test_cors_allows_any_origin` — PASS

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
uv run pytest tests/api/ -v
```

Expected: all pre-existing tests pass; `test_developers_page_returns_200` is the only FAIL.

---

## Task 3: Create `static/developers.html`

**Files:**
- Create: `src/gdelt_event_pipeline/api/static/developers.html`

- [ ] **Step 1: Create the file**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GDELT Pulse — Developer Docs</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
    padding: 48px 24px 80px;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  a { color: #7eb8f7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  h1 { font-size: 2rem; font-weight: 600; letter-spacing: -0.03em; margin-bottom: 8px; }
  h2 { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
       color: #888; margin: 48px 0 16px; border-bottom: 1px solid #222; padding-bottom: 8px; }
  h3 { font-size: 0.95rem; font-weight: 600; color: #ccc; margin: 28px 0 8px; }
  p { color: #aaa; margin-bottom: 12px; }
  code {
    font-family: "SF Mono", "Fira Code", Menlo, monospace;
    font-size: 0.88em;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 2px 6px;
    color: #e0e0e0;
  }
  pre {
    background: #111;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 16px 20px;
    overflow-x: auto;
    margin: 10px 0 18px;
  }
  pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 0.875rem;
    color: #c8d9f0;
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-bottom: 16px; }
  th {
    text-align: left;
    padding: 8px 12px;
    background: #111;
    border-bottom: 1px solid #2a2a2a;
    color: #888;
    font-weight: 500;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  td { padding: 9px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
  td:first-child { white-space: nowrap; }
  tr:last-child td { border-bottom: none; }
  .method { color: #7eb8f7; font-family: monospace; font-size: 0.8rem; font-weight: 600; }
  .path { font-family: monospace; color: #c8d9f0; }
  .param { font-family: monospace; font-size: 0.82rem; color: #aaa; }
  .badge {
    display: inline-block;
    background: #1a2a1a;
    border: 1px solid #2a3a2a;
    color: #6abf69;
    font-size: 0.75rem;
    font-family: monospace;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 4px;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 40px;
    padding-bottom: 20px;
    border-bottom: 1px solid #1a1a1a;
  }
  .topbar-back { color: #888; font-size: 0.875rem; }
  .swagger-link {
    display: inline-block;
    background: #1a2a3a;
    border: 1px solid #2a3a4a;
    color: #7eb8f7;
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 0.875rem;
    margin-top: 4px;
  }
  .swagger-link:hover { background: #1e3248; text-decoration: none; }
  .info-row { display: flex; gap: 32px; margin-bottom: 8px; }
  .info-item label { display: block; font-size: 0.75rem; text-transform: uppercase;
                     letter-spacing: 0.08em; color: #555; margin-bottom: 4px; }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <a class="topbar-back" href="/">&larr; GDELT Pulse</a>
    <a class="swagger-link" href="/api/docs">Interactive Docs (Swagger UI) &rarr;</a>
  </div>

  <h1>GDELT Pulse API</h1>
  <p>Hybrid semantic + keyword search over global news events. All endpoints are read-only (<code>GET</code>).</p>

  <h2>Quick Start</h2>

  <div class="info-row">
    <div class="info-item">
      <label>Base URL</label>
      <code id="baseUrl"></code>
    </div>
    <div class="info-item">
      <label>Auth Header</label>
      <code>X-API-Key: &lt;your-key&gt;</code>
    </div>
    <div class="info-item">
      <label>Rate Limit</label>
      <code>30 req / 60 s per IP</code>
    </div>
  </div>

  <p style="margin-top:12px">The <code>X-API-Key</code> header is only required when the server has an <code>API_KEY</code>
  environment variable set. Local dev instances and public deployments without a key set allow unauthenticated requests.</p>

  <pre><code>curl "<span class="base-url-placeholder"></span>/api/stats" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h2>Endpoints</h2>

  <h3>Core</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/stats</span></td>
        <td>—</td>
        <td>Total articles, clusters, embeddings, largest cluster</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/articles</span></td>
        <td><span class="param">limit</span> (1–200, default 50)</td>
        <td>Recent articles, newest first</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/clusters</span></td>
        <td><span class="param">limit, sort, location, person, org, theme, domain, source, date_from, date_to</span></td>
        <td>Active event clusters with optional filtering</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/clusters/{id}</span></td>
        <td>—</td>
        <td>Single cluster with all member articles</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/search</span></td>
        <td><span class="param">q (required), limit, semantic_weight, clusters, location, person, org, theme, domain, source, date_from, date_to</span></td>
        <td>Hybrid semantic + keyword search via RRF</td>
      </tr>
    </tbody>
  </table>

  <h3>Globe</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/globe/clusters</span></td>
        <td><span class="param">mode (live|rising|silent), limit</span></td>
        <td>Top clusters with geographic coordinates for 3D globe rendering</td>
      </tr>
    </tbody>
  </table>

  <h3>Polarization</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/polarization</span></td>
        <td><span class="param">limit, min_articles</span></td>
        <td>Story clusters ranked by narrative tone divergence across sources</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/polarization/{id}</span></td>
        <td>—</td>
        <td>Per-article tone breakdown for a single cluster, grouped by source</td>
      </tr>
    </tbody>
  </table>

  <h3>Attention Asymmetry</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/asymmetry</span></td>
        <td>—</td>
        <td>Coverage volume vs crisis intensity by country; reveals over- and underreported regions</td>
      </tr>
    </tbody>
  </table>

  <h3>Geopolitical Gravity</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/gravity/graph</span></td>
        <td><span class="param">min_weight, limit_edges</span></td>
        <td>Country co-mention graph (nodes + edges) for gravity map</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/gravity/country/{code}</span></td>
        <td>—</td>
        <td>Single country: top connections, top clusters, avg tone (ISO 3166-1 alpha-2 code)</td>
      </tr>
    </tbody>
  </table>

  <h3>Source DNA</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/sources/fingerprints</span></td>
        <td><span class="param">limit, sort, min_articles</span></td>
        <td>Per-source tone fingerprints, top themes, and top countries</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/sources/{domain}/detail</span></td>
        <td>—</td>
        <td>Full fingerprint for one source domain: tone timeline, all themes, category breakdown</td>
      </tr>
    </tbody>
  </table>

  <h3>Story Propagation</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/propagation/stories</span></td>
        <td><span class="param">limit, min_sources</span></td>
        <td>Multi-source clusters suitable for propagation analysis</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/propagation/{id}</span></td>
        <td>—</td>
        <td>Chronological timeline of how a story spread across sources</td>
      </tr>
    </tbody>
  </table>

  <h3>Topic Velocity</h3>
  <table>
    <thead><tr><th>Method</th><th>Path</th><th>Key params</th><th>Description</th></tr></thead>
    <tbody>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/velocity/topics</span></td>
        <td><span class="param">hours (6–168, default 48), limit</span></td>
        <td>Themes with rising or declining velocity over a configurable time window</td>
      </tr>
      <tr>
        <td><span class="method">GET</span></td>
        <td><span class="path">/api/velocity/timeline</span></td>
        <td><span class="param">theme (required), hours</span></td>
        <td>Hourly article count for a specific GDELT theme code</td>
      </tr>
    </tbody>
  </table>

  <h2>Examples</h2>

  <h3>Search</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/search?q=earthquake+Turkey&semantic_weight=0.7&location=Turkey&limit=10" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>List clusters with filters</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/clusters?sort=articles&theme=MILITARY_CONFLICT&limit=20" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Globe data (rising stories)</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/globe/clusters?mode=rising&limit=12" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Polarization ranking</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/polarization?limit=20&min_articles=30" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Attention asymmetry</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/asymmetry" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Country gravity detail</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/gravity/country/US" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Source fingerprint</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/sources/bbc.co.uk/detail" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Story propagation timeline</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/propagation/stories?min_sources=5&limit=10" \
  -H "X-API-Key: your-key-here"</code></pre>

  <h3>Topic velocity (rising themes)</h3>
  <pre><code>curl "<span class="base-url-placeholder"></span>/api/velocity/topics?hours=24&limit=20" \
  -H "X-API-Key: your-key-here"</code></pre>

  <p style="margin-top:24px">For full parameter details and to try requests in the browser, use the
  <a href="/api/docs">interactive Swagger UI</a>.</p>

  <h2>OpenAPI Spec</h2>
  <p>The machine-readable spec is available at <a href="/api/openapi.json"><code>/api/openapi.json</code></a>
  — import it into Postman, Insomnia, or any OpenAPI-compatible tooling.</p>

</div>

<script>
  const base = window.location.origin;
  document.getElementById('baseUrl').textContent = base;
  document.querySelectorAll('.base-url-placeholder').forEach(el => {
    el.textContent = base;
  });
</script>
</body>
</html>
```

- [ ] **Step 2: Run the full test suite — all 4 new tests should now pass**

```bash
uv run pytest tests/api/test_app.py::TestDeveloperExperience -v
```

Expected: all 4 PASS.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
uv run pytest tests/api/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/api/test_app.py \
        src/gdelt_event_pipeline/api/app.py \
        src/gdelt_event_pipeline/api/static/developers.html
git commit -m "feat: enable Swagger UI, open CORS to *, add /developers page"
```

---

## Task 4: Add Developers link to dashboard nav

**Files:**
- Modify: `src/gdelt_event_pipeline/api/static/index.html`

- [ ] **Step 1: Add the Developers feature link**

In `index.html`, find the last `<a class="feature-link"...>` entry in the features grid. It currently ends with the Search link:

```html
        <a class="feature-link" onclick="navTo('clusters')" style="cursor:pointer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Search
        </a>
      </div>
```

Add the Developers link immediately before the closing `</div>` of the grid:

```html
        <a class="feature-link" onclick="navTo('clusters')" style="cursor:pointer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Search
        </a>
        <a class="feature-link" href="/developers">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          Developers
        </a>
      </div>
```

- [ ] **Step 2: Verify the change visually**

Start the API server (with DB mocked — just check the HTML is served):

```bash
uv run python -m gdelt_event_pipeline.api.app
```

Open `http://localhost:8000` in a browser. Confirm the "Developers" tile appears in the features grid and clicking it opens `/developers`.

Open `http://localhost:8000/developers`. Confirm the page loads, the base URL is correctly auto-filled (showing `http://localhost:8000`), curl examples show the right base URL, and the Swagger UI link at the top right navigates to `/api/docs`.

Open `http://localhost:8000/api/docs`. Confirm Swagger UI loads and all endpoints are listed.

- [ ] **Step 3: Commit**

```bash
git add src/gdelt_event_pipeline/api/static/index.html
git commit -m "feat: add Developers link to dashboard landing page"
```
