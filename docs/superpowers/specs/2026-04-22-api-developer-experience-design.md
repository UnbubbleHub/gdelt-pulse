---
name: API Developer Experience
description: Enable Swagger UI, open CORS to *, and add a developer landing page discoverable from the main dashboard nav.
type: project
date: 2026-04-22
---

# API Developer Experience Design

## Goal

Make the GDELT Pulse API immediately usable by external developers without any setup friction: interactive docs, open CORS, and a single page that explains everything they need to know.

## Scope

Three changes to `app.py`, one new static HTML file, and one nav addition to `index.html`.

---

## 1. Swagger UI (`app.py`)

Re-enable FastAPI's built-in interactive docs:

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

- Swagger UI at `/api/docs` — interactive, lets developers try requests in-browser
- OpenAPI JSON at `/api/openapi.json` — importable into Postman, Insomnia, etc.
- ReDoc stays disabled — one UI is enough

---

## 2. CORS (`app.py`)

Change `allow_origins` from the env-var-driven list to unconditional wildcard:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

Remove the `_cors_origins` variable and the `settings.api.cors_origins` lookup — they are no longer needed. The `ApiSettings` dataclass and `CORS_ORIGINS` env var can be left in place (no harm) or removed as part of this change.

**Why `*` is safe here:** the API uses `X-API-Key` header auth (not cookies). Browsers do not block credentialed requests to wildcard origins for header-based auth.

---

## 3. Developer page route (`app.py`)

Add one route following the existing pattern:

```python
@app.get("/developers", include_in_schema=False)
def developers_page():
    return FileResponse(STATIC_DIR / "developers.html")
```

---

## 4. `static/developers.html`

Plain HTML file with inline styles only (no external CSS or JS dependencies). Content sections:

1. **Header** — "GDELT Pulse API" title + one-line description
2. **Quick start** — base URL (production), local dev URL, `X-API-Key` header with a note that the key is optional when `API_KEY` env var is not set
3. **Rate limits** — 30 requests / 60 seconds, per-IP
4. **Endpoints reference** — table with Method, Path, Key params, Description for all `/api/*` routes:
   - `/api/stats`
   - `/api/articles`
   - `/api/clusters`
   - `/api/clusters/{id}`
   - `/api/search`
   - `/api/globe/clusters`
   - `/api/polarization`
   - `/api/polarization/{id}`
   - `/api/asymmetry`
   - `/api/gravity/graph`
   - `/api/gravity/country/{code}`
   - `/api/sources/fingerprints`
   - `/api/sources/{domain}/detail`
   - `/api/propagation/stories`
   - `/api/propagation/{id}`
   - `/api/velocity/topics`
   - `/api/velocity/timeline`
5. **Example curl requests** — one per major endpoint group
6. **Interactive docs link** — prominent link to `/api/docs`
7. **Footer** — link back to `/` (dashboard)

Style: dark background (`#0a0a0a`), monospace code blocks, minimal — no framework, no pulse.css dependency.

---

## 5. `static/index.html` nav addition

Add a "Developers" entry to the `features-grid` in the landing page alongside the existing 8 feature links (Globe, Polarization, Gravity, Asymmetry, Sources, Propagation, Velocity, Search):

```html
<a class="feature-link" href="/developers">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
    <polyline points="16 18 22 12 16 6"/>
    <polyline points="8 6 2 12 8 18"/>
  </svg>
  Developers
</a>
```

The grid currently has 8 items in a 2-column layout — adding a 9th creates an odd row. This is acceptable; the grid is responsive and the item will sit naturally at the bottom left.

---

## Files changed

| File | Change |
|------|--------|
| `src/gdelt_event_pipeline/api/app.py` | Enable Swagger UI, change CORS to `*`, add `/developers` route |
| `src/gdelt_event_pipeline/api/static/developers.html` | New file — developer reference page |
| `src/gdelt_event_pipeline/api/static/index.html` | Add "Developers" link to features grid |

## Files not changed

- `src/gdelt_event_pipeline/config/settings.py` — `ApiSettings` / `CORS_ORIGINS` left in place
- All other static pages — no changes needed
- Tests — no new API behaviour to test; existing tests unaffected

---

## Out of scope

- API key issuance endpoint or key management table
- Rate limiting per API key (remains per-IP)
- Signup form or key distribution flow
