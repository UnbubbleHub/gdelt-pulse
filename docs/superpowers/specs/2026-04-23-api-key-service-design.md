# API Key Service Design

**Date:** 2026-04-23  
**Status:** Approved  

---

## Overview

Replace the single static `API_KEY` environment variable with a full per-user API key management service. Users create an account (email+password or OAuth), generate one API key from a dashboard, and use it to authenticate requests against the GDELT Pulse API with a higher rate limit than anonymous callers.

---

## Architecture

```
User browser
  │
  ├─ /developers  ──────→  Public API docs page (no auth required)
  │                         Links to /dashboard to get a key
  │
  ├─ /login, /signup  ──→  Clerk hosted pages (email+password / Google / GitHub)
  │                         Clerk issues a session JWT stored in cookie
  │
  ├─ /dashboard  ──────→  Clerk-gated vanilla JS page
  │                         Redirect to /login if not authenticated
  │                         Authenticated users create and revoke their API key here
  │                         calls GET/POST/DELETE /api/auth/keys
  │
  └─ /api/*  ──────────→  FastAPI middleware
                             if X-API-Key present → SHA-256 hash → lookup in Neon
                               ✓ found + active → 200 req/60s limit (per-key)
                               ✗ invalid → 401
                             if no key → 30 req/60s limit (per-IP, existing behavior)
```

**Access rules:**
- `/developers` — fully public, no auth required
- `/dashboard` — requires Clerk session; redirects to sign-in if unauthenticated
- `/api/auth/keys` — requires valid Clerk JWT in `Authorization: Bearer` header
- `/api/*` (data endpoints) — public, optional `X-API-Key` for higher rate limit

**New pieces:**
- **Clerk** — managed auth (email+password, Google OAuth, GitHub OAuth), issues JWTs
- **`api_keys` table** — added to existing Neon DB, stores hashed keys
- **`/api/auth/keys` endpoints** — Clerk-JWT-protected, generate/revoke keys
- **`/dashboard` page** — Clerk-gated vanilla JS key management UI
- **Updated middleware** — hash-based DB lookup replaces single env-var compare

**Unchanged:** `/developers` page, all existing `/api/*` endpoints, Upstash Redis rate limiter, pipeline logic.

---

## Database Schema

```sql
CREATE TABLE api_keys (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT        NOT NULL,          -- Clerk user ID (e.g. "user_2abc...")
    key_prefix   TEXT        NOT NULL,          -- gdp_ + first 8 random chars, shown in dashboard
    key_hash     TEXT        NOT NULL UNIQUE,   -- SHA-256 of the full key
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

-- enforces one active key per user
CREATE UNIQUE INDEX api_keys_one_active_per_user
    ON api_keys (user_id)
    WHERE revoked_at IS NULL;
```

**Key format:** `gdp_` prefix + 32 random hex chars (e.g. `gdp_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`)

- Full key shown to user **once** at creation, never stored plaintext
- `key_prefix` (`gdp_` + first 8 random chars = 12 chars total) stored for dashboard display
- `last_used_at` updated async on every authenticated request (fire-and-forget, no latency impact)
- Revocation sets `revoked_at` — rows are retained for audit, not deleted

---

## API Endpoints

All endpoints under `/api/auth/keys`, protected by Clerk JWT (`Authorization: Bearer <session_token>`). The JWT is verified against Clerk's JWKS endpoint; `user_id` is extracted from the `sub` claim.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/auth/keys` | Get current key metadata |
| `POST` | `/api/auth/keys` | Generate new key (revokes existing active key) |
| `DELETE` | `/api/auth/keys` | Revoke current active key |

**GET response (key exists):**
```json
{
  "active": true,
  "prefix": "gdp_a1b2c3",
  "created_at": "2026-04-23T10:00:00Z",
  "last_used_at": "2026-04-23T11:30:00Z"
}
```

**GET response (no key):**
```json
{ "active": false }
```

**POST response (full key returned once only):**
```json
{
  "key": "gdp_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "prefix": "gdp_a1b2c3",
  "created_at": "2026-04-23T10:00:00Z"
}
```

**DELETE response:**
```json
{ "revoked": true }
```

All operations are scoped to the authenticated user — no cross-user access is possible.

---

## Middleware Changes

The existing `rate_limit_middleware` in `app.py` is updated:

```
Request hits /api/*
    │
    ├─ Skip /api/auth/* paths (no key check on auth endpoints themselves)
    │
    ├─ X-API-Key header present?
    │   ├─ YES → SHA-256 hash the key
    │   │         → SELECT from api_keys WHERE key_hash=? AND revoked_at IS NULL
    │   │         ├─ Found  → attach user_id to request state
    │   │         │           apply 200 req/60s rate limit (keyed on user_id via Redis)
    │   │         │           async UPDATE last_used_at (fire-and-forget)
    │   │         └─ Not found → 401 {"detail": "Invalid or missing API key."}
    │   │
    └─ NO key → apply 30 req/60s rate limit (keyed on IP, existing behavior)
               → continue to endpoint
```

The old `API_KEY` env var check is **removed**. Existing single-key deployments must migrate users to the new system.

The `key_hash` column has a UNIQUE index so the lookup is an index scan — latency impact is negligible against an already-pooled Neon connection.

---

## Dashboard UI

**Route:** `/dashboard`  
**Implementation:** Vanilla JS, dark theme matching the existing site. Clerk JS SDK loaded from CDN — no build step required.

### States

**Unauthenticated:** Redirect immediately to Clerk sign-in. No content rendered.

**Authenticated, no active key:**
```
Your API Key
─────────────────────────────
You don't have an active key yet.

[ Generate API Key ]
```

**Authenticated, key just generated (one-time reveal):**
```
Your API Key          ⚠ Save this now — it won't be shown again
─────────────────────────────
gdp_a1b2c3d4e5f6g7h8...    [ Copy ]

[ Revoke Key ]
```

**Authenticated, existing key:**
```
Your API Key
─────────────────────────────
gdp_a1b2c3••••••••   Created Apr 23
Last used: 2 hours ago

[ Regenerate Key ]   [ Revoke Key ]
```

"Regenerate Key" revokes the current key and immediately generates a new one, showing the one-time reveal state. A confirmation prompt appears before regenerating.

---

## Auth Integration (Clerk)

- **Provider:** Clerk (free tier: 10k MAU)
- **Methods:** Email+password, Google OAuth, GitHub OAuth
- **SDK:** `@clerk/clerk-js` loaded from CDN in dashboard and any auth-gated pages
- **JWT verification:** FastAPI dependency that fetches Clerk's JWKS on startup and caches it, verifying the `Authorization: Bearer` token on each `/api/auth/*` request
- **Environment variables added:**
  - `CLERK_PUBLISHABLE_KEY` — frontend SDK init
  - `CLERK_SECRET_KEY` — backend JWT verification (SDK derives JWKS URL automatically)

---

## Error Handling

| Scenario | Response |
|----------|----------|
| Invalid/missing API key on `/api/*` | `401 {"detail": "Invalid or missing API key."}` |
| Rate limit exceeded | `429 {"detail": "Rate limit exceeded."}` |
| Invalid/expired Clerk JWT on `/api/auth/*` | `401 {"detail": "Unauthorized."}` |
| Generate key when one already active | Silently revokes old key, returns new one |
| Revoke when no active key | `404 {"detail": "No active key found."}` |

---

## What Is Not In Scope

- Multiple keys per user
- Per-key rate limit customization
- Usage metrics / analytics dashboard
- Admin panel for managing all users' keys
- Key expiry dates
