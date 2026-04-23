# API Key Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single static `API_KEY` env-var with a full per-user API key service backed by Clerk auth, Neon DB, and a dashboard UI.

**Architecture:** Clerk handles email+password and OAuth; the backend verifies Clerk JWTs on `/api/auth/keys` endpoints. API keys are stored hashed in a new `api_keys` Neon table. The existing middleware is updated to do a hash-based DB lookup on `X-API-Key` — anonymous callers get 30 req/60s, key-holders get 200 req/60s. A new `/dashboard` page (vanilla JS + Clerk JS SDK) is where logged-in users create and revoke their key; `/developers` stays fully public.

**Tech Stack:** FastAPI, psycopg (sync), PyJWT[cryptography] for RS256 JWT verification, Clerk JS SDK (CDN), PostgreSQL/Neon, Upstash Redis (existing), pytest + unittest.mock

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `sql/002_api_keys.sql` | `api_keys` table + unique partial index |
| Create | `src/gdelt_event_pipeline/api/auth.py` | `require_clerk_user` FastAPI dependency |
| Create | `src/gdelt_event_pipeline/api/keys.py` | `/api/auth/keys` GET/POST/DELETE router |
| Create | `src/gdelt_event_pipeline/api/static/dashboard.html` | Clerk-gated key management UI |
| Create | `tests/api/test_auth.py` | Unit tests for JWT verification |
| Create | `tests/api/test_keys.py` | Tests for key CRUD endpoints |
| Modify | `pyproject.toml` | Add `PyJWT[cryptography]` dependency |
| Modify | `requirements.txt` | Pin `PyJWT[cryptography]` and `cryptography` |
| Modify | `src/gdelt_event_pipeline/api/app.py` | Update `_ensure_schema`, CORS methods, middleware, register router, add `/dashboard` + `/api/auth/config` routes |
| Modify | `tests/api/test_app.py` | Update `TestApiKeyAuth` for new DB-backed behavior |
| Modify | `src/gdelt_event_pipeline/api/static/developers.html` | Add "Get API Key" CTA |
| Modify | `.env.example` | Add `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `CLERK_JWKS_URL` |

---

## Task 1: Add PyJWT dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add to pyproject.toml**

In `pyproject.toml`, add `"PyJWT[cryptography]>=2.8.0"` to the `dependencies` list:

```toml
dependencies = [
    "psycopg[binary]>=3.2.0",
    "psycopg-pool>=3.2.0",
    "pgvector>=0.3.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-dotenv>=1.0.0",
    "fastembed>=0.8.0",
    "upstash-redis>=1.7.0",
    "PyJWT[cryptography]>=2.8.0",
]
```

- [ ] **Step 2: Add to requirements.txt**

Append to `requirements.txt` (after the `upstash-redis` block):

```
# PyJWT — RS256 JWT verification for Clerk tokens
PyJWT==2.10.1
cryptography==44.0.3
cffi==1.17.1
pycparser==2.22
```

- [ ] **Step 3: Install and verify**

```bash
pip install "PyJWT[cryptography]>=2.8.0"
python -c "import jwt; print(jwt.__version__)"
```

Expected output: `2.10.1` (or similar 2.x)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add PyJWT[cryptography] for Clerk JWT verification"
```

---

## Task 2: Create api_keys migration

**Files:**
- Create: `sql/002_api_keys.sql`

- [ ] **Step 1: Write the migration**

Create `sql/002_api_keys.sql`:

```sql
-- Migration 002: per-user API key management
-- Keys are stored hashed (SHA-256). The full key is never persisted.
-- Revoked keys are kept for audit; revoked_at IS NULL means active.

CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT        NOT NULL,
    key_prefix   TEXT        NOT NULL,
    key_hash     TEXT        NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

-- One active key per user (partial index: only rows where revoked_at IS NULL)
CREATE UNIQUE INDEX IF NOT EXISTS api_keys_one_active_per_user
    ON api_keys (user_id)
    WHERE revoked_at IS NULL;
```

- [ ] **Step 2: Commit**

```bash
git add sql/002_api_keys.sql
git commit -m "feat: add api_keys migration (002)"
```

---

## Task 3: Update _ensure_schema to run migration 002

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py` (lines 100–128)

- [ ] **Step 1: Replace `_ensure_schema` in app.py**

Replace the entire `_ensure_schema` function (lines 100–128) with this version that runs both migrations independently:

```python
def _ensure_schema() -> None:
    """Run any missing schema migrations on first deploy."""
    import logging

    from gdelt_event_pipeline.storage.database import get_pool

    logger = logging.getLogger(__name__)
    pool = get_pool()

    migrations = [
        ("articles", Path(__file__).resolve().parents[3] / "sql" / "001_schema.sql"),
        ("api_keys", Path(__file__).resolve().parents[3] / "sql" / "002_api_keys.sql"),
    ]

    with pool.connection() as conn:
        for table_name, schema_path in migrations:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_name = %s"
                    ")",
                    (table_name,),
                )
                row = cur.fetchone()
                exists = row[0] if isinstance(row, (tuple, list)) else row.get("exists", False)
            if not exists:
                logger.info("Table '%s' missing — running migration %s", table_name, schema_path.name)
                # Try the repo path first, then the Docker path
                if not schema_path.exists():
                    schema_path = Path(f"/app/sql/{schema_path.name}")
                sql = schema_path.read_text()
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
                logger.info("Migration %s complete.", schema_path.name)
```

- [ ] **Step 2: Verify the import `Path` is still imported at top of file**

`from pathlib import Path` is already on line 13 of `app.py` — no change needed.

- [ ] **Step 3: Commit**

```bash
git add src/gdelt_event_pipeline/api/app.py
git commit -m "feat: run 002_api_keys migration in _ensure_schema"
```

---

## Task 4: Create the Clerk JWT auth module

**Files:**
- Create: `src/gdelt_event_pipeline/api/auth.py`
- Create: `tests/api/test_auth.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_auth.py`:

```python
"""Tests for Clerk JWT verification dependency."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def clear_jwks_cache():
    """Clear the lru_cache between tests so env changes take effect."""
    import gdelt_event_pipeline.api.auth as auth_module
    auth_module._jwks_client.cache_clear()
    yield
    auth_module._jwks_client.cache_clear()


def _call(authorization: str) -> str:
    """Call require_clerk_user directly (FastAPI Header default is just a default value)."""
    from gdelt_event_pipeline.api.auth import require_clerk_user
    return require_clerk_user(authorization=authorization)


class TestRequireClerkUser:
    def test_missing_bearer_prefix_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            _call("Basic abc123")
        assert exc_info.value.status_code == 401

    def test_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            _call("")
        assert exc_info.value.status_code == 401

    def test_invalid_jwt_raises_401(self, monkeypatch):
        monkeypatch.setenv("CLERK_JWKS_URL", "https://example.clerk.accounts.dev/.well-known/jwks.json")

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.side_effect = Exception("invalid token")

        with patch("gdelt_event_pipeline.api.auth._jwks_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                _call("Bearer invalid.token.here")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized."

    def test_valid_jwt_returns_user_id(self, monkeypatch):
        monkeypatch.setenv("CLERK_JWKS_URL", "https://example.clerk.accounts.dev/.well-known/jwks.json")

        mock_signing_key = MagicMock()
        mock_signing_key.key = "fake-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key

        with (
            patch("gdelt_event_pipeline.api.auth._jwks_client", return_value=mock_client),
            patch("jwt.decode", return_value={"sub": "user_abc123"}),
        ):
            user_id = _call("Bearer valid.token.here")

        assert user_id == "user_abc123"

    def test_missing_jwks_url_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("CLERK_JWKS_URL", raising=False)
        import gdelt_event_pipeline.api.auth as auth_module
        auth_module._jwks_client.cache_clear()

        with pytest.raises(RuntimeError, match="CLERK_JWKS_URL"):
            auth_module._jwks_client()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/api/test_auth.py -v
```

Expected: `ModuleNotFoundError: No module named 'gdelt_event_pipeline.api.auth'`

- [ ] **Step 3: Write the auth module**

Create `src/gdelt_event_pipeline/api/auth.py`:

```python
"""Clerk JWT verification for FastAPI endpoints."""

from __future__ import annotations

import os
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    url = os.environ.get("CLERK_JWKS_URL", "")
    if not url:
        raise RuntimeError("CLERK_JWKS_URL environment variable is not set")
    return jwt.PyJWKClient(url)


def require_clerk_user(authorization: str = Header(...)) -> str:
    """FastAPI dependency: verifies Clerk JWT, returns user_id (sub claim)."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    token = authorization[7:]
    try:
        client = _jwks_client()
        signing_key = client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return payload["sub"]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_auth.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gdelt_event_pipeline/api/auth.py tests/api/test_auth.py
git commit -m "feat: add Clerk JWT auth dependency with tests"
```

> **Note on direct calls:** `require_clerk_user(authorization="Bearer token")` works as a plain Python call because `Header(...)` is just the *default value* of the `authorization` parameter — passing the argument explicitly overrides it.

---

## Task 5: Create the API key endpoints router

**Files:**
- Create: `src/gdelt_event_pipeline/api/keys.py`
- Create: `tests/api/test_keys.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_keys.py`:

```python
"""Tests for /api/auth/keys endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import gdelt_event_pipeline.api.app as app_module
from gdelt_event_pipeline.api.auth import require_clerk_user


TEST_USER_ID = "user_test123"
TEST_NOW = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def client():
    """TestClient with DB lifecycle and require_clerk_user mocked out."""
    app_module.app.dependency_overrides[require_clerk_user] = lambda: TEST_USER_ID
    with (
        patch("gdelt_event_pipeline.api.app.init_pool"),
        patch("gdelt_event_pipeline.api.app.close_pool"),
        patch("gdelt_event_pipeline.api.app._ensure_schema"),
    ):
        with TestClient(app_module.app) as c:
            yield c
    app_module.app.dependency_overrides.clear()


def _mock_pool_no_key():
    """Return a mock pool where the user has no active key."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = None

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn
    return mock_pool


def _mock_pool_with_key(prefix="gdp_a1b2c3d4", created_at=TEST_NOW, last_used_at=None):
    """Return a mock pool where the user has an active key."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = {
        "key_prefix": prefix,
        "created_at": created_at,
        "last_used_at": last_used_at,
    }

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    mock_pool = MagicMock()
    mock_pool.connection.return_value = mock_conn
    return mock_pool


class TestGetKey:
    def test_returns_inactive_when_no_key(self, client):
        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=_mock_pool_no_key()):
            resp = client.get("/api/auth/keys")
        assert resp.status_code == 200
        assert resp.json() == {"active": False, "prefix": None, "created_at": None, "last_used_at": None}

    def test_returns_active_key_metadata(self, client):
        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=_mock_pool_with_key()):
            resp = client.get("/api/auth/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["prefix"] == "gdp_a1b2c3d4"
        assert data["last_used_at"] is None

    def test_requires_auth(self):
        """Without dependency override, missing Authorization header returns 422."""
        with (
            patch("gdelt_event_pipeline.api.app.init_pool"),
            patch("gdelt_event_pipeline.api.app.close_pool"),
            patch("gdelt_event_pipeline.api.app._ensure_schema"),
        ):
            with TestClient(app_module.app) as c:
                resp = c.get("/api/auth/keys")
        assert resp.status_code == 422  # missing required Header


class TestCreateKey:
    def test_creates_key_and_returns_full_key_once(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.post("/api/auth/keys")

        assert resp.status_code == 201
        data = resp.json()
        assert data["key"].startswith("gdp_")
        assert len(data["key"]) == 36  # "gdp_" + 32 hex chars
        assert data["prefix"] == data["key"][:12]
        assert "created_at" in data

    def test_key_format_is_correct(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.post("/api/auth/keys")

        key = resp.json()["key"]
        assert key.startswith("gdp_")
        # The random part should be 32 lowercase hex chars
        random_part = key[4:]
        assert len(random_part) == 32
        assert all(c in "0123456789abcdef" for c in random_part)


class TestRevokeKey:
    def test_revokes_active_key(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = ("some-uuid",)  # RETURNING id

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.delete("/api/auth/keys")

        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}

    def test_returns_404_when_no_active_key(self, client):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = None  # no rows affected

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.keys.get_pool", return_value=mock_pool):
            resp = client.delete("/api/auth/keys")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "No active key found."


class TestGenerateKeyHelper:
    def test_generate_key_returns_correct_tuple(self):
        from gdelt_event_pipeline.api.keys import _generate_key

        full_key, prefix, key_hash = _generate_key()

        assert full_key.startswith("gdp_")
        assert len(full_key) == 36
        assert prefix == full_key[:12]
        assert len(key_hash) == 64  # SHA-256 hex digest

    def test_generate_key_produces_unique_keys(self):
        from gdelt_event_pipeline.api.keys import _generate_key

        keys = {_generate_key()[0] for _ in range(10)}
        assert len(keys) == 10  # all unique
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/api/test_keys.py -v
```

Expected: `ModuleNotFoundError: No module named 'gdelt_event_pipeline.api.keys'`

- [ ] **Step 3: Write the keys router**

Create `src/gdelt_event_pipeline/api/keys.py`:

```python
"""API key management endpoints — requires Clerk JWT auth."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel

from gdelt_event_pipeline.api.auth import require_clerk_user
from gdelt_event_pipeline.storage.database import get_pool

router = APIRouter(prefix="/api/auth", tags=["auth"])

_KEY_PREFIX_LEN = 12  # "gdp_" + 8 hex chars


class KeyMeta(BaseModel):
    active: bool
    prefix: str | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None


class KeyCreated(BaseModel):
    key: str
    prefix: str
    created_at: datetime


def _generate_key() -> tuple[str, str, str]:
    """Return (full_key, prefix, sha256_hash). Full key is never stored."""
    random_part = secrets.token_hex(16)  # 32 lowercase hex chars
    full_key = f"gdp_{random_part}"
    prefix = full_key[:_KEY_PREFIX_LEN]
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, prefix, key_hash


@router.get("/keys", response_model=KeyMeta)
def get_key(user_id: str = Depends(require_clerk_user)) -> KeyMeta:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT key_prefix, created_at, last_used_at "
                "FROM api_keys WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )
            row = cur.fetchone()
    if row is None:
        return KeyMeta(active=False)
    return KeyMeta(
        active=True,
        prefix=row["key_prefix"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
    )


@router.post("/keys", response_model=KeyCreated, status_code=201)
def create_key(user_id: str = Depends(require_clerk_user)) -> KeyCreated:
    full_key, prefix, key_hash = _generate_key()
    now = datetime.now(tz=timezone.utc)
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET revoked_at = %s WHERE user_id = %s AND revoked_at IS NULL",
                (now, user_id),
            )
            cur.execute(
                "INSERT INTO api_keys (user_id, key_prefix, key_hash, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, prefix, key_hash, now),
            )
        conn.commit()
    return KeyCreated(key=full_key, prefix=prefix, created_at=now)


@router.delete("/keys")
def revoke_key(user_id: str = Depends(require_clerk_user)) -> dict:
    now = datetime.now(tz=timezone.utc)
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET revoked_at = %s "
                "WHERE user_id = %s AND revoked_at IS NULL RETURNING id",
                (now, user_id),
            )
            deleted = cur.fetchone()
        conn.commit()
    if deleted is None:
        raise HTTPException(status_code=404, detail="No active key found.")
    return {"revoked": True}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_keys.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gdelt_event_pipeline/api/keys.py tests/api/test_keys.py
git commit -m "feat: add /api/auth/keys CRUD endpoints with tests"
```

---

## Task 6: Update CORS config and middleware in app.py

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`
- Modify: `tests/api/test_app.py`

- [ ] **Step 1: Update CORS to allow POST and DELETE**

In `app.py`, locate the `CORSMiddleware` block (lines 154–160) and change `allow_methods`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=False,
)
```

- [ ] **Step 2: Add `hashlib` import and `RATE_LIMIT_MAX_KEY` constant**

At the top of `app.py`, add `hashlib` to the imports:

```python
import hashlib
import hmac
import os
import threading as _threading
import time
```

Remove `import hmac` (it's no longer needed after this task). After the constants block (around line 167), add:

```python
RATE_LIMIT_MAX = 30       # anonymous requests per window
RATE_LIMIT_MAX_KEY = 200  # key-authenticated requests per window
RATE_LIMIT_WINDOW = 60    # seconds
```

Remove `RATE_LIMIT_MAX = 30` and `RATE_LIMIT_WINDOW = 60` (replacing them with the three lines above).

- [ ] **Step 3: Replace `rate_limit_middleware`**

Replace the entire `rate_limit_middleware` function (lines 190–239) with:

```python
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Optional API key auth + per-IP/per-key rate limiting for /api/* paths."""
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # /api/auth/* endpoints handle their own Clerk JWT auth — skip middleware
    if request.url.path.startswith("/api/auth/"):
        return await call_next(request)

    rate_limit_max = RATE_LIMIT_MAX
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_id = client_ip

    # Optional API key auth
    provided_key = request.headers.get("X-API-Key")
    if provided_key:
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        from gdelt_event_pipeline.storage.database import get_pool as _get_pool

        pool = _get_pool()
        with pool.connection() as conn:
            from psycopg.rows import dict_row as _dict_row

            with conn.cursor(row_factory=_dict_row) as cur:
                cur.execute(
                    "SELECT id, user_id FROM api_keys "
                    "WHERE key_hash = %s AND revoked_at IS NULL",
                    (key_hash,),
                )
                row = cur.fetchone()
        if row is None:
            return Response(
                content='{"detail":"Invalid or missing API key."}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": 'ApiKey realm="GDELT Pulse API"'},
            )
        rate_limit_max = RATE_LIMIT_MAX_KEY
        rate_limit_id = f"key:{row['user_id']}"
        # Update last_used_at without blocking the response
        key_id = row["id"]

        def _update_last_used(_pool=pool, _key_id=key_id):
            try:
                with _pool.connection() as _conn:
                    with _conn.cursor() as _cur:
                        _cur.execute(
                            "UPDATE api_keys SET last_used_at = NOW() WHERE id = %s",
                            (_key_id,),
                        )
                    _conn.commit()
            except Exception:
                pass

        import threading as _t
        _t.Thread(target=_update_last_used, daemon=True).start()

    # Rate limiting (Redis or in-memory fallback)
    now = time.time()
    redis = _get_redis()

    if redis is not None:
        key = f"ratelimit:{rate_limit_id}"
        window_start = now - RATE_LIMIT_WINDOW
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}-{os.urandom(4).hex()}": now})
        pipe.expire(key, RATE_LIMIT_WINDOW)
        results = pipe.execute()
        count = results[1]
    else:
        timestamps = _rate_limit_store[rate_limit_id]
        _rate_limit_store[rate_limit_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        count = len(_rate_limit_store[rate_limit_id])
        if count < rate_limit_max:
            _rate_limit_store[rate_limit_id].append(now)

    if count >= rate_limit_max:
        return Response(
            content='{"detail":"Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )

    return await call_next(request)
```

- [ ] **Step 4: Update TestApiKeyAuth in test_app.py**

The old `TestApiKeyAuth` class tested the removed env-var behavior. Replace the entire `TestApiKeyAuth` class in `tests/api/test_app.py` with:

```python
class TestApiKeyAuth:
    def test_no_key_header_passes_with_anonymous_limit(self, client_no_db, monkeypatch):
        """Requests with no X-API-Key go through with the anonymous rate limit."""
        monkeypatch.setattr(app_module, "_redis", None)
        monkeypatch.setattr(app_module, "_SEARCH_AVAILABLE", False)

        response = client_no_db.get("/api/search?q=test")
        assert response.status_code == 501  # middleware passed; endpoint returned 501

    def test_invalid_key_returns_401(self, client_no_db, monkeypatch):
        """When X-API-Key does not match any active key in DB, return 401."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(app_module, "_redis", None)

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.return_value = None  # key not found

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        mock_pool = MagicMock()
        mock_pool.connection.return_value = mock_conn

        with patch("gdelt_event_pipeline.api.app._get_pool", return_value=mock_pool):  # noqa: F821
            # patch the lazy import inside the middleware
            pass

        import gdelt_event_pipeline.storage.database as db_module
        with patch.object(db_module, "get_pool", return_value=mock_pool):
            response = client_no_db.get("/api/search?q=test", headers={"X-API-Key": "gdp_invalidkey"})

        assert response.status_code == 401
        assert "api key" in response.json()["detail"].lower()

    def test_auth_endpoints_skip_middleware(self, client_no_db):
        """/api/auth/* paths bypass the key check and rate limiter."""
        # /api/auth/keys requires a valid Clerk JWT — without one it returns 422 (missing Header)
        # but NOT 401 from the middleware, proving middleware was skipped
        response = client_no_db.get("/api/auth/keys")
        assert response.status_code == 422  # FastAPI schema validation, not 401 from middleware

    def test_static_pages_not_protected(self, client_no_db):
        """Static routes (/, /globe, etc.) must be accessible without any API key."""
        response = client_no_db.get("/")
        assert response.status_code == 200
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass (some `TestApiKeyAuth` tests replaced — that's correct). If `test_invalid_key_returns_401` fails due to import patching, see note below.

> **Note:** The `get_pool` inside the middleware is a lazy import (`from gdelt_event_pipeline.storage.database import get_pool as _get_pool`). To patch it in tests, patch `gdelt_event_pipeline.storage.database.get_pool` as shown in the test.

- [ ] **Step 6: Commit**

```bash
git add src/gdelt_event_pipeline/api/app.py tests/api/test_app.py
git commit -m "feat: replace env-var API key check with DB hash lookup in middleware"
```

---

## Task 7: Wire router + new routes into app.py

**Files:**
- Modify: `src/gdelt_event_pipeline/api/app.py`

- [ ] **Step 1: Import and register the keys router**

Near the top of `app.py`, after the existing imports, add:

```python
from gdelt_event_pipeline.api.keys import router as keys_router
```

After `app.mount("/static", ...)` (line 162), add:

```python
app.include_router(keys_router)
```

- [ ] **Step 2: Add `/dashboard` and `/api/auth/config` routes**

After the existing `/developers` route (after line 293), add:

```python
@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    """Serve the API key dashboard (Clerk-gated in the browser)."""
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/auth/config", include_in_schema=False)
def auth_config():
    """Return public Clerk configuration for the frontend."""
    return {"clerk_publishable_key": os.environ.get("CLERK_PUBLISHABLE_KEY", "")}
```

- [ ] **Step 3: Verify the app starts cleanly**

```bash
python -c "from gdelt_event_pipeline.api.app import app; print('OK')"
```

Expected: `OK` with no import errors.

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gdelt_event_pipeline/api/app.py
git commit -m "feat: register keys router, add /dashboard and /api/auth/config routes"
```

---

## Task 8: Create dashboard.html

**Files:**
- Create: `src/gdelt_event_pipeline/api/static/dashboard.html`

- [ ] **Step 1: Create the dashboard page**

Create `src/gdelt_event_pipeline/api/static/dashboard.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GDELT Pulse — Dashboard</title>
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
  .wrap { max-width: 600px; margin: 0 auto; }
  a { color: #7eb8f7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  h1 { font-size: 2rem; font-weight: 600; letter-spacing: -0.03em; margin-bottom: 6px; }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 48px; }
  .subtitle a { color: #555; }
  .card {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 28px 32px;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 20px;
  }
  .key-display {
    font-family: "SF Mono", "Fira Code", Menlo, monospace;
    font-size: 0.9rem;
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c8d9f0;
    word-break: break-all;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .key-value { flex: 1; }
  .copy-btn {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    color: #888;
    cursor: pointer;
    font-size: 0.75rem;
    padding: 4px 10px;
    white-space: nowrap;
    transition: color 0.15s, border-color 0.15s;
  }
  .copy-btn:hover { color: #e0e0e0; border-color: #555; }
  .copy-btn.copied { color: #4caf50; border-color: #4caf50; }
  .warning {
    font-size: 0.8rem;
    color: #e6a817;
    margin-bottom: 20px;
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }
  .key-meta {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 20px;
  }
  .key-meta span { color: #888; }
  .no-key-msg { color: #555; font-size: 0.9rem; margin-bottom: 20px; }
  .actions { display: flex; gap: 10px; flex-wrap: wrap; }
  .btn {
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 9px 18px;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #7eb8f7; color: #0a0a0a; }
  .btn-danger { background: transparent; border: 1px solid #444; color: #e07070; }
  .btn-danger:hover { border-color: #e07070; }
  .divider { border: none; border-top: 1px solid #1a1a1a; margin: 24px 0; }
  .user-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
  }
  .user-email { font-size: 0.85rem; color: #666; }
  .sign-out-btn {
    background: none;
    border: none;
    color: #444;
    cursor: pointer;
    font-size: 0.8rem;
    padding: 0;
  }
  .sign-out-btn:hover { color: #888; }
  #loading { color: #444; font-size: 0.9rem; padding: 40px 0; }
</style>
</head>
<body>
<div class="wrap">
  <p id="loading">Loading…</p>
  <div id="app" style="display:none">
    <div class="user-row">
      <span class="user-email" id="user-email"></span>
      <button class="sign-out-btn" id="sign-out-btn">Sign out</button>
    </div>
    <h1>API Key</h1>
    <p class="subtitle">Use your key in the <code>X-API-Key</code> header. <a href="/developers">API docs →</a></p>

    <!-- State: no active key -->
    <div id="state-no-key" class="card" style="display:none">
      <div class="card-title">Your API Key</div>
      <p class="no-key-msg">You don't have an active key yet.</p>
      <div class="actions">
        <button class="btn btn-primary" id="btn-generate">Generate API Key</button>
      </div>
    </div>

    <!-- State: key just created (one-time reveal) -->
    <div id="state-new-key" class="card" style="display:none">
      <div class="card-title">Your API Key</div>
      <div class="warning">⚠ Save this key now — it won't be shown again.</div>
      <div class="key-display">
        <span class="key-value" id="new-key-value"></span>
        <button class="copy-btn" id="copy-new-key">Copy</button>
      </div>
      <hr class="divider">
      <div class="actions">
        <button class="btn btn-danger" id="btn-revoke-new">Revoke Key</button>
      </div>
    </div>

    <!-- State: existing key -->
    <div id="state-existing-key" class="card" style="display:none">
      <div class="card-title">Your API Key</div>
      <div class="key-display">
        <span class="key-value" id="existing-key-prefix"></span>
      </div>
      <div class="key-meta" id="existing-key-meta"></div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-regenerate">Regenerate Key</button>
        <button class="btn btn-danger" id="btn-revoke-existing">Revoke Key</button>
      </div>
    </div>
  </div>
</div>

<script>
(async () => {
  const loading = document.getElementById('loading');
  const appDiv = document.getElementById('app');

  // 1. Fetch Clerk publishable key from backend config
  let publishableKey;
  try {
    const res = await fetch('/api/auth/config');
    const cfg = await res.json();
    publishableKey = cfg.clerk_publishable_key;
  } catch (e) {
    loading.textContent = 'Failed to load configuration.';
    return;
  }

  if (!publishableKey) {
    loading.textContent = 'Auth is not configured on this server.';
    return;
  }

  // 2. Load Clerk JS SDK from CDN
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@clerk/clerk-js@5/dist/clerk.browser.js';
    s.crossOrigin = 'anonymous';
    s.onload = resolve;
    s.onerror = () => reject(new Error('Failed to load Clerk JS'));
    document.head.appendChild(s);
  });

  // 3. Initialize and load Clerk
  const clerk = new window.Clerk(publishableKey);
  await clerk.load();

  if (!clerk.user) {
    clerk.redirectToSignIn({ redirectUrl: window.location.href });
    return;
  }

  // 4. Show the app
  loading.style.display = 'none';
  appDiv.style.display = 'block';
  document.getElementById('user-email').textContent = clerk.user.primaryEmailAddress?.emailAddress || '';
  document.getElementById('sign-out-btn').addEventListener('click', async () => {
    await clerk.signOut();
    window.location.href = '/developers';
  });

  // Helper: get a fresh session token for API calls
  async function getToken() {
    return clerk.session.getToken();
  }

  // Helper: authenticated fetch
  async function apiFetch(path, options = {}) {
    const token = await getToken();
    return fetch(path, {
      ...options,
      headers: { 'Authorization': `Bearer ${token}`, ...(options.headers || {}) },
    });
  }

  // 5. UI state management
  const stateNoKey = document.getElementById('state-no-key');
  const stateNewKey = document.getElementById('state-new-key');
  const stateExistingKey = document.getElementById('state-existing-key');

  function showState(name) {
    stateNoKey.style.display = name === 'no-key' ? '' : 'none';
    stateNewKey.style.display = name === 'new-key' ? '' : 'none';
    stateExistingKey.style.display = name === 'existing-key' ? '' : 'none';
  }

  function formatDate(iso) {
    if (!iso) return 'never';
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function formatRelative(iso) {
    if (!iso) return 'never';
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  // 6. Load current key state
  async function loadKeyState() {
    try {
      const res = await apiFetch('/api/auth/keys');
      const data = await res.json();
      if (data.active) {
        const prefix = document.getElementById('existing-key-prefix');
        prefix.textContent = data.prefix + '••••••••••••••••••••••••';
        const meta = document.getElementById('existing-key-meta');
        meta.innerHTML = `Created <span>${formatDate(data.created_at)}</span> · Last used <span>${formatRelative(data.last_used_at)}</span>`;
        showState('existing-key');
      } else {
        showState('no-key');
      }
    } catch (e) {
      showState('no-key');
    }
  }

  await loadKeyState();

  // 7. Copy button
  document.getElementById('copy-new-key').addEventListener('click', function () {
    const key = document.getElementById('new-key-value').textContent;
    navigator.clipboard.writeText(key).then(() => {
      this.textContent = 'Copied!';
      this.classList.add('copied');
      setTimeout(() => { this.textContent = 'Copy'; this.classList.remove('copied'); }, 2000);
    });
  });

  // 8. Generate key
  document.getElementById('btn-generate').addEventListener('click', async function () {
    this.disabled = true;
    this.textContent = 'Generating…';
    try {
      const res = await apiFetch('/api/auth/keys', { method: 'POST' });
      const data = await res.json();
      document.getElementById('new-key-value').textContent = data.key;
      showState('new-key');
    } catch (e) {
      alert('Failed to generate key. Please try again.');
    } finally {
      this.disabled = false;
      this.textContent = 'Generate API Key';
    }
  });

  // 9. Regenerate key (from existing-key state)
  document.getElementById('btn-regenerate').addEventListener('click', async function () {
    if (!confirm('This will revoke your current key. All integrations using it will stop working. Continue?')) return;
    this.disabled = true;
    this.textContent = 'Regenerating…';
    try {
      const res = await apiFetch('/api/auth/keys', { method: 'POST' });
      const data = await res.json();
      document.getElementById('new-key-value').textContent = data.key;
      showState('new-key');
    } catch (e) {
      alert('Failed to regenerate key. Please try again.');
    } finally {
      this.disabled = false;
      this.textContent = 'Regenerate Key';
    }
  });

  // 10. Revoke key (from new-key state)
  document.getElementById('btn-revoke-new').addEventListener('click', async function () {
    if (!confirm('Revoke this key? You can generate a new one at any time.')) return;
    this.disabled = true;
    try {
      await apiFetch('/api/auth/keys', { method: 'DELETE' });
      showState('no-key');
    } catch (e) {
      alert('Failed to revoke key. Please try again.');
    } finally {
      this.disabled = false;
    }
  });

  // 11. Revoke key (from existing-key state)
  document.getElementById('btn-revoke-existing').addEventListener('click', async function () {
    if (!confirm('Revoke your API key? All integrations using it will stop working immediately.')) return;
    this.disabled = true;
    try {
      await apiFetch('/api/auth/keys', { method: 'DELETE' });
      showState('no-key');
    } catch (e) {
      alert('Failed to revoke key. Please try again.');
    } finally {
      this.disabled = false;
    }
  });
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the route exists**

```bash
python -c "
from unittest.mock import patch
patch('gdelt_event_pipeline.api.app.init_pool').start()
patch('gdelt_event_pipeline.api.app.close_pool').start()
patch('gdelt_event_pipeline.api.app._ensure_schema').start()
from fastapi.testclient import TestClient
from gdelt_event_pipeline.api.app import app
with TestClient(app) as c:
    r = c.get('/dashboard')
    print(r.status_code, r.headers.get('content-type', ''))
"
```

Expected: `200 text/html; charset=utf-8`

- [ ] **Step 3: Commit**

```bash
git add src/gdelt_event_pipeline/api/static/dashboard.html
git commit -m "feat: add Clerk-gated dashboard.html for API key management"
```

---

## Task 9: Update developers.html with Get API Key CTA

**Files:**
- Modify: `src/gdelt_event_pipeline/api/static/developers.html`

- [ ] **Step 1: Find the first `<h2>` tag in developers.html**

Open `src/gdelt_event_pipeline/api/static/developers.html`. Find the first `<h2>` tag (the first section heading, e.g. `<h2>Quick Start</h2>` or similar).

- [ ] **Step 2: Insert the CTA block before the first `<h2>`**

Insert the following HTML immediately before the first `<h2>` tag:

```html
  <div style="background:#111;border:1px solid #222;border-radius:8px;padding:20px 24px;margin-bottom:32px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <div>
      <div style="font-weight:600;margin-bottom:4px;">Get your API key</div>
      <div style="color:#666;font-size:0.85rem;">Create an account to unlock a higher rate limit (200 req/60s vs 30 for anonymous).</div>
    </div>
    <a href="/dashboard" style="background:#7eb8f7;color:#0a0a0a;font-weight:500;font-size:0.875rem;padding:9px 18px;border-radius:6px;white-space:nowrap;text-decoration:none;">Get API Key →</a>
  </div>
```

- [ ] **Step 3: Verify the developers page still loads**

```bash
python -c "
from unittest.mock import patch
patch('gdelt_event_pipeline.api.app.init_pool').start()
patch('gdelt_event_pipeline.api.app.close_pool').start()
patch('gdelt_event_pipeline.api.app._ensure_schema').start()
from fastapi.testclient import TestClient
from gdelt_event_pipeline.api.app import app
with TestClient(app) as c:
    r = c.get('/developers')
    print(r.status_code, 'dashboard' in r.text)
"
```

Expected: `200 True`

- [ ] **Step 4: Commit**

```bash
git add src/gdelt_event_pipeline/api/static/developers.html
git commit -m "feat: add Get API Key CTA to developers page"
```

---

## Task 10: Update .env.example with Clerk vars

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add Clerk env vars to .env.example**

Append to `.env.example`:

```bash
# Clerk — authentication for the /dashboard and /api/auth/keys endpoints
# Create a Clerk app at https://clerk.com, then copy the keys from the API Keys section.
CLERK_PUBLISHABLE_KEY=pk_test_your-publishable-key-here
CLERK_SECRET_KEY=sk_test_your-secret-key-here
# JWKS URL: Clerk dashboard → API Keys → Advanced → JWT verification
# Format: https://<your-clerk-frontend-api>/.well-known/jwks.json
CLERK_JWKS_URL=https://your-clerk-domain.clerk.accounts.dev/.well-known/jwks.json
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add Clerk environment variables to .env.example"
```

---

## Post-Implementation Checklist

Before declaring this complete, verify:

- [ ] `pytest tests/ -v` — all tests green
- [ ] `python -c "from gdelt_event_pipeline.api.app import app; print('OK')"` — no import errors
- [ ] `/developers` returns 200 with "Get API Key" CTA visible
- [ ] `/dashboard` returns 200 (HTML page with Clerk JS)
- [ ] `GET /api/auth/keys` without Authorization header returns 422 (not 401 from middleware)
- [ ] `GET /api/stats` without X-API-Key returns 200 (not 401)
- [ ] `GET /api/stats` with invalid X-API-Key returns 401 (requires DB mock in manual test)
- [ ] `GET /api/auth/config` returns `{"clerk_publishable_key": ""}` when env var not set

---

## Clerk Setup (Human steps, after code ships)

1. Create a Clerk app at https://clerk.com
2. Enable Google and GitHub OAuth in the Clerk dashboard
3. Copy `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, and the JWKS URL
4. Add them as environment variables on Railway/Vercel
5. Set the allowed redirect URL in Clerk to `<your-domain>/dashboard`
