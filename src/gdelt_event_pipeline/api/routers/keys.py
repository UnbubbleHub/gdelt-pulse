"""API key management endpoints — requires Clerk JWT auth."""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import UTC, datetime

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
    random_part = secrets.token_hex(16)
    full_key = f"gdp_{random_part}"
    prefix = full_key[:_KEY_PREFIX_LEN]
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, prefix, key_hash


@router.get("/config", include_in_schema=False)
def auth_config():
    """Return public Clerk configuration for the frontend."""
    return {"clerk_publishable_key": os.environ.get("CLERK_PUBLISHABLE_KEY", "")}


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
    now = datetime.now(tz=UTC)
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
    now = datetime.now(tz=UTC)
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
