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
