"""Rate limiting and API key authentication middleware."""

from __future__ import annotations

import hashlib
import os
import threading as _threading
import time
from collections import defaultdict

from fastapi import Request, Response

RATE_LIMIT_MAX = 30
RATE_LIMIT_MAX_KEY = 200
RATE_LIMIT_WINDOW = 60

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_redis = None
_redis_lock = _threading.Lock()


def _get_redis():
    """Return the Upstash Redis client, or None if not configured."""
    global _redis
    if _redis is None:
        with _redis_lock:
            if _redis is None:
                url = os.environ.get("UPSTASH_REDIS_REST_URL")
                token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
                if url and token:
                    from upstash_redis import Redis

                    _redis = Redis(url=url, token=token)
    return _redis


async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Optional API key auth + per-IP/per-key rate limiting for /api/* paths."""
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    if request.url.path.startswith("/api/auth/"):
        return await call_next(request)

    rate_limit_max = RATE_LIMIT_MAX
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_id = client_ip

    provided_key = request.headers.get("X-API-Key")
    if provided_key:
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        from gdelt_event_pipeline.storage.database import get_pool as _get_pool

        pool = _get_pool()
        with pool.connection() as conn:
            from psycopg.rows import dict_row as _dict_row

            with conn.cursor(row_factory=_dict_row) as cur:
                cur.execute(
                    "SELECT id, user_id FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
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

        _threading.Thread(target=_update_last_used, daemon=True).start()

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
