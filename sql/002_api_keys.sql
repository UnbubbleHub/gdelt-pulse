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
