-- gdelt-pulse database schema
-- PostgreSQL + pgvector

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================
-- articles
-- Main input table. One record per unique canonical URL.
-- ============================================================
CREATE TABLE articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gkg_record_id   TEXT UNIQUE NOT NULL,
    gdelt_timestamp TIMESTAMPTZ NOT NULL,
    url             TEXT NOT NULL,
    canonical_url   TEXT UNIQUE NOT NULL,
    domain          TEXT NOT NULL,
    source_common_name TEXT,
    canonical_source   TEXT,
    title           TEXT,

    -- GKG metadata (stored structured, not raw strings)
    themes          JSONB,
    locations       JSONB,
    organizations   JSONB,
    persons         JSONB,
    all_names       JSONB,
    tone            JSONB,

    -- Scraping
    scrape_attempts INTEGER NOT NULL DEFAULT 0,

    -- Embedding
    embedding       vector,
    embedding_model TEXT,

    -- Raw GKG payload for debugging / re-parsing
    raw_payload     JSONB,

    -- Full-text search (auto-maintained by PostgreSQL)
    title_tsv       tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', COALESCE(title, ''))) STORED,

    -- Timestamps
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for common query patterns
CREATE INDEX idx_articles_gdelt_timestamp ON articles (gdelt_timestamp);
CREATE INDEX idx_articles_domain          ON articles (domain);
CREATE INDEX idx_articles_canonical_source ON articles (canonical_source);
CREATE INDEX idx_articles_created_at      ON articles (created_at);

-- GIN indexes for JSONB metadata queries
CREATE INDEX idx_articles_themes        ON articles USING gin (themes);
CREATE INDEX idx_articles_locations     ON articles USING gin (locations);
CREATE INDEX idx_articles_organizations ON articles USING gin (organizations);
CREATE INDEX idx_articles_persons       ON articles USING gin (persons);

-- Full-text search on titles
CREATE INDEX idx_articles_title_tsv ON articles USING gin (title_tsv);

-- Vector similarity search (HNSW for approximate nearest neighbor)
-- Dimension must match your embedding model; set to 1536 as a common default.
-- Adjust if using a different model.
CREATE INDEX idx_articles_embedding ON articles USING hnsw (embedding vector_cosine_ops);


-- ============================================================
-- clusters
-- Each cluster represents one real-world event entity.
-- ============================================================
CREATE TABLE clusters (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    representative_title TEXT,
    summary             TEXT,
    centroid_embedding   vector,
    article_count       INTEGER NOT NULL DEFAULT 0,
    first_article_at    TIMESTAMPTZ,
    last_article_at     TIMESTAMPTZ,
    is_active           BOOLEAN NOT NULL DEFAULT true,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_clusters_is_active       ON clusters (is_active);
CREATE INDEX idx_clusters_last_article_at ON clusters (last_article_at);

-- Vector search on cluster centroids
CREATE INDEX idx_clusters_centroid ON clusters USING hnsw (centroid_embedding vector_cosine_ops);


-- ============================================================
-- cluster_memberships
-- Links articles to clusters. Explicit and traceable.
-- ============================================================
CREATE TABLE cluster_memberships (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id        UUID NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    cluster_id        UUID NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
    similarity_score  DOUBLE PRECISION,
    assignment_method TEXT,
    assigned_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (article_id, cluster_id)
);

CREATE INDEX idx_memberships_article ON cluster_memberships (article_id);
CREATE INDEX idx_memberships_cluster ON cluster_memberships (cluster_id);


-- ============================================================
-- pipeline_state
-- Tracks incremental ingestion progress per source.
-- ============================================================
CREATE TABLE pipeline_state (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name                 TEXT UNIQUE NOT NULL,
    last_processed_timestamp    TIMESTAMPTZ,
    last_processed_record_id    TEXT,
    last_successful_run_at      TIMESTAMPTZ,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed the initial source entry
INSERT INTO pipeline_state (source_name)
VALUES ('gdelt_gkg');
