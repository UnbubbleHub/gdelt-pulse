# gdelt-pulse

A backend pipeline for ingesting GDELT-derived news records, clustering them into evolving event entities, and making those events queryable.

This project is developed as part of the UnbubbleHub open-source ecosystem.

---

## Project Goal

Transform the continuous GDELT news stream into a structured event index.

The pipeline converts document-level signals into event-level entities.

GDELT records -> normalized articles -> clustered events -> queryable event database

---

## Core Idea

GDELT provides structured metadata about global news coverage but does not directly provide semantic news story clusters.

This project builds an incremental event-construction layer on top of GDELT.

Each event represents a group of related articles describing the same real-world development.

---

## Pipeline Overview

1. Ingestion  
	Fetch GDELT-derived records continuously.

2. Normalization  
	Normalize URLs, sources, and article metadata.

3. Storage  
	Store article records in PostgreSQL.

4. Embedding  
	Generate vector embeddings for article content.

5. Clustering  
	Assign incoming articles to existing event clusters or create new clusters.

6. Query Layer  
	Retrieve events and the articles supporting them.

---

## Initial Scope

Phase 1 focuses on building a reliable backend:

- incremental GDELT ingestion
- normalized article storage
- event clustering
- PostgreSQL + pgvector
- event querying

Later phases may add:

- event analytics
- coverage analysis
- narrative comparison
- APIs

---

## Architecture

GDELT -> ingestion -> normalization -> database -> embeddings -> clustering -> event store -> queries

---

## Repository Structure

```text
src/gdelt_event_pipeline/
  ingestion/
  normalization/
  storage/
  embeddings/
  clustering/
  query/
  pipeline/
```

---

## First Milestone

Build a working incremental pipeline:

1. ingest GDELT-derived records
2. normalize and store articles
3. generate embeddings
4. cluster articles into events
5. query events and their articles

---

## Development

This project is configured to work well with `uv`.

```bash
uv sync
uv run pytest
uv run ruff check .
```
