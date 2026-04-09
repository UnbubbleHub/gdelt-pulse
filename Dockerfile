FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY sql/ sql/
RUN uv sync --frozen --no-dev

# Download the embedding model at build time so it's baked into the image
# (avoids downloading ~90MB on every container start)
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Expose the API port (Railway reads this)
EXPOSE 8000

# Run both the pipeline and the API server
CMD ["sh", "-c", "uv run python -m gdelt_event_pipeline.runner & uv run uvicorn gdelt_event_pipeline.api.app:app --host 0.0.0.0 --port ${PORT:-8000} & wait"]
