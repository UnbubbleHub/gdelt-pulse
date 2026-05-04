# Contributing to GDELT Pulse

Thank you for your interest in contributing to GDELT Pulse! This project is part of the [UnbubbleHub](https://github.com/UnbubbleHub) open-source ecosystem.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/gdelt-pulse.git
   cd gdelt-pulse
   ```
3. **Install dependencies** with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```
4. **Set up your environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your PostgreSQL credentials
   ```

See the [Getting Started Guide](docs/getting-started.md) for detailed setup instructions including PostgreSQL and pgvector installation.

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes
3. Run the test suite and linter:
   ```bash
   uv run --group api pytest tests/ -q
   ruff check .
   ruff format . --check
   ```
4. Commit your changes with a descriptive message
5. Push to your fork and open a pull request against `main`

## Code Style

- **Linter/formatter**: [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **Target version**: Python 3.11
- **Import sorting**: Handled by ruff (`isort`-compatible)

Run before committing:

```bash
ruff check . --fix   # auto-fix lint issues
ruff format .        # auto-format
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`. Install them with:

```bash
uv run pre-commit install
```

## Testing

Tests mirror the `src/` directory structure under `tests/`. The project's tests cover all layers:

```
tests/
├── api/            # Endpoint tests (search, clusters, articles, auth, keys)
├── clustering/     # Cluster assignment, scoring, centroid computation
├── config/         # Settings and configuration
├── embeddings/     # Embedding generation and text composition
├── ingestion/      # GKG fetching, parsing, scraping
├── normalization/  # URL, source, and field normalization
├── pipeline/       # Continuous runner
├── query/          # Search: vector, keyword, ranking, filters
└── storage/        # Database operations (articles, clusters, state)
```

Running tests:

```bash
# Full suite
uv run --group api pytest tests/ -q

# Single module
uv run --group api pytest tests/api/ -v

# Single test
uv run --group api pytest tests/api/test_search.py::TestSearch::test_happy_path
```

When adding new functionality, write tests in the corresponding `tests/` subdirectory. Shared fixtures and factories are in `tests/conftest.py`.

## What to Contribute

- **Bug reports** -- Open an [issue](https://github.com/UnbubbleHub/gdelt-pulse/issues) with steps to reproduce
- **Bug fixes** -- Reference the issue number in your PR
- **New features** -- Open an issue or [discussion](https://github.com/UnbubbleHub/gdelt-pulse/discussions) first to align on scope
- **Documentation** -- Improvements to docs are always welcome
- **Tests** -- Expanding coverage is a great way to start contributing

## Pull Request Guidelines

- Keep PRs focused -- one concern per PR
- Include tests for new functionality
- Ensure CI passes (tests + lint + format)
- Write a clear PR description explaining what and why
- Reference related issues with `Fixes #N` or `Closes #N`

## Architecture Overview

Before diving in, read the [Architecture Guide](docs/architecture.md) to understand how the pipeline, API, and storage layers connect. The [API Reference](docs/api-reference.md) documents all exposed endpoints.

## Questions?

Open a [Discussion](https://github.com/UnbubbleHub/gdelt-pulse/discussions) on GitHub -- we're happy to help.
