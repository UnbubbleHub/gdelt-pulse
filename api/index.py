"""Vercel ASGI entry point.

Adds src/ to sys.path so the gdelt_event_pipeline package is importable
without a package install step, then re-exports the FastAPI app object
that Vercel's runtime uses as the ASGI handler.
"""

import sys
from pathlib import Path

# src/ is not on sys.path in Vercel's runtime environment.
# Insert it so `from gdelt_event_pipeline...` imports resolve correctly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gdelt_event_pipeline.api.app import app  # noqa: E402 — intentional late import

__all__ = ["app"]
