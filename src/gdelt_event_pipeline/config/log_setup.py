"""Logging configuration.

Text format in dev, JSON in production (detected via VERCEL or
RAILWAY_ENVIRONMENT env vars). JSON output goes to stdout where
Vercel and Railway log collectors pick it up.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
_TEXT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_STD_LOGRECORD_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


class JSONFormatter(logging.Formatter):
    """Emit a single-line JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Anything passed via logger.info("msg", extra={...}) lands as record attrs.
        for key, value in record.__dict__.items():
            if key in _STD_LOGRECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def is_production() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("RAILWAY_ENVIRONMENT"))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)
    # Drop any pre-existing handlers (uvicorn / pytest install their own)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if is_production():
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=_TEXT_DATEFMT))
    root.addHandler(handler)
