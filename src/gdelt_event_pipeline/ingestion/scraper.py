"""Scrape HTML <title> tags for articles missing titles."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
MAX_WORKERS = 8
USER_AGENT = "gdelt-pulse/0.1"
MAX_READ_BYTES = 64 * 1024  # only read first 64KB — title is always near the top

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _fetch_title(url: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """Fetch a page and extract the <title> tag via regex.

    Returns the cleaned title string, or None on failure.
    """
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_READ_BYTES)

        # Try utf-8 first, fall back to latin-1
        for encoding in ("utf-8", "latin-1"):
            try:
                html = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return None

        match = _TITLE_RE.search(html)
        if not match:
            return None

        title = match.group(1).strip()
        # Collapse whitespace
        title = re.sub(r"\s+", " ", title)
        # Strip common HTML entities
        title = (
            title.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&apos;", "'")
        )
        return title if title else None

    except Exception:
        logger.debug("Failed to fetch title for %s", url, exc_info=True)
        return None


def scrape_titles(
    articles: list[dict[str, Any]],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_workers: int = MAX_WORKERS,
) -> dict[str, str]:
    """Scrape titles for a batch of articles concurrently.

    Args:
        articles: list of article dicts, each must have 'id' and 'url' keys.
        timeout: per-request timeout in seconds.
        max_workers: max concurrent threads.

    Returns:
        dict mapping article_id (str) → scraped title.
        Only includes articles where a title was successfully extracted.
    """
    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(_fetch_title, article["url"], timeout): str(article["id"])
            for article in articles
        }

        for future in as_completed(future_to_id):
            article_id = future_to_id[future]
            title = future.result()
            if title:
                results[article_id] = title

    logger.info(
        "Scraped titles: %d/%d successful",
        len(results),
        len(articles),
    )
    return results
