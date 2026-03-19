"""Scrape HTML <title> tags for articles missing titles."""

from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds per request
MAX_WORKERS = 8  # max concurrent threads for scraping
USER_AGENT = "gdelt-pulse/0.1"  # identify ourselves when fetching pages
MAX_READ_BYTES = 64 * 1024  # only read first 64KB — title is always near the top

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_CHARSET_RE = re.compile(r'<meta[^>]+charset=["\']?([^"\'\s;>]+)', re.IGNORECASE)
_OG_TITLE_RE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _detect_encoding(raw: bytes, resp_charset: str | None) -> str:
    """Pick the best encoding from response headers and meta tags."""
    if resp_charset:
        return resp_charset
    # Peek at the first 2KB for a <meta charset> declaration
    head = raw[:2048].decode("ascii", errors="ignore")
    match = _META_CHARSET_RE.search(head)
    if match:
        return match.group(1).strip()
    return "utf-8"


def _fetch_title(url: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """Fetch a page and extract the <title> tag via regex.

    Returns the cleaned title string, or None on failure.
    """
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_READ_BYTES)
            resp_charset = resp.headers.get_content_charset()

        encoding = _detect_encoding(raw, resp_charset)

        # Try detected encoding first, then utf-8, then latin-1
        text: str | None = None
        for enc in dict.fromkeys([encoding, "utf-8", "latin-1"]):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            return None

        title = _extract_title(text)
        return title if title else None

    except Exception:
        logger.debug("Failed to fetch title for %s", url, exc_info=True)
        return None


def _extract_title(text: str) -> str | None:
    """Extract and clean a title from HTML text.

    Tries <title> tag first, then falls back to og:title meta tag.
    """
    match = _TITLE_RE.search(text)
    if not match:
        match = _OG_TITLE_RE.search(text)
    if not match:
        return None

    title = match.group(1).strip()
    title = re.sub(r"\s+", " ", title)
    title = html.unescape(title)
    return title if title else None


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
