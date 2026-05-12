"""Scrape HTML <title> tags for articles missing titles."""

from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds — used only as a fallback for the legacy timeout kwarg
MAX_WORKERS = 64  # max concurrent threads for scraping
USER_AGENT = "gdelt-pulse/0.1"  # identify ourselves when fetching pages
MAX_READ_BYTES = 64 * 1024  # only read first 64KB — title is always near the top

# Granular per-phase timeouts. urllib's single `timeout=` only covered the connect
# + first response phase, so a server that was slow to send body bytes could stall
# a worker thread arbitrarily long. With httpx we cap each phase independently —
# worst case ~20s, but average much less.
_HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)

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


def _fetch_title(client: httpx.Client, url: str) -> str | None:
    """Fetch a page and extract the <title> tag via regex.

    Streams the response and stops after MAX_READ_BYTES so a server with a huge
    body can't keep a worker busy. Returns the cleaned title string, or None on
    any failure.
    """
    try:
        with client.stream("GET", url) as resp:
            # Collect at most MAX_READ_BYTES of body data.
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_READ_BYTES:
                    break
            raw = b"".join(chunks)[:MAX_READ_BYTES]
            resp_charset = resp.charset_encoding

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
        timeout: per-request total timeout in seconds. Kept for backwards
            compatibility — httpx uses granular per-phase timeouts instead, so
            this only applies if the caller passes a smaller value than the
            default phase budget.
        max_workers: max concurrent threads.

    Returns:
        dict mapping article_id (str) → scraped title.
        Only includes articles where a title was successfully extracted.
    """
    results: dict[str, str] = {}

    # If the caller asked for a tighter overall budget than our per-phase
    # defaults imply (>20s worst case), shrink the phase timeouts proportionally
    # so the call still respects their intent.
    if timeout < DEFAULT_TIMEOUT:
        phase = max(1.0, float(timeout) / 2.0)
        request_timeout = httpx.Timeout(
            connect=phase, read=phase, write=phase, pool=phase
        )
    else:
        request_timeout = _HTTPX_TIMEOUT

    # Single shared client across worker threads — httpx.Client is thread-safe
    # and gives us connection pooling for free.
    with httpx.Client(
        timeout=request_timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_id = {
                pool.submit(_fetch_title, client, article["url"]): str(article["id"])
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
