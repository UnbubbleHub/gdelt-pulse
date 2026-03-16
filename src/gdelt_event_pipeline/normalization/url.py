"""URL canonicalization for article deduplication."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


# Query parameters that are tracking/analytics noise, not part of article identity.
_STRIP_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_cid", "utm_reader",
    "fbclid", "gclid", "gclsrc", "dclid",
    "mc_cid", "mc_eid",
    "ref", "referrer", "source",
    "ncid", "ocid", "sr_share",
    "_ga", "_gl",
})


def canonicalize_url(raw_url: str) -> str:
    """Produce a canonical URL suitable for deduplication.

    - Lowercases scheme and host
    - Strips tracking query parameters
    - Removes fragment
    - Strips trailing slash from path (except root "/")
    - Removes default ports (80 for http, 443 for https)
    """
    raw_url = raw_url.strip()
    if not raw_url:
        return ""

    parsed = urlparse(raw_url)

    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()

    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Remove www. prefix
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parsed.path
    # Strip trailing slash unless it's the root
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Filter tracking query parameters
    query_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {
        k: v for k, v in query_params.items() if k.lower() not in _STRIP_PARAMS
    }
    query = urlencode(filtered, doseq=True) if filtered else ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def extract_domain(url: str) -> str:
    """Extract the domain (netloc without www.) from a URL."""
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Strip port
    if ":" in netloc:
        netloc = netloc.rsplit(":", 1)[0]
    return netloc
