"""Source name normalization."""

from __future__ import annotations

import re

# Known mappings from messy GDELT source names to canonical identifiers.
# Extend this as you encounter new sources.
_SOURCE_ALIASES: dict[str, str] = {
    "nytimes.com": "new_york_times",
    "nyt.com": "new_york_times",
    "bbc.co.uk": "bbc",
    "bbc.com": "bbc",
    "reuters.com": "reuters",
    "apnews.com": "associated_press",
    "ap.org": "associated_press",
    "washingtonpost.com": "washington_post",
    "theguardian.com": "the_guardian",
    "guardian.co.uk": "the_guardian",
    "cnn.com": "cnn",
    "aljazeera.com": "al_jazeera",
    "aljazeera.net": "al_jazeera",
    "foxnews.com": "fox_news",
    "nbcnews.com": "nbc_news",
    "cbsnews.com": "cbs_news",
    "abcnews.go.com": "abc_news",
    "france24.com": "france24",
    "dw.com": "deutsche_welle",
}

_DOMAIN_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)


def normalize_source(source_common_name: str | None, domain: str | None) -> str | None:
    """Return a canonical source identifier.

    Tries the domain first (more reliable), then falls back to
    the GDELT SourceCommonName. Returns None if both are empty.
    """
    if domain:
        clean_domain = domain.lower().strip()
        if clean_domain in _SOURCE_ALIASES:
            return _SOURCE_ALIASES[clean_domain]

    if source_common_name:
        # GDELT sometimes passes domains as the source name
        cleaned = _DOMAIN_PATTERN.sub("", source_common_name.strip()).rstrip("/").lower()
        if cleaned in _SOURCE_ALIASES:
            return _SOURCE_ALIASES[cleaned]
        # Fall back to a slug of the raw name
        return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_") or None

    if domain:
        return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_") or None

    return None
