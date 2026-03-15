"""Fetch and parse GDELT GKG files."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2/"
DEFAULT_TIMEOUT = 30
USER_AGENT = "gdelt-pulse/0.1"


@dataclass
class GkgFile:
    """Metadata about a GKG file from the GDELT lastupdate manifest."""
    size: int
    md5: str
    url: str


def fetch_latest_file_list(timeout: int = DEFAULT_TIMEOUT) -> list[GkgFile]:
    """Fetch the GDELT lastupdate.txt and return GKG file entries."""
    req = Request(LAST_UPDATE_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="replace")

    results: list[GkgFile] = []
    for line in payload.splitlines():
        parts = line.split()
        if len(parts) >= 3 and ".gkg.csv.zip" in parts[2]:
            results.append(GkgFile(
                size=int(parts[0]),
                md5=parts[1],
                url=parts[2],
            ))
    return results


def get_latest_gkg_url(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return the URL of the most recent GKG file."""
    files = fetch_latest_file_list(timeout)
    if not files:
        raise RuntimeError("No GKG file found in lastupdate.txt")
    return files[0].url


def download_and_parse_gkg(url: str, timeout: int = DEFAULT_TIMEOUT) -> list[list[str]]:
    """Download a GKG zip file and return all rows as lists of strings."""
    logger.info("Downloading %s", url)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        archive_bytes = response.read()

    logger.info("Downloaded %d bytes", len(archive_bytes))

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_member = next(
            (name for name in archive.namelist() if name.endswith(".csv")),
            None,
        )
        if not csv_member:
            raise RuntimeError(f"No CSV file found in archive from {url}")

        with archive.open(csv_member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter="\t")
            rows = [row for row in reader if row]

    logger.info("Parsed %d rows from %s", len(rows), csv_member)
    return rows
