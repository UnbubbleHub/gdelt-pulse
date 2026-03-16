"""Browse the first N articles with all stored info."""

from __future__ import annotations

import json
import sys

import psycopg
from psycopg.rows import dict_row

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.storage.database import close_pool, init_pool, get_pool


LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEP = "─" * 100


def _fmt_json(val, max_items: int = 5) -> str:
    """Pretty-print a JSONB value, truncating long arrays."""
    if val is None:
        return "(none)"
    if isinstance(val, list):
        shown = val[:max_items]
        extra = f"  … +{len(val) - max_items} more" if len(val) > max_items else ""
        return json.dumps(shown, ensure_ascii=False, indent=2) + extra
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, indent=2)
    return str(val)


def main() -> None:
    settings = get_settings()
    init_pool(settings.db)
    pool = get_pool()

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM articles
                ORDER BY gdelt_timestamp ASC
                LIMIT %s
                """,
                (LIMIT,),
            )
            rows = cur.fetchall()

    print(f"\n  Browsing first {len(rows)} articles (ordered by gdelt_timestamp)\n")
    print(SEP)

    for i, a in enumerate(rows, 1):
        title = a.get("title") or "(no title scraped)"
        tone_val = a["tone"].get("tone", "?") if a.get("tone") else "?"
        themes = a.get("themes") or []
        locations = a.get("locations") or []
        persons = a.get("persons") or []
        orgs = a.get("organizations") or []

        print(f"  [{i:>3}]  {title}")
        print(f"         URL:       {a['canonical_url']}")
        print(f"         Domain:    {a['domain']}  |  Source: {a['canonical_source']}")
        print(f"         GKG ID:    {a['gkg_record_id']}")
        print(f"         Timestamp: {a['gdelt_timestamp']}")
        print(f"         Tone:      {tone_val}")
        print(f"         Themes ({len(themes):>3}): "
              f"{', '.join(t['theme'] for t in themes[:6])}"
              f"{'…' if len(themes) > 6 else ''}")
        print(f"         Locations ({len(locations):>2}): "
              f"{', '.join(loc['name'] for loc in locations[:5])}"
              f"{'…' if len(locations) > 5 else ''}")
        print(f"         Persons ({len(persons):>2}): "
              f"{', '.join(persons[:5])}"
              f"{'…' if len(persons) > 5 else ''}")
        print(f"         Orgs    ({len(orgs):>2}): "
              f"{', '.join(orgs[:5])}"
              f"{'…' if len(orgs) > 5 else ''}")
        print(SEP)

    close_pool()


if __name__ == "__main__":
    main()
