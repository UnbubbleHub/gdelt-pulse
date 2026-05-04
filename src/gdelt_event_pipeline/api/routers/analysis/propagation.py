"""Story Propagation endpoints — how stories spread across sources over time."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/api/propagation/stories")
def propagation_stories(
    limit: int = Query(20, ge=1, le=50, description="Max clusters to return"),
    min_sources: int = Query(3, ge=2, description="Min distinct sources"),
):
    """Top stories suitable for propagation analysis (multi-source clusters)."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       count(DISTINCT a.domain) AS source_count,
                       EXTRACT(EPOCH FROM (c.last_article_at - c.first_article_at)) / 3600
                           AS span_hours
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id
                WHERE c.is_active = true
                  AND c.article_count >= 5
                  AND a.title IS NOT NULL
                GROUP BY c.id
                HAVING count(DISTINCT a.domain) >= %s
                ORDER BY c.article_count DESC
                LIMIT %s
                """,
                (min_sources, limit),
            )
            stories = cur.fetchall()

    return [
        {
            "id": str(s["id"]),
            "title": s["representative_title"],
            "article_count": s["article_count"],
            "source_count": s["source_count"],
            "span_hours": round(s["span_hours"], 1) if s["span_hours"] else 0,
            "first_article_at": s["first_article_at"],
            "last_article_at": s["last_article_at"],
        }
        for s in stories
    ]


@router.get("/api/propagation/{cluster_id}")
def propagation_timeline(cluster_id: str):
    """Timeline of how a story propagated across sources over time."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Verify cluster exists
            cur.execute(
                "SELECT id, representative_title, article_count, "
                "first_article_at, last_article_at "
                "FROM clusters WHERE id = %s",
                (cluster_id,),
            )
            cluster = cur.fetchone()
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            # All articles in chronological order with source info
            cur.execute(
                """
                SELECT a.title, a.url, a.domain,
                       COALESCE(a.source_common_name, a.domain) AS source_name,
                       a.gdelt_timestamp,
                       (a.tone->>'tone')::float AS tone_score,
                       a.locations
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = %s
                  AND a.title IS NOT NULL
                ORDER BY a.gdelt_timestamp ASC
                """,
                (cluster_id,),
            )
            articles = cur.fetchall()

    if not articles:
        raise HTTPException(status_code=404, detail="No articles found for cluster")

    # Build timeline events
    events = []
    seen_sources = set()
    source_first_seen: dict[str, int] = {}

    for i, a in enumerate(articles):
        domain = a["domain"]
        is_first = domain not in seen_sources
        if is_first:
            source_first_seen[domain] = i
            seen_sources.add(domain)

        # Extract primary location
        loc_name = None
        if a["locations"] and isinstance(a["locations"], list) and a["locations"]:
            loc_name = a["locations"][0].get("name")

        events.append(
            {
                "title": a["title"],
                "url": a["url"],
                "domain": domain,
                "source_name": a["source_name"],
                "timestamp": a["gdelt_timestamp"],
                "tone": (round(a["tone_score"], 2) if a["tone_score"] is not None else None),
                "location": loc_name,
                "is_first_from_source": is_first,
                "order": i,
            }
        )

    # Source summary: when each source first picked up the story
    source_summary = []
    for domain, first_idx in sorted(source_first_seen.items(), key=lambda x: x[1]):
        arts = [e for e in events if e["domain"] == domain]
        source_summary.append(
            {
                "domain": domain,
                "source_name": arts[0]["source_name"],
                "first_timestamp": arts[0]["timestamp"],
                "article_count": len(arts),
                "avg_tone": round(
                    sum(a["tone"] for a in arts if a["tone"] is not None)
                    / max(1, sum(1 for a in arts if a["tone"] is not None)),
                    2,
                ),
                "order": first_idx,
            }
        )

    # Hourly histogram
    if articles:
        first_ts = articles[0]["gdelt_timestamp"]
        if first_ts.tzinfo is None:
            first_ts = first_ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        hourly: dict[int, int] = defaultdict(int)
        for a in articles:
            ts = a["gdelt_timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=first_ts.tzinfo)
            hour = int((ts - first_ts).total_seconds() / 3600)
            hourly[hour] += 1
        max_hour = max(hourly.keys()) if hourly else 0
        histogram = [{"hour": h, "count": hourly.get(h, 0)} for h in range(max_hour + 1)]
    else:
        histogram = []

    return {
        "cluster_id": str(cluster["id"]),
        "title": cluster["representative_title"],
        "article_count": cluster["article_count"],
        "first_article_at": cluster["first_article_at"],
        "last_article_at": cluster["last_article_at"],
        "source_count": len(source_summary),
        "timeline": events,
        "sources": source_summary,
        "hourly_histogram": histogram,
    }
