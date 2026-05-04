"""Globe clusters endpoint — top clusters with geographic coordinates for the 3D globe."""

from __future__ import annotations

from collections import Counter, defaultdict

from fastapi import APIRouter, Query

from gdelt_event_pipeline.api.routers.analysis._helpers import _categorize_themes

router = APIRouter()


@router.get("/api/globe/clusters")
def globe_clusters(
    mode: str = Query("live", description="Filter mode: live, rising, silent"),
    limit: int = Query(12, ge=1, le=50, description="Max clusters"),
):
    """Return top clusters with geographic coordinates for the 3D globe.

    Modes:
    - live:   Clusters with articles in the last 2 hours, sorted by article_count.
    - rising: Clusters whose article count grew fastest in the last 6 hours.
    - silent: Large clusters that have gone quiet (no new articles for 6-72h).
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Use the latest article timestamp as reference instead of now(),
            # because the database is a static snapshot (not live-fetched).
            cur.execute("SELECT MAX(gdelt_timestamp) FROM articles")
            ref_time = cur.fetchone()["max"]
            if ref_time is None:
                return []

            if mode == "rising":
                # Stories picking up steam: clusters that existed for a while
                # but got a burst of new articles in the last 2 hours.
                # Ranked by recent article count — the surge signal.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           recent.recent_count,
                           CASE WHEN c.article_count > 0
                                THEN recent.recent_count::float / c.article_count
                                ELSE 0 END AS velocity,
                           EXTRACT(EPOCH FROM (%s - c.first_article_at)) / 3600
                               AS age_hours
                    FROM clusters c
                    JOIN (
                        SELECT cm.cluster_id, count(*) AS recent_count
                        FROM cluster_memberships cm
                        JOIN articles a ON a.id = cm.article_id
                        WHERE a.gdelt_timestamp >= %s - interval '2 hours'
                        GROUP BY cm.cluster_id
                        HAVING count(*) >= 2
                    ) recent ON recent.cluster_id = c.id
                    WHERE c.is_active = true
                      AND c.article_count >= 5
                      AND c.first_article_at <= %s - interval '1 hour'
                    ORDER BY recent.recent_count DESC, velocity DESC
                    LIMIT %s
                    """,
                    (ref_time, ref_time, ref_time, limit),
                )
            elif mode == "silent":
                # Big stories that went quiet — no new articles in 3h+
                # but were active within the last 48h.
                # Shows hours since last article so the UI can display
                # "quiet for 8h" etc.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           0 AS recent_count, 0 AS velocity,
                           EXTRACT(EPOCH FROM (%s - c.last_article_at)) / 3600
                               AS silent_hours
                    FROM clusters c
                    WHERE c.is_active = true
                      AND c.article_count >= 5
                      AND c.last_article_at < %s - interval '6 hours'
                      AND c.last_article_at >= %s - interval '72 hours'
                    ORDER BY c.article_count DESC
                    LIMIT %s
                    """,
                    (ref_time, ref_time, ref_time, limit),
                )
            else:
                # live: what newsrooms are writing about RIGHT NOW.
                # Clusters ranked by how many articles arrived in the
                # last 2 hours — the hottest stories this moment.
                cur.execute(
                    """
                    SELECT c.id, c.representative_title, c.article_count,
                           c.first_article_at, c.last_article_at,
                           c.created_at, c.updated_at,
                           COALESCE(recent.recent_count, 0) AS recent_count,
                           0 AS velocity
                    FROM clusters c
                    JOIN (
                        SELECT cm.cluster_id, count(*) AS recent_count
                        FROM cluster_memberships cm
                        JOIN articles a ON a.id = cm.article_id
                        WHERE a.gdelt_timestamp >= %s - interval '2 hours'
                        GROUP BY cm.cluster_id
                    ) recent ON recent.cluster_id = c.id
                    WHERE c.is_active = true
                      AND c.article_count >= 2
                    ORDER BY recent.recent_count DESC, c.article_count DESC
                    LIMIT %s
                    """,
                    (ref_time, limit),
                )

            clusters = cur.fetchall()
            cluster_ids = [c["id"] for c in clusters]

            if not cluster_ids:
                return []

            # Fetch primary location (most common lat/lon) for each cluster
            cur.execute(
                """
                SELECT cm.cluster_id,
                       a.locations,
                       a.themes,
                       a.title,
                       a.url,
                       a.domain,
                       a.gdelt_timestamp
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.locations IS NOT NULL
                  AND jsonb_array_length(a.locations) > 0
                ORDER BY a.gdelt_timestamp DESC
                """,
                (cluster_ids,),
            )
            article_rows = cur.fetchall()

    # Build location + metadata per cluster
    cluster_locations: dict[str, list[dict]] = defaultdict(list)
    cluster_articles_data: dict[str, list[dict]] = defaultdict(list)

    for row in article_rows:
        cid = str(row["cluster_id"])
        locs = row["locations"]
        if isinstance(locs, list):
            for loc in locs:
                if loc.get("lat") is not None and loc.get("lon") is not None:
                    cluster_locations[cid].append(loc)
        cluster_articles_data[cid].append(
            {
                "title": row["title"],
                "url": row["url"],
                "domain": row["domain"],
                "gdelt_timestamp": row["gdelt_timestamp"],
                "themes": row["themes"],
            }
        )

    results = []
    for c in clusters:
        cid = str(c["id"])
        locs = cluster_locations.get(cid, [])

        # Pick the most common location as primary
        lat, lon, location_name, country_code = None, None, None, None
        if locs:
            # Count occurrences of each (lat, lon) rounded to 1 decimal
            coord_counts = Counter()
            coord_info: dict[tuple, dict] = {}
            for loc in locs:
                key = (round(loc["lat"], 1), round(loc["lon"], 1))
                coord_counts[key] += 1
                if key not in coord_info:
                    coord_info[key] = loc
            best = coord_counts.most_common(1)[0][0]
            info = coord_info[best]
            lat, lon = info["lat"], info["lon"]
            location_name = info.get("name")
            country_code = info.get("country_code")

        # Collect top themes across articles
        theme_counts: dict[str, int] = {}
        for art in cluster_articles_data.get(cid, [])[:20]:
            if art.get("themes") and isinstance(art["themes"], list):
                for t in art["themes"][:5]:
                    tn = t.get("theme", "")
                    if (
                        tn
                        and not tn.startswith("TAX_ETHNICITY")
                        and not tn.startswith("TAX_WORLDLANGUAGE")
                    ):
                        theme_counts[tn] = theme_counts.get(tn, 0) + 1
        top_themes = sorted(theme_counts, key=theme_counts.get, reverse=True)[:5]

        # Category from top theme
        category = _categorize_themes(top_themes)

        # Sample articles (latest 5)
        sample_articles = []
        for art in cluster_articles_data.get(cid, [])[:5]:
            sample_articles.append(
                {
                    "title": art["title"],
                    "url": art["url"],
                    "domain": art["domain"],
                }
            )

        results.append(
            {
                "id": cid,
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "recent_count": c.get("recent_count", 0),
                "velocity": round(c.get("velocity", 0), 3),
                "silent_hours": (round(c["silent_hours"], 1) if c.get("silent_hours") else None),
                "age_hours": round(c["age_hours"], 1) if c.get("age_hours") else None,
                "lat": lat,
                "lon": lon,
                "location_name": location_name,
                "country_code": country_code,
                "category": category,
                "top_themes": top_themes,
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
                "sample_articles": sample_articles,
            }
        )

    return results
