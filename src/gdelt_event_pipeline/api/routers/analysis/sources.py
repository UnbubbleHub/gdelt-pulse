"""Source DNA endpoints — per-source fingerprints and detailed breakdowns."""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from gdelt_event_pipeline.api.routers.analysis._helpers import _categorize_themes

router = APIRouter()


@router.get("/api/sources/fingerprints")
def source_fingerprints(
    limit: int = Query(50, ge=1, le=200, description="Max sources to return"),
    sort: str = Query("articles", description="Sort: articles, tone_positive, tone_negative"),
    min_articles: int = Query(10, ge=1, description="Minimum articles to include source"),
):
    """Per-source fingerprints: article count, avg tone, top themes, top countries."""
    from gdelt_event_pipeline.storage.database import get_pool

    order_clause = {
        "tone_positive": "avg_tone DESC",
        "tone_negative": "avg_tone ASC",
    }.get(sort, "article_count DESC")

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT
                    COALESCE(source_common_name, domain) AS source_name,
                    domain,
                    count(*) AS article_count,
                    avg((tone->>'tone')::float) FILTER (WHERE tone->>'tone' IS NOT NULL)
                        AS avg_tone,
                    avg((tone->>'positive_score')::float)
                        FILTER (WHERE tone->>'positive_score' IS NOT NULL) AS avg_positive,
                    avg((tone->>'negative_score')::float)
                        FILTER (WHERE tone->>'negative_score' IS NOT NULL) AS avg_negative,
                    avg((tone->>'polarity')::float)
                        FILTER (WHERE tone->>'polarity' IS NOT NULL) AS avg_polarity,
                    min(gdelt_timestamp) AS first_article,
                    max(gdelt_timestamp) AS last_article
                FROM articles
                WHERE title IS NOT NULL
                GROUP BY COALESCE(source_common_name, domain), domain
                HAVING count(*) >= %s
                ORDER BY {order_clause}
                LIMIT %s
                """,
                (min_articles, limit),
            )
            sources = cur.fetchall()

            if not sources:
                return []

            # Get top themes per source (top 8)
            source_domains = [s["domain"] for s in sources]
            cur.execute(
                """
                SELECT domain,
                       t->>'theme' AS theme,
                       count(*) AS cnt
                FROM articles,
                     jsonb_array_elements(themes) AS t
                WHERE domain = ANY(%s)
                  AND t->>'theme' IS NOT NULL
                  AND t->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                  AND t->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                  AND t->>'theme' NOT LIKE 'TAX_FNCACT%%'
                GROUP BY domain, t->>'theme'
                ORDER BY domain, count(*) DESC
                """,
                (source_domains,),
            )
            theme_rows = cur.fetchall()

            # Get top countries per source (top 5)
            cur.execute(
                """
                SELECT domain,
                       loc->>'country_code' AS country_code,
                       count(*) AS cnt
                FROM articles,
                     jsonb_array_elements(locations) AS loc
                WHERE domain = ANY(%s)
                  AND loc->>'country_code' IS NOT NULL
                GROUP BY domain, loc->>'country_code'
                ORDER BY domain, count(*) DESC
                """,
                (source_domains,),
            )
            country_rows = cur.fetchall()

    # Build theme map (top 8 per domain)
    theme_map: dict[str, list[dict]] = defaultdict(list)
    for row in theme_rows:
        d = row["domain"]
        if len(theme_map[d]) < 8:
            category = _categorize_themes([row["theme"]])
            theme_map[d].append({"theme": row["theme"], "count": row["cnt"], "category": category})

    # Build country map (top 5 per domain)
    country_map: dict[str, list[dict]] = defaultdict(list)
    for row in country_rows:
        d = row["domain"]
        if len(country_map[d]) < 5:
            country_map[d].append({"country_code": row["country_code"], "count": row["cnt"]})

    results = []
    for s in sources:
        results.append(
            {
                "source_name": s["source_name"],
                "domain": s["domain"],
                "article_count": s["article_count"],
                "avg_tone": round(s["avg_tone"], 3) if s["avg_tone"] else 0,
                "avg_positive": round(s["avg_positive"], 3) if s["avg_positive"] else 0,
                "avg_negative": round(s["avg_negative"], 3) if s["avg_negative"] else 0,
                "avg_polarity": round(s["avg_polarity"], 3) if s["avg_polarity"] else 0,
                "first_article": s["first_article"],
                "last_article": s["last_article"],
                "top_themes": theme_map.get(s["domain"], []),
                "top_countries": country_map.get(s["domain"], []),
            }
        )

    return results


@router.get("/api/sources/{domain}/detail")
def source_detail(domain: str):
    """Detailed fingerprint for a single source domain."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Basic stats
            cur.execute(
                """
                SELECT
                    COALESCE(source_common_name, domain) AS source_name,
                    domain,
                    count(*) AS article_count,
                    avg((tone->>'tone')::float) FILTER (WHERE tone->>'tone' IS NOT NULL)
                        AS avg_tone,
                    avg((tone->>'positive_score')::float)
                        FILTER (WHERE tone->>'positive_score' IS NOT NULL) AS avg_positive,
                    avg((tone->>'negative_score')::float)
                        FILTER (WHERE tone->>'negative_score' IS NOT NULL) AS avg_negative,
                    avg((tone->>'polarity')::float)
                        FILTER (WHERE tone->>'polarity' IS NOT NULL) AS avg_polarity,
                    min(gdelt_timestamp) AS first_article,
                    max(gdelt_timestamp) AS last_article
                FROM articles
                WHERE domain = %s AND title IS NOT NULL
                GROUP BY COALESCE(source_common_name, domain), domain
                """,
                (domain,),
            )
            stats = cur.fetchone()
            if not stats:
                raise HTTPException(status_code=404, detail="Source not found")

            # Theme distribution (all themes)
            cur.execute(
                """
                SELECT t->>'theme' AS theme, count(*) AS cnt
                FROM articles, jsonb_array_elements(themes) AS t
                WHERE domain = %s
                  AND t->>'theme' IS NOT NULL
                  AND t->>'theme' NOT LIKE 'TAX_ETHNICITY%%'
                  AND t->>'theme' NOT LIKE 'TAX_WORLDLANGUAGE%%'
                  AND t->>'theme' NOT LIKE 'TAX_FNCACT%%'
                GROUP BY t->>'theme'
                ORDER BY count(*) DESC
                LIMIT 20
                """,
                (domain,),
            )
            themes = cur.fetchall()

            # Country distribution
            cur.execute(
                """
                SELECT loc->>'country_code' AS country_code, count(*) AS cnt
                FROM articles, jsonb_array_elements(locations) AS loc
                WHERE domain = %s AND loc->>'country_code' IS NOT NULL
                GROUP BY loc->>'country_code'
                ORDER BY count(*) DESC
                LIMIT 15
                """,
                (domain,),
            )
            countries = cur.fetchall()

            # Category distribution
            cur.execute(
                """
                SELECT t->>'theme' AS theme
                FROM articles, jsonb_array_elements(themes) AS t
                WHERE domain = %s AND t->>'theme' IS NOT NULL
                """,
                (domain,),
            )
            all_theme_rows = cur.fetchall()

            # Tone over time (by day)
            cur.execute(
                """
                SELECT date_trunc('day', gdelt_timestamp) AS day,
                       avg((tone->>'tone')::float) AS avg_tone,
                       count(*) AS article_count
                FROM articles
                WHERE domain = %s
                  AND tone->>'tone' IS NOT NULL
                  AND gdelt_timestamp IS NOT NULL
                GROUP BY date_trunc('day', gdelt_timestamp)
                ORDER BY day
                """,
                (domain,),
            )
            tone_timeline = cur.fetchall()

            # Recent articles (last 10)
            cur.execute(
                """
                SELECT title, url, gdelt_timestamp,
                       (tone->>'tone')::float AS tone_score
                FROM articles
                WHERE domain = %s AND title IS NOT NULL
                ORDER BY gdelt_timestamp DESC
                LIMIT 10
                """,
                (domain,),
            )
            recent = cur.fetchall()

    # Build category breakdown
    cat_counts: dict[str, int] = defaultdict(int)
    for row in all_theme_rows:
        cat = _categorize_themes([row["theme"]])
        cat_counts[cat] += 1
    total_themes = sum(cat_counts.values()) or 1
    categories = [
        {"category": cat, "count": cnt, "pct": round(cnt / total_themes * 100, 1)}
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "source_name": stats["source_name"],
        "domain": stats["domain"],
        "article_count": stats["article_count"],
        "avg_tone": round(stats["avg_tone"], 3) if stats["avg_tone"] else 0,
        "avg_positive": round(stats["avg_positive"], 3) if stats["avg_positive"] else 0,
        "avg_negative": round(stats["avg_negative"], 3) if stats["avg_negative"] else 0,
        "avg_polarity": round(stats["avg_polarity"], 3) if stats["avg_polarity"] else 0,
        "first_article": stats["first_article"],
        "last_article": stats["last_article"],
        "themes": [
            {
                "theme": t["theme"],
                "count": t["cnt"],
                "category": _categorize_themes([t["theme"]]),
            }
            for t in themes
        ],
        "countries": [{"country_code": c["country_code"], "count": c["cnt"]} for c in countries],
        "categories": categories,
        "tone_timeline": [
            {
                "day": t["day"].isoformat() if t["day"] else None,
                "avg_tone": round(t["avg_tone"], 3),
                "article_count": t["article_count"],
            }
            for t in tone_timeline
        ],
        "recent_articles": [dict(r) for r in recent],
    }
