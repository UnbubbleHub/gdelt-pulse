"""Attention Asymmetry endpoint — coverage by country vs crisis theme intensity."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/asymmetry")
def get_asymmetry():
    """Return attention asymmetry data: coverage by country vs crisis
    theme intensity, revealing overcovered and underreported regions.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Coverage volume per country
            cur.execute("""
                SELECT
                    loc->>'country_code' AS code,
                    COUNT(DISTINCT a.id) AS article_count,
                    round(avg((a.tone->>'tone')::float)::numeric, 3) AS avg_tone
                FROM articles a, jsonb_array_elements(a.locations) AS loc
                WHERE a.locations IS NOT NULL
                  AND loc->>'country_code' IS NOT NULL
                  AND a.tone IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
                """)
            countries = cur.fetchall()

            # Crisis-tagged articles per country (themes indicating conflict,
            # disaster, or humanitarian issues)
            cur.execute("""
                SELECT
                    loc->>'country_code' AS code,
                    COUNT(DISTINCT a.id) AS crisis_articles
                FROM articles a,
                    jsonb_array_elements(a.locations) AS loc,
                    jsonb_array_elements(a.themes) AS theme
                WHERE a.locations IS NOT NULL
                  AND a.themes IS NOT NULL
                  AND loc->>'country_code' IS NOT NULL
                  AND (
                    theme->>'theme' ILIKE '%%CONFLICT%%'
                    OR theme->>'theme' ILIKE '%%CRISIS%%'
                    OR theme->>'theme' ILIKE '%%DISASTER%%'
                    OR theme->>'theme' ILIKE '%%KILL%%'
                    OR theme->>'theme' ILIKE '%%TERROR%%'
                    OR theme->>'theme' ILIKE '%%ARMED%%'
                    OR theme->>'theme' ILIKE '%%WAR%%'
                    OR theme->>'theme' ILIKE '%%REFUGEE%%'
                    OR theme->>'theme' ILIKE '%%FAMINE%%'
                    OR theme->>'theme' ILIKE '%%WOUND%%'
                  )
                GROUP BY 1
                """)
            crisis_rows = cur.fetchall()

            # Top overcovered clusters (most articles, any country)
            cur.execute("""
                SELECT id, representative_title, article_count,
                       first_article_at, last_article_at
                FROM clusters
                WHERE is_active = true
                ORDER BY article_count DESC
                LIMIT 10
                """)
            overcovered = cur.fetchall()

            # True distinct article count (articles can mention multiple countries)
            cur.execute("""
                SELECT COUNT(DISTINCT a.id) AS cnt
                FROM articles a, jsonb_array_elements(a.locations) AS loc
                WHERE a.locations IS NOT NULL
                  AND loc->>'country_code' IS NOT NULL
                  AND a.tone IS NOT NULL
                """)
            true_total_articles = cur.fetchone()["cnt"]

            # Underreported: clusters where MOST articles are crisis-tagged
            # but total coverage is low
            cur.execute("""
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       crisis.crisis_count,
                       round(crisis.crisis_count::numeric / c.article_count, 2) AS crisis_pct
                FROM clusters c
                JOIN (
                    SELECT cm.cluster_id, COUNT(DISTINCT a.id) AS crisis_count
                    FROM cluster_memberships cm
                    JOIN articles a ON a.id = cm.article_id,
                        jsonb_array_elements(a.themes) AS theme
                    WHERE a.themes IS NOT NULL
                      AND (
                        theme->>'theme' ILIKE '%%CONFLICT%%'
                        OR theme->>'theme' ILIKE '%%KILL%%'
                        OR theme->>'theme' ILIKE '%%TERROR%%'
                        OR theme->>'theme' ILIKE '%%ARMED%%'
                        OR theme->>'theme' ILIKE '%%REFUGEE%%'
                        OR theme->>'theme' ILIKE '%%FAMINE%%'
                        OR theme->>'theme' ILIKE '%%WOUND%%'
                      )
                    GROUP BY cm.cluster_id
                    HAVING COUNT(DISTINCT a.id) >= 3
                ) crisis ON crisis.cluster_id = c.id
                WHERE c.is_active = true
                  AND c.article_count BETWEEN 5 AND 40
                  AND crisis.crisis_count::float / c.article_count >= 0.6
                ORDER BY crisis.crisis_count DESC, c.article_count ASC
                LIMIT 20
                """)
            underreported = cur.fetchall()

    # Build crisis map
    crisis_map = {r["code"]: r["crisis_articles"] for r in crisis_rows}

    # Use the true distinct count for the headline stat;
    # per-country sum is kept for coverage_pct (share of mentions)
    total_articles = true_total_articles
    total_mentions = sum(c["article_count"] for c in countries)

    country_data = []
    for c in countries:
        crisis_count = crisis_map.get(c["code"], 0)
        country_data.append(
            {
                "code": c["code"],
                "article_count": c["article_count"],
                "coverage_pct": (
                    round(c["article_count"] / total_mentions * 100, 3) if total_mentions else 0
                ),
                "crisis_articles": crisis_count,
                "crisis_ratio": (
                    round(crisis_count / c["article_count"], 3) if c["article_count"] else 0
                ),
                "avg_tone": float(c["avg_tone"]) if c["avg_tone"] is not None else 0,
            }
        )

    return {
        "total_articles": total_articles,
        "total_countries": len(countries),
        "countries": country_data,
        "overcovered": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in overcovered
        ],
        "underreported": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in underreported
        ],
    }
