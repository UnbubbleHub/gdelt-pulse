"""Narrative Polarization endpoints — tone divergence across sources within clusters."""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/api/polarization")
def get_polarization(
    limit: int = Query(30, ge=1, le=100, description="Max clusters to return"),
    min_articles: int = Query(20, ge=5, le=500, description="Minimum articles per cluster"),
):
    """Return the most narratively polarized story clusters.

    Polarization score = standard deviation of tone across articles
    within the same cluster.  High stddev means the same story is
    being framed very differently by different sources.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.representative_title,
                    c.article_count,
                    c.first_article_at,
                    c.last_article_at,
                    round(avg((a.tone->>'tone')::float)::numeric, 3)     AS avg_tone,
                    round(stddev((a.tone->>'tone')::float)::numeric, 3)  AS tone_stddev,
                    round(min((a.tone->>'tone')::float)::numeric, 3)     AS tone_min,
                    round(max((a.tone->>'tone')::float)::numeric, 3)     AS tone_max,
                    count(DISTINCT a.domain)                              AS distinct_sources
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id
                WHERE c.is_active = true
                  AND c.article_count >= %s
                  AND a.tone IS NOT NULL
                GROUP BY c.id
                HAVING count(*) >= %s AND stddev((a.tone->>'tone')::float) IS NOT NULL
                ORDER BY tone_stddev DESC
                LIMIT %s
                """,
                (min_articles, min_articles, limit),
            )
            clusters = cur.fetchall()

            if not clusters:
                return []

            cluster_ids = [c["id"] for c in clusters]

            # Per-source tone breakdown for each cluster
            cur.execute(
                """
                SELECT
                    cm.cluster_id,
                    COALESCE(a.canonical_source, a.domain) AS source,
                    round(avg((a.tone->>'tone')::float)::numeric, 3) AS avg_tone,
                    count(*) AS article_count
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.tone IS NOT NULL
                GROUP BY cm.cluster_id, COALESCE(a.canonical_source, a.domain)
                HAVING count(*) >= 2
                ORDER BY cm.cluster_id, avg((a.tone->>'tone')::float) ASC
                """,
                (cluster_ids,),
            )
            source_rows = cur.fetchall()

            # Tone histogram buckets per cluster
            cur.execute(
                """
                SELECT
                    cm.cluster_id,
                    width_bucket((a.tone->>'tone')::float, -10, 10, 20) AS bucket,
                    count(*) AS cnt
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = ANY(%s)
                  AND a.tone IS NOT NULL
                GROUP BY cm.cluster_id, bucket
                ORDER BY cm.cluster_id, bucket
                """,
                (cluster_ids,),
            )
            hist_rows = cur.fetchall()

    # Build per-cluster source breakdown
    sources_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        cid = str(row["cluster_id"])
        sources_by_cluster[cid].append(
            {
                "source": row["source"],
                "avg_tone": float(row["avg_tone"]),
                "article_count": row["article_count"],
            }
        )

    # Build per-cluster histogram
    hist_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in hist_rows:
        cid = str(row["cluster_id"])
        bucket_idx = row["bucket"]
        # width_bucket(val, -10, 10, 20) → buckets 1..20, each 1.0 wide
        low = -10 + (bucket_idx - 1) * 1.0
        hist_by_cluster[cid].append(
            {
                "range_low": low,
                "range_high": low + 1.0,
                "count": row["cnt"],
            }
        )

    results = []
    for c in clusters:
        cid = str(c["id"])
        sources = sources_by_cluster.get(cid, [])
        histogram = hist_by_cluster.get(cid, [])

        results.append(
            {
                "id": cid,
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
                "polarization_score": float(c["tone_stddev"]),
                "avg_tone": float(c["avg_tone"]),
                "tone_min": float(c["tone_min"]),
                "tone_max": float(c["tone_max"]),
                "distinct_sources": c["distinct_sources"],
                "sources": sources[:15],  # top 15 sources
                "histogram": histogram,
            }
        )

    return results


@router.get("/api/polarization/{cluster_id}")
def get_polarization_detail(cluster_id: str):
    """Detailed polarization breakdown for a single cluster.

    Returns every article with its tone, grouped by source.
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, representative_title, article_count, "
                "first_article_at, last_article_at FROM clusters WHERE id = %s",
                (cluster_id,),
            )
            cluster = cur.fetchone()
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            cur.execute(
                """
                SELECT
                    a.id,
                    a.title,
                    a.url,
                    a.canonical_url,
                    a.domain,
                    COALESCE(a.canonical_source, a.domain) AS source,
                    a.gdelt_timestamp,
                    (a.tone->>'tone')::float AS tone,
                    (a.tone->>'polarity')::float AS polarity,
                    (a.tone->>'positive_score')::float AS positive_score,
                    (a.tone->>'negative_score')::float AS negative_score
                FROM cluster_memberships cm
                JOIN articles a ON a.id = cm.article_id
                WHERE cm.cluster_id = %s
                  AND a.tone IS NOT NULL
                ORDER BY (a.tone->>'tone')::float ASC
                """,
                (cluster_id,),
            )
            articles = cur.fetchall()

    # Group by source
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        source_groups[a["source"]].append(
            {
                "title": a["title"],
                "url": a["canonical_url"] or a["url"],
                "domain": a["domain"],
                "gdelt_timestamp": a["gdelt_timestamp"],
                "tone": round(a["tone"], 3) if a["tone"] is not None else None,
                "polarity": (round(a["polarity"], 3) if a["polarity"] is not None else None),
                "positive_score": (
                    round(a["positive_score"], 3) if a["positive_score"] is not None else None
                ),
                "negative_score": (
                    round(a["negative_score"], 3) if a["negative_score"] is not None else None
                ),
            }
        )

    # Sort sources by average tone
    source_summaries = []
    for source, arts in source_groups.items():
        tones = [a["tone"] for a in arts if a["tone"] is not None]
        avg_t = sum(tones) / len(tones) if tones else 0
        source_summaries.append(
            {
                "source": source,
                "avg_tone": round(avg_t, 3),
                "article_count": len(arts),
                "articles": arts,
            }
        )
    source_summaries.sort(key=lambda x: x["avg_tone"])

    # Overall stats
    all_tones = [a["tone"] for a in articles if a["tone"] is not None]
    avg = sum(all_tones) / len(all_tones) if all_tones else 0
    variance = sum((t - avg) ** 2 for t in all_tones) / len(all_tones) if all_tones else 0
    stddev = variance**0.5

    return {
        "cluster": {
            "id": str(cluster["id"]),
            "title": cluster["representative_title"],
            "article_count": cluster["article_count"],
            "first_article_at": cluster["first_article_at"],
            "last_article_at": cluster["last_article_at"],
        },
        "polarization_score": round(stddev, 3),
        "avg_tone": round(avg, 3),
        "tone_min": round(min(all_tones), 3) if all_tones else 0,
        "tone_max": round(max(all_tones), 3) if all_tones else 0,
        "total_articles_with_tone": len(all_tones),
        "sources": source_summaries,
    }
