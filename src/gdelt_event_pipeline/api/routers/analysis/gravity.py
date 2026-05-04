"""Geopolitical Gravity Map endpoints — country co-mention graph."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/api/gravity/graph")
def gravity_graph(
    min_weight: int = Query(50, ge=10, description="Minimum co-mention count for edges"),
    limit_edges: int = Query(80, ge=10, le=300, description="Max edges to return"),
):
    """Return country co-mention graph for the geopolitical gravity map.

    Nodes = countries (with article counts and avg tone).
    Edges = co-mention relationships (with weight = co-mention count).
    """
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Max weight (for dynamic slider range)
            cur.execute("SELECT MAX(weight) AS mw FROM mv_country_comentions")
            max_weight_row = cur.fetchone()
            max_weight = max_weight_row["mw"] if max_weight_row and max_weight_row["mw"] else 0

            # Edges from materialized view
            cur.execute(
                """
                SELECT source_code AS source, target_code AS target, weight
                FROM mv_country_comentions
                WHERE weight >= %s
                ORDER BY weight DESC
                LIMIT %s
                """,
                (min_weight, limit_edges),
            )
            edges = cur.fetchall()

            # Collect all country codes that appear in edges
            codes = set()
            for e in edges:
                codes.add(e["source"])
                codes.add(e["target"])

            if not codes:
                return {"nodes": [], "edges": []}

            # Nodes from materialized view
            code_list = list(codes)
            cur.execute(
                """
                SELECT code, article_count, avg_tone
                FROM mv_country_stats
                WHERE code = ANY(%s)
                """,
                (code_list,),
            )
            node_rows = cur.fetchall()

    nodes = [
        {
            "id": n["code"],
            "article_count": n["article_count"],
            "avg_tone": float(n["avg_tone"]) if n["avg_tone"] is not None else 0,
        }
        for n in node_rows
    ]

    edge_list = [
        {
            "source": e["source"],
            "target": e["target"],
            "weight": e["weight"],
        }
        for e in edges
    ]

    return {"nodes": nodes, "edges": edge_list, "max_weight": max_weight}


@router.get("/api/gravity/country/{code}")
def gravity_country_detail(code: str):
    """Detail view for a single country: top connected countries,
    top clusters, and tone breakdown."""
    from gdelt_event_pipeline.storage.database import get_pool

    pool = get_pool()
    with pool.connection() as conn:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            # Country stats from materialized view
            cur.execute(
                "SELECT code, article_count, avg_tone FROM mv_country_stats WHERE code = %s",
                (code,),
            )
            stats = cur.fetchone()
            if not stats or stats["article_count"] == 0:
                raise HTTPException(status_code=404, detail="Country not found")

            # Top connected countries from materialized view
            cur.execute(
                """
                SELECT
                    CASE WHEN source_code = %s THEN target_code ELSE source_code END
                        AS connected_code,
                    weight
                FROM mv_country_comentions
                WHERE source_code = %s OR target_code = %s
                ORDER BY weight DESC
                LIMIT 15
                """,
                (code, code, code),
            )
            connections = cur.fetchall()

            # Top clusters mentioning this country
            cur.execute(
                """
                SELECT c.id, c.representative_title, c.article_count,
                       c.first_article_at, c.last_article_at,
                       COUNT(*) AS mentions
                FROM clusters c
                JOIN cluster_memberships cm ON cm.cluster_id = c.id
                JOIN articles a ON a.id = cm.article_id,
                    jsonb_array_elements(a.locations) AS loc
                WHERE a.locations IS NOT NULL
                  AND loc->>'country_code' = %s
                  AND c.is_active = true
                GROUP BY c.id
                ORDER BY mentions DESC
                LIMIT 10
                """,
                (code,),
            )
            top_clusters = cur.fetchall()

    return {
        "code": code,
        "article_count": stats["article_count"],
        "avg_tone": float(stats["avg_tone"]) if stats["avg_tone"] is not None else 0,
        "connections": [{"code": c["connected_code"], "weight": c["weight"]} for c in connections],
        "top_clusters": [
            {
                "id": str(c["id"]),
                "title": c["representative_title"],
                "article_count": c["article_count"],
                "mentions": c["mentions"],
                "first_article_at": c["first_article_at"],
                "last_article_at": c["last_article_at"],
            }
            for c in top_clusters
        ],
    }
