"""Generate an HTML page to visually inspect clusters and their articles."""

from __future__ import annotations

import html
import json
import webbrowser
from pathlib import Path

from psycopg.rows import dict_row

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.storage.database import close_pool, get_pool, init_pool

OUTPUT = Path(__file__).parent.parent / "cluster_report.html"


def _esc(text: str | None) -> str:
    return html.escape(str(text)) if text else ""


def _entity_pills(items: list | None, css_class: str) -> str:
    if not items:
        return ""
    pills = []
    for item in items[:8]:
        name = item["name"] if isinstance(item, dict) else str(item)
        pills.append(f'<span class="pill {css_class}">{_esc(name)}</span>')
    extra = f'<span class="pill muted">+{len(items) - 8}</span>' if len(items) > 8 else ""
    return " ".join(pills) + extra


def fetch_data() -> list[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT id, representative_title, article_count, is_active,
                       first_article_at, last_article_at
                FROM clusters
                ORDER BY article_count DESC
            """)
            clusters = cur.fetchall()

            for cluster in clusters:
                cur.execute("""
                    SELECT a.title, a.canonical_url, a.domain, a.canonical_source,
                           a.gdelt_timestamp, a.locations, a.persons, a.organizations,
                           a.tone, cm.similarity_score, cm.assignment_method
                    FROM articles a
                    JOIN cluster_memberships cm ON cm.article_id = a.id
                    WHERE cm.cluster_id = %s
                    ORDER BY cm.similarity_score DESC NULLS LAST
                """, (cluster["id"],))
                cluster["articles"] = cur.fetchall()
    return clusters


def build_html(clusters: list[dict]) -> str:
    total_articles = sum(c["article_count"] for c in clusters)
    singletons = sum(1 for c in clusters if c["article_count"] == 1)
    multi = len(clusters) - singletons

    rows = []
    for i, c in enumerate(clusters):
        title = c["representative_title"] or "(no title)"
        count = c["article_count"]
        is_open = "open" if count >= 5 else ""
        badge_class = "large" if count > 10 else "medium" if count >= 5 else "small" if count >= 2 else "singleton"

        article_rows = []
        for a in c["articles"]:
            t = a.get("tone") or {}
            tone_val = t.get("tone", "")
            tone_class = "pos" if isinstance(tone_val, (int, float)) and tone_val > 0 else "neg" if isinstance(tone_val, (int, float)) and tone_val < 0 else ""
            tone_str = f"{tone_val:.1f}" if isinstance(tone_val, (int, float)) else ""

            locs = a.get("locations") or []
            pers = a.get("persons") or []
            orgs = a.get("organizations") or []

            sim = a.get("similarity_score")
            sim_str = f"{sim:.3f}" if sim is not None else ""
            method = a.get("assignment_method") or ""

            article_rows.append(f"""
                <tr>
                    <td class="article-title">
                        <a href="{_esc(a.get('canonical_url', ''))}" target="_blank">
                            {_esc(a.get('title')) or '<em>no title</em>'}
                        </a>
                    </td>
                    <td class="source">{_esc(a.get('canonical_source') or a.get('domain'))}</td>
                    <td class="tone {tone_class}">{tone_str}</td>
                    <td class="sim">{sim_str}</td>
                    <td class="method">{_esc(method)}</td>
                    <td class="entities">
                        {_entity_pills(locs, 'loc')}
                        {_entity_pills(pers, 'per')}
                        {_entity_pills(orgs, 'org')}
                    </td>
                </tr>
            """)

        rows.append(f"""
        <details class="cluster" {is_open}>
            <summary>
                <span class="badge {badge_class}">{count}</span>
                <span class="cluster-title">{_esc(title)}</span>
                <span class="cluster-meta">
                    {_esc(str(c.get('first_article_at', ''))[:16])} &mdash;
                    {_esc(str(c.get('last_article_at', ''))[:16])}
                </span>
            </summary>
            <table class="articles">
                <thead>
                    <tr>
                        <th>Title</th><th>Source</th><th>Tone</th>
                        <th>Sim</th><th>Method</th><th>Entities</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(article_rows)}
                </tbody>
            </table>
        </details>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GDELT Pulse - Cluster Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
           background: #0d1117; color: #c9d1d9; padding: 24px; line-height: 1.5; }}
    h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.6em; }}
    .stats {{ color: #8b949e; margin-bottom: 24px; font-size: 0.95em; }}
    .stats span {{ color: #58a6ff; font-weight: 600; }}
    .filter-bar {{ margin-bottom: 20px; display: flex; gap: 12px; align-items: center; }}
    .filter-bar input {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
                         padding: 8px 12px; border-radius: 6px; width: 300px; font-size: 0.9em; }}
    .filter-bar select {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
                          padding: 8px; border-radius: 6px; font-size: 0.9em; }}
    .cluster {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                margin-bottom: 8px; }}
    .cluster summary {{ padding: 12px 16px; cursor: pointer; display: flex;
                        align-items: center; gap: 12px; font-size: 0.95em; }}
    .cluster summary:hover {{ background: #1c2129; }}
    .cluster[open] {{ border-color: #58a6ff; }}
    .badge {{ display: inline-flex; align-items: center; justify-content: center;
              min-width: 32px; height: 24px; border-radius: 12px; font-size: 0.8em;
              font-weight: 700; padding: 0 8px; }}
    .badge.large {{ background: #1f6feb; color: #fff; }}
    .badge.medium {{ background: #238636; color: #fff; }}
    .badge.small {{ background: #30363d; color: #8b949e; }}
    .badge.singleton {{ background: #21262d; color: #484f58; }}
    .cluster-title {{ flex: 1; font-weight: 500; }}
    .cluster-meta {{ color: #484f58; font-size: 0.8em; white-space: nowrap; }}
    .articles {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
    .articles th {{ text-align: left; padding: 8px 12px; color: #8b949e;
                    border-bottom: 1px solid #30363d; font-weight: 500; }}
    .articles td {{ padding: 6px 12px; border-bottom: 1px solid #21262d;
                    vertical-align: top; }}
    .articles tr:hover td {{ background: #1c2129; }}
    .article-title a {{ color: #58a6ff; text-decoration: none; }}
    .article-title a:hover {{ text-decoration: underline; }}
    .source {{ color: #8b949e; white-space: nowrap; }}
    .tone {{ font-family: monospace; white-space: nowrap; }}
    .tone.pos {{ color: #3fb950; }}
    .tone.neg {{ color: #f85149; }}
    .sim {{ font-family: monospace; color: #8b949e; }}
    .method {{ color: #484f58; font-size: 0.8em; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
             font-size: 0.75em; margin: 1px 2px; }}
    .pill.loc {{ background: #0d419d; color: #79c0ff; }}
    .pill.per {{ background: #3d1f00; color: #f0883e; }}
    .pill.org {{ background: #1b3a1b; color: #56d364; }}
    .pill.muted {{ background: #21262d; color: #484f58; }}
    .entities {{ max-width: 400px; }}
</style>
</head>
<body>
<h1>GDELT Pulse &mdash; Cluster Report</h1>
<div class="stats">
    <span>{len(clusters)}</span> clusters &middot;
    <span>{total_articles}</span> articles &middot;
    <span>{multi}</span> multi-article clusters &middot;
    <span>{singletons}</span> singletons
</div>
<div class="filter-bar">
    <input type="text" id="search" placeholder="Filter clusters by title..." oninput="filterClusters()">
    <select id="minSize" onchange="filterClusters()">
        <option value="1">All clusters</option>
        <option value="2" selected>2+ articles</option>
        <option value="5">5+ articles</option>
        <option value="10">10+ articles</option>
    </select>
</div>
<div id="clusters">
    {''.join(rows)}
</div>
<script>
function filterClusters() {{
    const q = document.getElementById('search').value.toLowerCase();
    const min = parseInt(document.getElementById('minSize').value);
    document.querySelectorAll('.cluster').forEach(el => {{
        const title = el.querySelector('.cluster-title').textContent.toLowerCase();
        const count = parseInt(el.querySelector('.badge').textContent);
        el.style.display = (title.includes(q) && count >= min) ? '' : 'none';
    }});
}}
filterClusters();
</script>
</body>
</html>"""


def main():
    settings = get_settings()
    init_pool(settings.db)
    try:
        clusters = fetch_data()
        page = build_html(clusters)
        OUTPUT.write_text(page, encoding="utf-8")
        print(f"Report written to {OUTPUT}")
        webbrowser.open(OUTPUT.as_uri())
    finally:
        close_pool()


if __name__ == "__main__":
    main()
