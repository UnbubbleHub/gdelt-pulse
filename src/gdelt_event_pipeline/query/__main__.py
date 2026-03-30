"""Allow running the query layer as a module: python -m gdelt_event_pipeline.query"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from gdelt_event_pipeline.config.settings import get_settings
from gdelt_event_pipeline.query.models import SearchFilters, SearchRequest
from gdelt_event_pipeline.query.search import hybrid_search
from gdelt_event_pipeline.storage.database import close_pool, init_pool


def main() -> int:
    parser = argparse.ArgumentParser(description="Search articles and clusters.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--limit", type=int, default=20, help="Max results (default: 20).")
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Weight for semantic vs keyword (0.0=keyword only, 1.0=semantic only, default: 0.5).",
    )
    parser.add_argument("--location", action="append", dest="locations", help="Filter by location.")
    parser.add_argument("--person", action="append", dest="persons", help="Filter by person.")
    parser.add_argument(
        "--organization", action="append", dest="organizations", help="Filter by organization."
    )
    parser.add_argument("--theme", action="append", dest="themes", help="Filter by theme.")
    parser.add_argument("--domain", action="append", dest="domains", help="Filter by domain.")
    parser.add_argument("--source", action="append", dest="sources", help="Filter by source.")
    parser.add_argument("--date-from", type=str, default=None, help="Start date (ISO format).")
    parser.add_argument("--date-to", type=str, default=None, help="End date (ISO format).")
    parser.add_argument("--clusters", action="store_true", help="Also search cluster centroids.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings = get_settings()
    init_pool(settings.db)

    try:
        filters = SearchFilters(
            locations=args.locations,
            persons=args.persons,
            organizations=args.organizations,
            themes=args.themes,
            domains=args.domains,
            sources=args.sources,
            date_from=datetime.fromisoformat(args.date_from) if args.date_from else None,
            date_to=datetime.fromisoformat(args.date_to) if args.date_to else None,
        )
        has_filters = any(getattr(filters, f) is not None for f in filters.__dataclass_fields__)

        request = SearchRequest(
            query=args.query,
            filters=filters if has_filters else None,
            limit=args.limit,
            semantic_weight=args.semantic_weight,
            search_clusters=args.clusters,
        )

        result = hybrid_search(request)

        print(f'\nSearch: "{result.query}"')
        print(f"Hits: {result.total_semantic_hits} semantic, {result.total_keyword_hits} keyword")
        print(f"Results: {len(result.articles)} articles")

        for i, scored in enumerate(result.articles, 1):
            a = scored.article
            sem = f"sem={scored.semantic_rank}" if scored.semantic_rank else "sem=-"
            kw = f"kw={scored.keyword_rank}" if scored.keyword_rank else "kw=-"
            print(f"\n  {i}. [{scored.rrf_score:.6f}] ({sem}, {kw})")
            print(f"     {a.get('title', '(no title)')}")
            print(f"     {a.get('canonical_url', '')}")
            ts = a.get("gdelt_timestamp")
            if ts:
                if isinstance(ts, datetime):
                    print(f"     {ts.strftime('%Y-%m-%d %H:%M')} UTC")
                else:
                    print(f"     {ts}")

        if result.clusters:
            print(f"\nClusters: {len(result.clusters)}")
            for sc in result.clusters:
                c = sc.cluster
                sim = 1 - sc.cosine_distance
                title = c.get("representative_title", "(no title)")
                print(f"\n  {sc.rank}. [sim={sim:.4f}] {title}")
                print(f"     Articles: {c.get('article_count', 0)}")

    finally:
        close_pool()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
