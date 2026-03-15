"""Allow running the ingestion pipeline as a module: python -m gdelt_event_pipeline.ingestion"""
from gdelt_event_pipeline.ingestion.run import main

raise SystemExit(main())
