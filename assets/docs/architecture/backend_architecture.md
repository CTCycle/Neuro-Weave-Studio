# Backend Architecture

Last updated: 2026-08-02

The flat `app/server` layout exposes `app.py` as the FastAPI entry point, with `api`, `common`, `configurations`, `domain`, `repositories`, and `services` as the primary layers. NeuroWeave-specific `catalog`, `compilation`, `exporters`, and `validation` packages remain direct server packages so the graph behavior stays unchanged.

FastAPI routers expose health, metadata, node catalog, projects, validation, compilation, and export APIs. Pydantic models define contracts. `common.catalogs` loads immutable manifests. `validation` enforces graph and shape rules. `compilation` produces framework-independent IR. `exporters` translate IR and graph data. SQLAlchemy repositories persist projects and revisions. Route handlers coordinate typed services and contain no graph logic.
