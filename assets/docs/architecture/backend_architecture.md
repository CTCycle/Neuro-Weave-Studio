# Backend Architecture

Last updated: 2026-07-30

FastAPI routers expose health, metadata, node catalog, projects, validation, compilation, and export APIs. Pydantic models define contracts. `catalog` loads immutable manifests. `validation` enforces graph and shape rules. `compilation` produces framework-independent IR. `exporters` translate IR and graph data. SQLAlchemy repositories persist projects and revisions. Route handlers coordinate typed services and contain no graph logic.
