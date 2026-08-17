# Persistence

Last updated: 2026-08-17

NeuroWeave uses one SQLAlchemy schema backed exclusively by SQLite. A project is the model/workflow aggregate currently exposed by `/api/projects`; application sessions, separate models, and separate workflows are not persisted because they are not product concepts in the current MVP.

The canonical tables are `schema_metadata`, `projects`, `project_graph_settings`, `project_nodes`, `project_edges`, `project_revisions`, and `generated_code_snapshots`. Projects own current graph settings, ordered nodes, and ordered edges. Composite project/node foreign keys prevent an edge from connecting nodes belonging to another project. Project revisions are immutable graph snapshots. Generated source is linked to a specific project revision and is returned only for the current revision.

The catalog uses the existing `/api/projects` contract and enriches each `ProjectSummary` with batched node/edge counts and `compiled_targets`. The service obtains graph summaries and current-revision source-snapshot targets through repository queries inside the existing unit of work, so catalog cards do not issue per-model detail requests. A target is compiled only when a generated source snapshot exists for the model's current revision and configured code target; saving a new revision therefore invalidates the previous compiled state without a schema change.

Known queryable values use relational columns. JSON is limited to node parameters, dynamic graph configuration maps, and immutable revision snapshots. Enums are strings with checks, UUIDs are application-generated strings, and timestamps use a UTC-normalizing SQLite type.

Repositories never commit independently. A unit of work supplies one SQLAlchemy session to focused project, graph, revision, and code-snapshot repositories. Services own transaction boundaries; API routes only validate requests, call services, and map domain errors to HTTP responses. Optimistic revision checks prevent stale project updates from overwriting newer saves.

`repositories/initializer.py` is the single schema lifecycle source of truth. Missing SQLite files are initialized once. Existing SQLite files receive one lightweight `schema_metadata` readiness query during ordinary startup. No request path recreates or repairs a schema.

This project intentionally has no migration or legacy compatibility layer. The explicit initializer accepts `--reset` to recreate the managed schema from scratch. The launcher confirms this destructive action before passing the flag. Existing databases using the previous two-table schema must be recreated before launch.
