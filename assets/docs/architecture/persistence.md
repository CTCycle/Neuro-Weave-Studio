# Persistence

Last updated: 2026-08-21

NeuroWeave uses one SQLAlchemy schema backed exclusively by SQLite. Alembic is the authoritative schema lifecycle and `repositories.base.Base.metadata` is its autogeneration source of truth. A project is the model/workflow aggregate currently exposed by `/api/projects`; application sessions, separate models, and separate workflows are not persisted because they are not product concepts in the current MVP.

The canonical tables are `projects`, `project_graph_settings`, `project_nodes`, `project_edges`, `project_revisions`, and `generated_code_snapshots`, plus Alembic’s `alembic_version` table. Projects own current graph settings, ordered nodes, and ordered edges. Composite project/node foreign keys prevent an edge from connecting nodes belonging to another project. Project revisions are immutable graph snapshots. Generated source is linked to a specific project revision and is returned only for the current revision.

The catalog uses the existing `/api/projects` contract and enriches each `ProjectSummary` with batched node/edge counts and `has_generated_source`. The service obtains graph summaries and current-revision source-snapshot state through repository queries inside the existing unit of work, so catalog cards do not issue per-model detail requests. A source snapshot is associated with the model's current revision and persisted execution profile; saving a new revision therefore invalidates the previous compiled state without a schema change.

Known queryable values use relational columns. JSON is limited to node parameters, dynamic graph configuration maps, and immutable revision snapshots. Enums are strings with checks, UUIDs are application-generated strings, and timestamps use a UTC-normalizing SQLite type.

Repositories never commit independently. A unit of work supplies one SQLAlchemy session to focused project, graph, revision, and code-snapshot repositories. Services own transaction boundaries; API routes only validate requests, call services, and map domain errors to HTTP responses. Optimistic revision checks prevent stale project updates from overwriting newer saves.

`repositories/initializer.py` coordinates the Alembic environment. Missing or empty SQLite files are upgraded from base to head. Every application startup checks the current Alembic heads and applies pending revisions before repository sessions are created. A database-local lock serializes concurrent migration attempts, and one SQLAlchemy transaction covers migration execution.

Databases that contain tables but do not carry the current Alembic history are unsupported. They are rejected without stamping, adoption, translation, or mutation; there is no legacy schema reader or compatibility path. The application exposes no destructive reset path.
