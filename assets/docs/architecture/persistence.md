# Persistence

Last updated: 2026-08-04

NeuroWeave uses one SQLAlchemy schema for SQLite and PostgreSQL. A project is the model/workflow aggregate currently exposed by `/api/projects`; application sessions, separate models, and separate workflows are not persisted because they are not product concepts in the current MVP.

The canonical tables are `schema_metadata`, `projects`, `project_graph_settings`, `project_nodes`, `project_edges`, `project_revisions`, and `generated_code_snapshots`. Projects own current graph settings, ordered nodes, and ordered edges. Composite project/node foreign keys prevent an edge from connecting nodes belonging to another project. Project revisions are immutable graph snapshots. Generated source is linked to a specific project revision and is returned only for the current revision.

Known queryable values use relational columns. JSON is limited to node parameters, dynamic graph configuration maps, and immutable revision snapshots. Enums are portable strings with checks, UUIDs are application-generated strings, and timestamps use a UTC-normalizing SQLAlchemy type so SQLite and PostgreSQL expose the same application contract.

Repositories never commit independently. A unit of work supplies one SQLAlchemy session to focused project, graph, revision, and code-snapshot repositories. Services own transaction boundaries; API routes only validate requests, call services, and map domain errors to HTTP responses. Optimistic revision checks prevent stale project updates from overwriting newer saves.

`repositories/initializer.py` is the single schema lifecycle source of truth. Missing SQLite files are initialized once. Existing SQLite and PostgreSQL databases receive one lightweight `schema_metadata` readiness query during ordinary startup. No request path recreates or repairs a schema.

This project intentionally has no migration or legacy compatibility layer. The explicit initializer accepts `--reset` to recreate the managed schema from scratch. The launcher confirms this destructive action before passing the flag. Existing databases using the previous two-table schema must be recreated before launch.
