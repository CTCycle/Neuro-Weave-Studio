# Persistence

Last updated: 2026-09-03

NeuroWeave uses one SQLAlchemy schema backed exclusively by SQLite. Alembic is the authoritative schema lifecycle and `repositories.base.Base.metadata` is its autogeneration source of truth. A project is the model/workflow aggregate exposed by `/api/projects`; installation-wide reusable definitions are persisted separately from project graphs.

The canonical project tables are `projects`, `project_graph_settings`, `project_nodes`, `project_edges`, `project_revisions`, and `generated_code_snapshots`, plus Alembic’s `alembic_version` table. Projects own current graph settings, ordered nodes, and ordered edges. Composite project/node foreign keys prevent an edge from connecting nodes belonging to another project. Project revisions are immutable graph snapshots. Generated source is linked to a specific project revision and is returned only for the current revision.

Installation-wide authoring definitions use focused tables outside the project aggregate. `custom_metric_definitions` stores persisted user metric/loss equations. `custom_node_definitions` stores user-authored node-library metadata with relational identity, kind, normalized display name, category, timestamps, and a JSON definition payload. Built-in node manifests are checked-in application resources and are never copied into this table. Custom node definitions therefore cannot overwrite, edit, or delete built-in manifests.

`custom_node_definitions.spec_json` is intentionally bounded authoring metadata. Mathematical definitions store a LaTeX formula and input-symbol contract; composite definitions store an ordered list of built-in manifest identities and normalized parameter values; repository definitions store public Hugging Face coordinates and optional class/entrypoint metadata. The persistence layer does not store executable Python or JavaScript, trusted remote-code flags, downloaded model code, or arbitrary callables.

The model catalog uses the existing `/api/projects` contract and enriches each `ProjectSummary` with batched node/edge counts and `has_generated_source`. The service obtains graph summaries and current-revision source-snapshot state through repository queries inside the existing unit of work, so catalog cards do not issue per-model detail requests. A source snapshot is associated with the model's current revision and persisted execution profile; saving a new revision therefore invalidates the previous compiled state without a schema change.

Known queryable values use relational columns. JSON is limited to node parameters, dynamic graph configuration maps, immutable revision snapshots, expression trees, and bounded custom-definition payloads. Enums are strings with checks, UUIDs or stable identifiers are application-generated strings, and timestamps use a UTC-normalizing SQLite type.

Repositories never commit independently. A unit of work supplies one SQLAlchemy session to focused project, graph, revision, code-snapshot, metric, and custom-node repositories. Services own transaction boundaries; API routes only validate requests, call services, and map domain errors to HTTP responses. Optimistic revision checks prevent stale project updates from overwriting newer saves.

`repositories/initializer.py` coordinates the Alembic environment. Missing or empty SQLite files are upgraded from base to head. Every application startup checks the current Alembic heads and applies pending revisions before repository sessions are created. A database-local lock serializes concurrent migration attempts, and one SQLAlchemy transaction covers migration execution. ORM registration includes both the canonical project models and installation-wide custom-node models before the migrated table set is compared with `Base.metadata`.

Databases that contain tables but do not carry the current Alembic history are unsupported. They are rejected without stamping, adoption, translation, or mutation; there is no legacy schema reader or compatibility path. The application exposes no destructive reset path.
