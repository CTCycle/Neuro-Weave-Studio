# Persistence

Last updated: 2026-08-02

## Database contract

NeuroWeave uses SQLAlchemy with one immutable `RepositoryContext` per process.
`EMBEDDED_DATABASE=true` selects SQLite; `false` selects PostgreSQL. The exact
operator contract is documented in [runtime configuration](../runtime/configuration.md).

SQLite defaults to `app/resources/runtime/neuro_weave.db`. Its connections use
foreign keys, a 30-second busy timeout, and WAL journaling. PostgreSQL uses the
`psycopg` driver, escaped connection credentials, optional SSL, pre-ping, and
bounded pool recycling.

## Schema and ownership

The database contract is adapted to NeuroWeave's existing ORM models; it does
not import a schema from another application. The table names and payload shape
remain stable so the current API and frontend continue to work.

- `projects` stores the current validated graph, metadata, framework target,
  validation status, and generated source snapshots.
- `project_revisions` stores immutable graph snapshots for every saved revision
  and cascades when its project is deleted.
- JSON is persisted as validated canonical graph text, never as framework
  objects.

Focused repositories own SQL queries and transactions. Route handlers and
services receive repositories, not raw database sessions. Reads are
side-effect-free; writes commit or roll back as one unit.

## Initialization policy

Normal startup initializes only a missing embedded SQLite file. Existing SQLite
files and PostgreSQL databases are schema-checked but never silently rebuilt.
`app/scripts/initialize_database.py --drop-existing` and
`start_on_windows.bat -Mode InitializeDatabase` are the explicit destructive
reset paths. SQLite resets delete the configured file and its WAL/SHM sidecars;
PostgreSQL resets drop and recreate only NeuroWeave-owned tables.

There is no migration runtime or legacy URL compatibility path. Schema changes are
handled as intentional destructive rebuilds until a future migration strategy is
designed and approved.

## Portable project data

Portable bundles include a format identifier, bundle version, export timestamp,
and `ProjectCreate` payload. They are independent of the SQL database and can be
imported into a fresh project store. Optimistic revision checks prevent silent
overwrite of a newer save.
