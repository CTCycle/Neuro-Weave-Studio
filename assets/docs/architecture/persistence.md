# Persistence

Last updated: 2026-08-03

SQLite is the default database. `repositories` owns the SQLAlchemy base, models, session lifecycle, and project repository. `projects` stores the current document and metadata. `project_revisions` stores immutable graph snapshots for each save. JSON is stored as validated canonical graph text rather than framework objects. Portable bundles include a format identifier, bundle version, export timestamp, and `ProjectCreate` payload. Optimistic revision checks prevent silent overwrite of a newer save.

`repositories/initializer.py` is the single database lifecycle source of truth. During ordinary startup, a missing SQLite file is created once from ORM metadata; an existing SQLite file and every PostgreSQL database receive only one lightweight readiness query against `projects`. No request path recreates or fully inspects the schema.

PostgreSQL database creation and schema initialization are explicit launcher operations through `app/scripts/initialize_database.py`. The operation creates the configured database only when absent, creates missing ORM tables, is safe to rerun, and never drops existing data.
