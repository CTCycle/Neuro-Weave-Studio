# Persistence

Last updated: 2026-08-02

SQLite is the initial database. `repositories` owns the SQLAlchemy base, models, session lifecycle, and project repository. `projects` stores the current document and metadata. `project_revisions` stores immutable graph snapshots for each save. JSON is stored as validated canonical graph text rather than framework objects. Portable bundles include a format identifier, bundle version, export timestamp, and `ProjectCreate` payload. Optimistic revision checks prevent silent overwrite of a newer save. The explicit database initializer is `app/scripts/initialize_database.py`; the API lifespan keeps the same create-on-start contract.
