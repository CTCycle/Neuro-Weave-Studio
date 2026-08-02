# Persistence

Last updated: 2026-07-30

SQLite is the initial database. `projects` stores the current document and metadata. `project_revisions` stores immutable graph snapshots for each save. JSON is stored as validated canonical graph text rather than framework objects. Portable bundles include a format identifier, bundle version, export timestamp, and `ProjectCreate` payload. Optimistic revision checks prevent silent overwrite of a newer save.
