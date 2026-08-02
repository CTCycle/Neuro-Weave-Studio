# Testing Strategy

Last updated: 2026-08-02

Backend tests cover manifest loading, DAG validation, shape propagation,
parameter counts, serialization, database settings, SQLite initialization,
transactional project persistence, API save/reopen behavior, JSON import/export,
and both code generators. PostgreSQL contract tests run when
`TEST_DATABASE_URL` is configured; an absent PostgreSQL service is reported as
skipped rather than passed. Optional Keras binary export is skipped when its
large dependencies are absent.

Frontend tests cover graph-cycle logic and catalog filtering. The production
frontend build is a required gate. Browser-level canvas, save/reopen, and export
coverage remains a documented next step because the current test suite is not a
full Playwright certification.

Run the standard checks from the repository root through the launcher or run
`scripts/verify.sh`. Keep temporary databases, logs, screenshots, and reports
under `assets/QA/`; never point a test at the developer’s default database.
