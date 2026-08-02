# Troubleshooting

Last updated: 2026-08-02

## Common failures

- A blank catalog usually means a node manifest failed Pydantic validation;
  inspect backend startup logs.
- Save conflicts return HTTP 409. Reopen the project before overwriting a newer
  revision.
- Keras binary export returning 501 means optional dependencies are absent or a
  node is outside the binary-export subset.
- A 422 compilation response means graph validation or framework compatibility
  failed.
- A startup error about missing tables means an existing SQLite file or the
  configured PostgreSQL database is stale. Run the explicit initializer after
  confirming that resetting application data is acceptable.
- A PostgreSQL connection error should be checked against `DATABASE_HOST`,
  `DATABASE_PORT`, credentials, SSL settings, and whether the target database
  exists. The initializer may create the target database only with appropriate
  PostgreSQL administrative privileges.

Never delete a database path by broad pattern. Use the exact configured path and
the documented `InitializeDatabase` mode when intentionally resetting local data.
