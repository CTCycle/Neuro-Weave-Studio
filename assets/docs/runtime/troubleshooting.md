# Troubleshooting

Last updated: 2026-08-03

A blank catalog usually means node JSON failed Pydantic validation; inspect the backend log under `app/resources/runtime/logs`. If a configured port is occupied, the launcher reports the listener PID and leaves that process untouched. Save conflicts return HTTP 409 and require reopening before overwrite. Keras export returning 501 means optional dependencies are absent or a node is outside the binary subset. A 422 compilation response means validation or compatibility failed.

If an existing SQLite file fails the startup readiness query, repair it deliberately or restore a known-good database; normal startup never rebuilds an existing file. For PostgreSQL, set `DATABASE_EMBEDDED=false`, verify the connection fields, and select **Initialize PostgreSQL** before launching the application. The launcher reports sanitized connection errors without printing passwords or URLs.
