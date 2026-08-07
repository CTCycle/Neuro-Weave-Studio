# Configuration

Last updated: 2026-08-06

Configuration is loaded through `pydantic-settings` from inherited environment variables and `settings/.env`. Inherited process variables take precedence, which allows isolated QA databases without changing the local file. The launcher never prints environment values.

Database settings use `DATABASE_EMBEDDED`, `DATABASE_SQLITE_PATH`, `DATABASE_URL`, `DATABASE_ENGINE`, `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_SSL`, `DATABASE_SSL_CA`, and `DATABASE_CONNECT_TIMEOUT`.

Application settings use `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, `NEURO_WEAVE_LOG_FILE`, and `NEURO_WEAVE_EXPORT_ROOT`. Relative SQLite, log, and export paths are anchored to the repository root. The default database is `app/resources/runtime/neuro_weave.db`; generated artifacts use `app/resources/runtime/exports`.

`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT`, and `RELOAD` configure the local servers. `ALWAYS_REBUILD` defaults to `true`; when set to `false`, the launcher reuses `app/client/dist/index.html` when it already exists. `VITE_API_BASE_URL` defaults to `/api`; Vite reads these values from `settings/.env` and proxies the API base and `/health` to the backend in both development and preview modes.
