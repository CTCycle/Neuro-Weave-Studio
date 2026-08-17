# Configuration

Last updated: 2026-08-17

Configuration is loaded through `pydantic-settings` from inherited environment variables and `settings/.env`. Inherited process variables take precedence, which allows isolated QA databases without changing the local file. The launcher never prints environment values.

Database settings are SQLite-only and use `DATABASE_SQLITE_PATH` and `DATABASE_CONNECT_TIMEOUT`.

Application settings use `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, `NEURO_WEAVE_LOG_FILE`, and `NEURO_WEAVE_EXPORT_ROOT`. Relative SQLite, log, and export paths are anchored to the repository root. The default database is `app/resources/runtime/neuro_weave.db`; generated artifacts use `app/resources/runtime/exports`.

`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT`, and `RELOAD` configure the local servers. `ALWAYS_REBUILD` defaults to `true`; when set to `false`, the launcher reuses `app/client/dist/index.html` when it already exists. `VITE_API_BASE_URL` defaults to `/api`; Vite reads these values from `settings/.env` and proxies the API base and `/health` to the backend in both development and preview modes.

If `settings/.env` is absent, both the launcher and direct backend startup copy the exact tracked `settings/.env.example` template before loading settings. Existing local environment files are never overwritten. Inherited process variables take precedence over values loaded from the file.
