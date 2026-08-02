# Configuration

Last updated: 2026-08-02

Configuration is loaded through `pydantic-settings` from environment variables and `settings/.env`. Supported variables are `NEURO_WEAVE_DATABASE_URL`, `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, and `NEURO_WEAVE_EXPORT_ROOT`. Vite reads `VITE_API_BASE_URL` from the same settings directory.

Relative SQLite and export paths are anchored to the repository root, not the process working directory. This keeps the application, migrations, launch scripts, and manual startup commands on one canonical runtime location. The default database is `app/resources/runtime/neuro_weave.db`; generated artifacts use `app/resources/runtime/exports`. The default CORS origins are `http://127.0.0.1:5173` and `http://localhost:5173`.
