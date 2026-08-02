# Configuration

Last updated: 2026-07-30

Configuration is loaded through `pydantic-settings` from environment variables and the repository-root `.env` file. Supported variables are `NEURO_WEAVE_DATABASE_URL`, `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, and `NEURO_WEAVE_EXPORT_ROOT`.

Relative SQLite and export paths are anchored to the repository root, not the process working directory. This keeps the application, migrations, launch scripts, and manual startup commands on one canonical runtime location. The default database is `runtime/neuro_weave.db`; generated artifacts use `runtime/exports`.
