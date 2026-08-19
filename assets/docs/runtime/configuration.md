# Configuration

Last updated: 2026-08-20

Configuration is loaded through `pydantic-settings` from inherited environment variables and `settings/.env`. Inherited process variables take precedence, which allows isolated QA databases without changing the local file. The launcher never prints environment values.

Database settings are SQLite-only and use `DATABASE_SQLITE_PATH` and `DATABASE_CONNECT_TIMEOUT`.

Application settings use `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, `NEURO_WEAVE_LOG_FILE`, and `NEURO_WEAVE_EXPORT_ROOT`. Relative SQLite, log, export, and ML-job paths are anchored to the repository root. The default database is `app/resources/database.db`; canonical artifacts use `app/resources/runtime/exports`; ephemeral worker workspaces use `app/resources/runtime/ml-jobs`.

The process controller settings are `NEURO_WEAVE_ML_JOB_ROOT`, `NEURO_WEAVE_ML_WORKER_TIMEOUT_SECONDS`, `NEURO_WEAVE_ML_CANCEL_GRACE_SECONDS`, `NEURO_WEAVE_ML_TERMINATE_GRACE_SECONDS`, and `NEURO_WEAVE_ML_JOB_HISTORY_LIMIT`. Defaults are a bounded runtime root, 300 seconds, five seconds cooperative cancellation, five seconds termination escalation, and 100 terminal records. These values govern worker lifecycle only; they do not enable training or native inference.

Memory policies are explicit per job and are never translated between backends: TensorFlow supports `default`, `tensorflow_growth`, and `tensorflow_limit_mb`; JAX supports `default`, `jax_no_preallocation`, `jax_preallocate_fraction`, and the slow/debug `jax_platform_allocator`; PyTorch supports `default` and `pytorch_fraction`. Metadata exposes only policies discovered as supported for each live device.

`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, and `UI_PORT` configure the local servers. `RELOAD` controls backend reload mode. `BACKEND_LOGS_VISIBLE` controls whether the launcher opens a dedicated terminal for backend logs and defaults to `true` when absent. `VITE_API_BASE_URL` defaults to `/api`; Vite reads these values from `settings/.env` and proxies the API base and `/health` to the backend in both development and preview modes.

If `settings/.env` is absent, both the launcher and direct backend startup copy the exact tracked `settings/.env.example` template before loading settings. Existing local environment files are never overwritten. Inherited process variables take precedence over values loaded from the file.
