# Configuration

Last updated: 2026-09-17

Configuration is loaded through `pydantic-settings` from inherited environment variables and `settings/.env`. Inherited process variables take precedence, which allows isolated QA databases without changing the local file. The launcher never prints environment values.

This launch/runtime configuration remains outside the Angular Settings workspace. The Settings page does not expose or edit `.env`, inherited environment variables, database paths, ports, hosts, logging, job roots, timeouts, runtime-discovery paths, or equivalent environment-backed values. There is no speculative settings API or duplicate SQLite settings store for these fields.

Database settings are SQLite-only and use `DATABASE_SQLITE_PATH` and `DATABASE_CONNECT_TIMEOUT`.

Application settings use `NEURO_WEAVE_CORS_ORIGINS`, `NEURO_WEAVE_LOG_LEVEL`, `NEURO_WEAVE_LOG_FILE`, `NEURO_WEAVE_EXPORT_ROOT`, and `NEURO_WEAVE_RUNTIME_DISCOVERY_PATH`. Relative SQLite, log, export, ML-job, and runtime-discovery paths are anchored to the repository root. The default database is `app/resources/database.db`; canonical artifacts use `app/resources/runtime/exports`; ephemeral worker workspaces use `app/resources/runtime/ml-jobs`; the environment discovery snapshot uses `app/resources/runtime/environment-discovery.json`.

The process controller settings are `NEURO_WEAVE_ML_JOB_ROOT`, `NEURO_WEAVE_ML_WORKER_TIMEOUT_SECONDS`, `NEURO_WEAVE_ML_CANCEL_GRACE_SECONDS`, `NEURO_WEAVE_ML_TERMINATE_GRACE_SECONDS`, and `NEURO_WEAVE_ML_JOB_HISTORY_LIMIT`. Defaults are a bounded runtime root, 300 seconds, five seconds cooperative cancellation, five seconds termination escalation, and 100 terminal records. These values govern worker lifecycle only; they do not enable training or native inference.

Memory policies are explicit per job and are never translated between backends: TensorFlow supports `default`, `tensorflow_growth`, and `tensorflow_limit_mb`; JAX supports `default`, `jax_no_preallocation`, `jax_preallocate_fraction`, and the slow/debug `jax_platform_allocator`; PyTorch supports `default` and `pytorch_fraction`. Metadata exposes only policies discovered as supported for each live device.

`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, and `UI_PORT` configure the local servers. `RELOAD` controls backend reload mode and defaults to `false` for ordinary launches. `BACKEND_LOGS_VISIBLE` controls whether the launcher opens a dedicated terminal for backend logs and defaults to `true` when absent. The Angular development wrapper and Node preview server use the relative `/api` path; the development wrapper writes a local proxy descriptor for the configured backend and the preview server proxies `/health` and `/api` directly.

Runtime discovery snapshots are keyed by a SHA-256 fingerprint of the Python/platform identity, installed framework versions, selected runtime environment variables, hardware summary, and the persisted backend-installation record. Snapshot loading is display-only and never imports optional frameworks during startup. A matching snapshot is validated and reused when the requested capability is first needed; otherwise `POST /api/execution/discovery` probes only that backend/API pair in an isolated worker. Changing dependencies, hardware, runtime variables, or backend configuration causes the next requested check to refresh; the Environment page also exposes an explicit refresh action. `NEURO_WEAVE_RUNTIME_DISCOVERY_TIMEOUT_SECONDS` controls the isolated probe deadline and defaults to 60 seconds.

If `settings/.env` is absent, both the launcher and direct backend startup copy the exact tracked `settings/.env.example` template before loading settings. Existing local environment files are never overwritten. Inherited process variables take precedence over values loaded from the file.
