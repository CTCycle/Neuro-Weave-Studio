# Troubleshooting

Last updated: 2026-08-20

A blank catalog usually means node JSON failed Pydantic validation; inspect the backend log under `app/resources/runtime/logs`. If a configured port is occupied, the launcher reports the listener PID and leaves that process untouched. Save conflicts return HTTP 409 and require reopening before overwrite.

## Execution jobs

`POST /api/execution/jobs` returns structured submission errors for an invalid policy, unavailable API/backend/device, an occupied device lease, a missing project, or an incompatible graph. Accepted work is always observable through `GET /api/execution/jobs/{job_id}`; a worker crash, timeout, cancellation, or revision conflict is recorded on that job and does not make the application unusable for later submissions.

The controller requests cooperative cancellation first. After five seconds it terminates the worker tree and after another five seconds force-terminates it. A job is not reported `terminated` until its PID has exited and been reaped. If the application stops, it applies the same sequence to every active lease before repository disposal. An operating-system restart is reserved for a driver/runtime fault that survives worker and application restart.

`artifact_not_ready` means an export has not succeeded; `revision_conflict` means the project changed while the worker ran and the previous canonical artifact was preserved. Ephemeral job directories and bounded logs are removed on terminal states. The canonical artifact is outside that workspace and is replaced atomically only after reload/inference verification.

`dependency_missing`, `backend_import_failed`, and unsupported device/policy messages come from fresh probe workers. Native inference reports `not_implemented` by design. Memory numbers are best-effort observations: PyTorch allocated and reserved/cache memory are different, TensorFlow reports tracked current/peak values, and JAX reports device statistics only when the platform exposes them. Cleanup calls do not replace process teardown.

If a backend package is installed or removed, stop the application and restart it so probes and startup environment are refreshed. Normal backend/device/policy selection changes only affect new workers.

## Database migrations

The backend logs the current Alembic revision, pending upgrades, legacy version-2 adoption, and the verified head. If startup reports an unsupported or incomplete schema, do not delete the database: make a backup/export and restore the matching migration files or follow the project’s data-recovery procedure. Version-1 and the historical two-table prototype are intentionally not auto-adopted. A lock-timeout message means another process is migrating; close stale application instances and retry after confirming no migration process is still active.
