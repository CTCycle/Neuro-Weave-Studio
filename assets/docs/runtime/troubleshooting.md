# Troubleshooting

Last updated: 2026-10-22

A blank catalog usually means node JSON failed Pydantic validation; inspect the backend log under `data/runtime/logs`. If a configured port is occupied, the launcher reports the listener PID and leaves that process untouched. Save conflicts return HTTP 409 and require reopening before overwrite.

## Execution jobs

`POST /api/execution/jobs` returns structured submission errors for an invalid policy, unavailable API/backend/device, an occupied device lease, a missing project, or an incompatible graph. Accepted work is always observable through `GET /api/execution/jobs/{job_id}`; a worker crash, timeout, cancellation, or revision conflict is recorded on that job and does not make the application unusable for later submissions.

The controller requests cooperative cancellation first. After five seconds it terminates the worker tree and after another five seconds force-terminates it. A job is not reported `terminated` until its PID has exited and been reaped. If the application stops, it applies the same sequence to every active lease before repository disposal. An operating-system restart is reserved for a driver/runtime fault that survives worker and application restart.

`artifact_not_ready` means an export has not succeeded; `revision_conflict` means the project changed while the worker ran and the previous canonical artifact was preserved. Ephemeral job directories and bounded logs are removed on terminal states. The canonical artifact is outside that workspace and is replaced atomically only after reload/forward-pass verification.

`dependency_missing`, `backend_import_failed`, `probe_timeout`, and unsupported device/policy messages come from isolated runtime discovery. The Environment page starts a check only for the selected backend/API and shows `Not checked`, `Checking`, `Usable`, `Unavailable`, or `Failed`; an installed framework with an unchecked API is not reported as unavailable. Matching snapshots are display-only at process start and are validated when the requested capability is first used. After changing packages, hardware, runtime variables, or backend configuration, use the Environment refresh action or the explicit backend-management path to invalidate the snapshot. A stalled check has a bounded 60-second worker deadline and returns a visible diagnostic; it does not block `/health`, metadata, navigation, or source-only operations. Real native prediction reports `not_implemented` by design. Memory numbers are best-effort observations: PyTorch allocated and reserved/cache memory are different, TensorFlow reports tracked current/peak values, and JAX reports device statistics only when the platform exposes them. Cleanup calls do not replace process teardown.

If the launcher reports that the frontend is not served, run the explicit `RebuildFrontend` action: the backend serves `runtimes/cache/frontend-dist` and falls back to the API docs when no build exists. The launcher polls backend health with a bounded deadline, includes the exit code and recent log output when the backend exits, and stops its own process tree on Enter, Ctrl+C, or failure. Backend diagnostics are retained in `data/runtime/logs` and the launcher console.

If a backend package is installed or removed, stop the application and restart it so the process environment is refreshed, then run an explicit capability check. Normal backend/device/policy selection changes only affect new workers.

## Native compilation on Windows

Native PyTorch `torch.compile` uses the default Inductor backend. On Windows, Inductor needs a working Microsoft C++ toolchain and Windows SDK environment; having `cl.exe` installed is not sufficient if the application was started from a normal PowerShell process without the Visual Studio developer variables. Start the application from an x64 Native Tools Command Prompt, or import the matching environment before launching it, for example `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64`. Then restart the backend and rerun the native PyTorch capability check. Without that environment, the native compiler capability is reported unavailable and compile-enabled jobs are rejected with `native_compiler_unavailable`; eager materialization remains supported.

The model builder does not silently replace the configured native compiler with `aot_eager`: that would change the requested compilation profile. `aot_eager` can be used as a diagnostic or an explicitly selected backend when such a profile is exposed, but it is not treated as equivalent to the default Inductor path.

## Database migrations

The backend reports whether the database is already at the current Alembic head or is being initialized, then verifies the schema against current ORM metadata. If startup reports an unsupported schema, the initializer stops and does not mutate the file. Databases stamped with removed revisions and populated databases without an Alembic revision must be recreated as an explicit operator action; back up or export any data that must be retained. The application does not delete databases automatically. To point the application at a new database before startup, update the validated `database-config.json` under the application resource directory. A lock-timeout message means another process is initializing the database; close stale application instances and retry after confirming no initializer is still active.
