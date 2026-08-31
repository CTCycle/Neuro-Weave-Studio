# Startup

Last updated: 2026-08-31

On Windows run `start_on_windows.ps1` and select **Launch application**. On the first launch, after core dependencies are ready, the launcher offers a numbered backend choice covering Keras-only, PyTorch GPU/CUDA, PyTorch CPU-only, TensorFlow CPU-only, JAX CPU-only, and combined configurations. It detects NVIDIA CUDA with `nvidia-smi`, falls back to the CPU wheel when a PyTorch CUDA install fails, records the requested and resolved extras, and reports each backend's actual installed version and worker capability during explicit setup or management. A configured normal launch reuses that installation record; the separate **Manage compute backends** action can be revisited later and refuses to mutate the environment while the application is running.

The launcher validates local runtimes and dependencies. Option 2 installs or repairs dependencies and builds `app/tests/cache/frontend-dist`; option 3 rebuilds only the frontend from current source using the existing frontend dependencies; option 1 reuses the existing build and never rebuilds it automatically. If `app/tests/cache/frontend-dist/index.html` is missing, run the standalone `start_on_windows.ps1 -Action RebuildFrontend` action before launching. The launch action records timed stages for runtime setup, environment loading, dependency checks, backend setup, artifact/port checks, and service readiness; it starts Uvicorn without reload by default and the Node SPA preview server through independent process roots, then waits for `/health` and the frontend root. Once both services are ready it opens the frontend in the default browser when possible, prints the service PIDs, detaches the process roots, and closes the initial PowerShell launcher. Pass `-NoBrowser` to skip browser opening. If readiness fails, the launcher retains the error, cleans up only its tracked process trees, and does not close the interactive menu.

The **Initialize or upgrade database** action runs `app/scripts/initialize_database.py` without destructive flags. It creates a missing or empty SQLite file through Alembic and upgrades an existing database to the current head. The action is safe to repeat and refuses to run while the configured application ports are active.

FastAPI lifespan performs the same migration check before creating repository sessions, then initializes the node catalog and runtime controller and records timed startup stages. Optional runtime discovery is deliberately deferred: `/health` and `/api/metadata` read package and cached snapshot information without importing TensorFlow, PyTorch, JAX, Keras, GPU runtimes, or hardware utilities. `POST /api/execution/discovery` checks one requested backend/API pair in an isolated worker; concurrent requests for the same pair are joined, at most two probes run at once, and the result is exposed through the discovery endpoint and subsequent metadata reads. Matching `app/resources/runtime/environment-discovery.json` snapshots are validated on demand and reused safely; incompatible schema versions are discarded. When multiple instances start together, a database-local lock serializes the migration section; the later instance rechecks the head and continues without applying duplicate changes. A migration failure rolls back the shared SQLAlchemy transaction, disposes the engine, logs the failure, and prevents the application from serving requests against an incomplete schema.

Manual development startup is available when separate process control is required:

```powershell
cd app/server
$runtimeCacheRoot = (Resolve-Path ../..).Path + '\\runtimes\\cache'
$toolCacheRoot = (Resolve-Path ../..).Path + '\\app\\tests\\cache'
$env:UV_CACHE_DIR = $runtimeCacheRoot + '\\uv'
$env:PIP_CACHE_DIR = $runtimeCacheRoot + '\\pip'
$env:PYTHONPYCACHEPREFIX = $toolCacheRoot + '\\python-bytecode'
$env:COVERAGE_FILE = $toolCacheRoot + '\\coverage\\.coverage'
$env:PYTHONPATH = (Get-Location).Path
uv run uvicorn app:create_app --factory --host 127.0.0.1 --port 8000
```

Add `--reload` only for an intentional development reload session; the launcher and normal application startup keep reload disabled.

In another terminal, the Angular CLI wrapper starts the development server and its API proxy:

```powershell
cd app/client
npm run dev -- --host 127.0.0.1 --port 5173
```

Backend and frontend output from the launcher is written to `app/resources/runtime/logs` when backend logs are hidden. Set `BACKEND_LOGS_VISIBLE=false` in `settings/.env` to use that file-only mode; it defaults to `true`, which opens a dedicated terminal for backend logs while application log files continue to be written.

The launcher menu groups **Remove logs**, **Clear cache**, **Remove all data**, and **Uninstall application** under maintenance and data actions. Cache and data removal attempts one bulk operation per target and falls back to deterministic, deepest-first item removal only when a bulk operation encounters a locked or protected entry. Clear cache removes both current cache roots and known legacy locations; locked or admin-owned files are reported and skipped so the action continues. Remove all data shows a `[y/N]` confirmation; it removes the configured database, SQLite sidecars, runtime data, exports, job workspaces, logs, and backend state while preserving application files, settings, runtimes, dependencies, caches, and build output. Uninstall shows a `[y/N]` confirmation; it removes repository-local runtimes, the server virtual environment, frontend dependencies, both cache roots, and legacy cache locations while preserving settings, lockfiles, database files, and exports.

The **Check for updates** action performs a read-only `git fetch --dry-run` against `origin/main` and does not download or apply changes. **Update application** requires the application to be stopped and the checkout to be non-detached, clean, and already on `main`; it runs `git pull --ff-only origin main` without switching branches or modifying local changes.

The editor reads `/api/metadata` to discover recorded API capabilities, installed/operational backends, devices, normalized lease keys, and supported memory policies. The Environment page exposes separate API, backend, device, and policy controls. Keras is operational; native inference is visibly deferred. If a selection is used by an active job, the page offers **Stop and switch** (cancel, reap, then apply) or **Keep running** (preserve the immutable job and use the new selection for later jobs).

Inference and binary Export submit asynchronous jobs. The UI polls lifecycle/progress and diagnostics, exposes Stop while active, and retrieves an artifact only after reload/inference verification and revision-safe publication. Compilation source generation remains synchronous and its source viewer supports inspection/copy only. Navigation and component unmounting do not cancel server work. A fresh worker is used for each smoke or export operation, so ordinary selection changes do not require an application restart; installing/removing optional packages does.

Native Windows support is constrained by the official TensorFlow and JAX platform matrices. Unsupported GPU combinations are reported as unavailable rather than offered. The current machine may expose only PyTorch CPU; GPU and multi-GPU paths remain conditional until a supported runtime and device are installed.
