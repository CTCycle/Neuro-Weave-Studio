# Startup

Last updated: 2026-10-22

On Windows run `start_on_windows.ps1` and select **Launch application**. On the first launch, after core dependencies are ready, the launcher offers a numbered backend choice covering Keras-only, PyTorch GPU/CUDA, PyTorch CPU-only, TensorFlow CPU-only, JAX CPU-only, and combined configurations. It detects NVIDIA CUDA with `nvidia-smi`, falls back to the CPU wheel when a PyTorch CUDA install fails, records the requested and resolved extras, and reports each backend's actual installed version and worker capability during explicit setup or management. A configured normal launch reuses that installation record; the separate **Manage compute backends** action can be revisited later and refuses to mutate the environment while the application is running.

The launcher validates local runtimes and dependencies. After environment loading, launch checks the configured backend port with one listener snapshot. If a port is occupied, it reports the port, PID, and available process name, then asks once before terminating the unique approved PIDs. A declined confirmation, redirected/non-interactive console, launcher-PID invariant violation, failed termination, or port that remains occupied after the bounded recheck aborts launch without starting services. The application and maintenance actions use the same batched listener query when they require the application to be stopped.

Option 2 installs or repairs dependencies and builds `runtimes/cache/frontend-dist`; option 3 rebuilds only the frontend from current source using the existing frontend dependencies. Successful `npm ci` writes `runtimes/cache/frontend-dependency-state.json`, keyed by `package.json`, `package-lock.json`, and the pinned Node version. Normal launch requires that state before considering frontend dependencies ready. It then checks `runtimes/cache/frontend-build-state.json`, which contains a content fingerprint of production Angular inputs and the pinned Node version. A missing or unreadable build state, missing `index.html`, changed production input, or changed Node version triggers one automatic frontend build; a matching state reuses the existing build. Test/spec source, E2E configuration, QA files, documentation, and preview/development tooling are excluded from the production fingerprint. A failed build never publishes new build state. Explicit install and `-Action RebuildFrontend` remain available for deliberate rebuilds.

The launch action records timed stages for runtime setup, environment loading, port checks, dependency checks, backend setup, frontend artifact/build freshness, and service readiness. It starts one process, Uvicorn without reload by default, because the backend serves the built frontend from `runtimes/cache/frontend-dist` on the same origin as the API (`/assets`, `index.html`, client-route fallback, `/api`, `/health`, `/runtime-config.json`). The launcher waits for `/health`, verifies that `/` returns the Angular shell, opens `http://<FASTAPI_HOST>:<FASTAPI_PORT>/` in the default browser (skipped with `-NoBrowser`), and then stays running as the lifecycle owner. Press Enter or Ctrl+C in the launcher console to stop. Ctrl+C reaches Uvicorn through the shared console, so the launcher waits up to 15 seconds for its normal lifespan shutdown; Enter, an unexpected backend exit, or a launcher error end in a forced stop of only the process tree the launcher started. If the backend exits on its own the launcher reports its exit code and recent log output and exits non-zero. If readiness fails, the launcher cleans up its tracked process tree and returns to the menu. The Node preview server no longer exists; the Angular development server (below) is for development only.

The **Initialize or upgrade database** action runs `app/scripts/initialize_database.py` without destructive flags. It creates a missing or empty SQLite file through Alembic and upgrades an existing database to the current head. The action is safe to repeat and refuses to run while the configured application ports are active.

FastAPI lifespan performs the same migration check before creating repository sessions, then initializes the node catalog and runtime controller and records timed startup stages. Optional runtime discovery is deliberately deferred: `/health` and `/api/metadata` read package and cached snapshot information without importing TensorFlow, PyTorch, JAX, Keras, GPU runtimes, or hardware utilities. `POST /api/execution/discovery` checks one requested backend/API pair in an isolated worker; concurrent requests for the same pair are joined, at most two probes run at once, and the result is exposed through the discovery endpoint and subsequent metadata reads. Matching `runtimes/cache/environment-discovery.json` snapshots are validated on demand and reused safely; incompatible schema versions are discarded. When multiple instances start together, a database-local lock serializes the migration section; the later instance rechecks the head and continues without applying duplicate changes. A migration failure rolls back the shared SQLAlchemy transaction, disposes the engine, logs the failure, and prevents the application from serving requests against an incomplete schema.

Manual development startup is available when separate process control is required:

```powershell
cd app/server
$cacheRoot = (Resolve-Path ../..).Path + '\\runtimes\\cache'
$env:UV_CACHE_DIR = $cacheRoot + '\\uv'
$env:PIP_CACHE_DIR = $cacheRoot + '\\pip'
$env:npm_config_cache = $cacheRoot + '\\npm'
$env:PYTHONPYCACHEPREFIX = $cacheRoot + '\\python-bytecode'
$env:PYTHON_EGG_CACHE = $cacheRoot + '\\python-eggs'
$env:COVERAGE_FILE = $cacheRoot + '\\coverage\\.coverage'
$env:RUFF_CACHE_DIR = $cacheRoot + '\\ruff'
$env:PLAYWRIGHT_BROWSERS_PATH = $cacheRoot + '\\playwright\\browsers'
$env:PYTHONPATH = (Get-Location).Path
uv run uvicorn app:create_app --factory --host 127.0.0.1 --port 8000
```

Add `--reload` only for an intentional development reload session; the launcher and normal application startup keep reload disabled.

In another terminal, the Angular CLI wrapper starts the development server (`UI_HOST`/`UI_PORT`) and its API proxy; this is the only workflow that uses the frontend port:

```powershell
cd app/client
npm run dev -- --host 127.0.0.1 --port 5173
```

Backend output appears in the launcher console by default. Set `BACKEND_LOGS_VISIBLE=false` in `settings/.env` to run the backend hidden with output captured in `data/runtime/logs/backend.out.log` and `backend.err.log`; it defaults to `true`. Application log files continue to be written in both modes. The data directory defaults to `data` and can be changed with `NEURO_WEAVE_RESOURCES_DIR` or the launcher's `-ResourcesDir` parameter.

The launcher menu groups **Remove logs**, **Clear cache**, **Remove all data**, and **Uninstall application** under maintenance and data actions. Cache and data removal enumerates each target and removes individual files and directories in a deterministic, deepest-first order. Locked or protected items are reported and skipped so the action continues. Clear cache removes the canonical runtime and test cache roots; locked or admin-owned files are reported and skipped so the action continues. Remove all data shows a `[y/N]` confirmation; it removes the configured database, SQLite sidecars, runtime data, exports, job workspaces, logs, and backend state while preserving application files, settings, runtimes, dependencies, caches, and build output. Uninstall shows a `[y/N]` confirmation; it removes repository-local runtimes, the server virtual environment, frontend dependencies, and both canonical cache roots while preserving settings, lockfiles, database files, and exports.

The **Check for updates** action performs a read-only `git fetch --dry-run` against `origin/main` and does not download or apply changes. **Update application** requires the application to be stopped and the checkout to be non-detached, clean, and already on `main`; it runs `git pull --ff-only origin main` without switching branches or modifying local changes.

The editor reads `/api/metadata` to discover recorded API capabilities, installed/operational backends, devices, normalized lease keys, and supported memory policies. The Environment page exposes separate API, backend, device, and policy controls. Real prediction serving is outside the current scope; functional validation is profile-gated and runs only against the selected backend/API. If a selection is used by an active job, the page offers **Stop and switch** (cancel, reap, then apply) or **Keep running** (preserve the immutable job and use the new selection for later jobs).

Functional Validation and binary Export submit asynchronous jobs. The UI polls lifecycle/progress and diagnostics, exposes Stop while active, and retrieves an artifact only after reload/forward-pass verification and revision-safe publication. Compilation source generation remains synchronous and its source viewer supports inspection/copy only. Navigation and component unmounting do not cancel server work. A fresh worker is used for each validation or export operation, so ordinary selection changes do not require an application restart; installing/removing optional packages does.

Native Windows support is constrained by the official TensorFlow and JAX platform matrices. Unsupported GPU combinations are reported as unavailable rather than offered. The current machine may expose only PyTorch CPU; GPU and multi-GPU paths remain conditional until a supported runtime and device are installed.
