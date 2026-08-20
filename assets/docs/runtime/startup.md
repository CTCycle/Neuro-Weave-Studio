# Startup

Last updated: 2026-08-20

On Windows run `start_on_windows.ps1` and select **Launch application**. On the first launch, after core dependencies are ready, the launcher offers an explicit backend choice: Enter keeps the base Keras-only environment, `all` installs TensorFlow, PyTorch, and JAX, and a comma-separated selection installs only the requested frameworks. It detects NVIDIA CUDA with `nvidia-smi`, falls back to the CPU wheel when a PyTorch CUDA install fails, records the requested and resolved extras, and reports each backend's actual installed version and worker capability. The separate **Manage compute backends** action can be revisited later and refuses to mutate the environment while the application is running.

The launcher validates local runtimes and dependencies. Option 2 installs or repairs dependencies and builds `assets/cache/frontend-dist`; option 3 rebuilds only the frontend from current source using the existing frontend dependencies; option 1 reuses the existing build and performs a recovery build only when `assets/cache/frontend-dist/index.html` is missing. The standalone action is also available as `start_on_windows.ps1 -Action RebuildFrontend`. The launch action then checks both configured ports, starts Uvicorn and the Vite preview server through independent `cmd.exe` process roots, and waits for `/health` and the frontend root. Once both services are ready it opens the frontend in the default browser when possible, prints the service PIDs, detaches the process roots, and closes the initial PowerShell launcher. Pass `-NoBrowser` to skip browser opening. If readiness fails, the launcher retains the error, cleans up only its tracked process trees, and does not close the interactive menu.

The **Initialize or recreate database** action is the only explicit schema workflow. Without confirmation it initializes a missing SQLite file or checks an existing canonical schema. Typing `RESET` passes `--reset` to `app/scripts/initialize_database.py`, drops the managed local schema, and recreates it from scratch. Ordinary startup only checks the canonical `schema_metadata` marker for an existing SQLite file.

Manual development startup is available when separate process control is required:

```powershell
cd app/server
$cacheRoot = (Resolve-Path ../..).Path + '\\assets\\cache'
$env:PYTHONPYCACHEPREFIX = $cacheRoot + '\\python-bytecode'
$env:COVERAGE_FILE = $cacheRoot + '\\coverage\\.coverage'
$env:PYTHONPATH = (Get-Location).Path
uv run uvicorn app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd app/client
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Backend and frontend output from the launcher is written to `app/resources/runtime/logs` when backend logs are hidden. Set `BACKEND_LOGS_VISIBLE=false` in `settings/.env` to use that file-only mode; it defaults to `true`, which opens a dedicated terminal for backend logs while application log files continue to be written.

The launcher menu also provides **Remove logs**, **Clear cache**, and **Uninstall application** maintenance actions. Uninstall requires typing `UNINSTALL`; it removes repository-local runtimes, the server virtual environment, frontend dependencies, the centralized cache/artifact root, and legacy cache locations while preserving settings, lockfiles, database files, and exports.

The editor reads `/api/metadata` to discover live API capabilities, installed/operational backends, devices, normalized lease keys, and supported memory policies. The Environment page exposes separate API, backend, device, and policy controls. Keras is operational; native inference is visibly deferred. If a selection is used by an active job, the page offers **Stop and switch** (cancel, reap, then apply) or **Keep running** (preserve the immutable job and use the new selection for later jobs).

Inference and Compilation submit asynchronous jobs. The UI polls lifecycle/progress and diagnostics, exposes Stop while active, and retrieves a `.keras` artifact only after reload/inference verification and revision-safe publication. Navigation and component unmounting do not cancel server work. Source generation remains synchronous. A fresh worker is used for each smoke or export operation, so ordinary selection changes do not require an application restart; installing/removing optional packages does.

Native Windows support is constrained by the official TensorFlow and JAX platform matrices. Unsupported GPU combinations are reported as unavailable rather than offered. The current machine may expose only PyTorch CPU; GPU and multi-GPU paths remain conditional until a supported runtime and device are installed.
