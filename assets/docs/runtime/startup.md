# Startup

Last updated: 2026-08-19

On Windows run `start_on_windows.ps1` and select **Launch application**. On the first launch, after core dependencies are ready, the launcher offers an explicit backend choice: Enter keeps the base Keras-only environment, `all` installs TensorFlow, PyTorch, and JAX, and a comma-separated selection installs only the requested frameworks. It detects NVIDIA CUDA with `nvidia-smi`, falls back to the CPU wheel when a PyTorch CUDA install fails, records the requested and resolved extras, and reports each backend's actual installed version and worker capability. The separate **Manage compute backends** action can be revisited later and refuses to mutate the environment while the application is running.

The launcher validates local runtimes and dependencies. Option 2 installs or repairs dependencies and builds `app/client/dist`; option 1 reuses the existing build and performs a recovery build only when `app/client/dist/index.html` is missing. It then checks both configured ports, starts Uvicorn and the Vite preview server through independent `cmd.exe` process roots, and waits for `/health` and the frontend root. Once both services are ready it opens the frontend in the default browser when possible, prints the service PIDs, detaches the process roots, and closes the initial PowerShell launcher. Pass `-NoBrowser` to skip browser opening. If readiness fails, the launcher retains the error, cleans up only its tracked process trees, and does not close the interactive menu.

The **Initialize or recreate database** action is the only explicit schema workflow. Without confirmation it initializes a missing SQLite file or checks an existing canonical schema. Typing `RESET` passes `--reset` to `app/scripts/initialize_database.py`, drops the managed local schema, and recreates it from scratch. Ordinary startup only checks the canonical `schema_metadata` marker for an existing SQLite file.

Manual development startup is available when separate process control is required:

```powershell
cd app/server
$env:PYTHONPATH = (Get-Location).Path
uv run uvicorn app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd app/client
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Backend and frontend output from the launcher is written to `app/resources/runtime/logs` when backend logs are hidden. Set `BACKEND_LOGS_VISIBLE=false` in `settings/.env` to use that file-only mode; it defaults to `true`, which opens a dedicated terminal for backend logs while application log files continue to be written.

The launcher menu also provides **Remove logs**, **Clear cache**, and **Uninstall application** maintenance actions. Uninstall requires typing `UNINSTALL`; it removes repository-local runtimes, the server virtual environment, frontend dependencies, and build output while preserving settings, lockfiles, database files, and exports.

The editor reads `/api/metadata` to discover installed and operational execution backends. The compute-environment panel distinguishes missing, installed-but-unavailable, and usable frameworks and points to the launcher management action. With zero backends binary export and the smoke test remain disabled with an explanation; with multiple operational backends the user chooses TensorFlow, PyTorch, or JAX. The **Smoke test** control validates the current graph, creates compatible zero-filled dummy tensors, and performs exactly one `training=False` forward pass without compiling, fitting, or updating weights. Results show input/output shapes, output dtype, parameter count, accelerator/device, and status. The choice is per request and is not persisted in project data. Backend logs include sanitized requested/public id, worker-reported Keras backend, duration, and outcome.
