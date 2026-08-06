# Startup

Last updated: 2026-08-05

On Windows run `start_on_windows.ps1` and select **Launch application**. The launcher validates local runtimes and dependencies, installs the locked client dependencies, builds `app/client/dist`, checks both configured ports, starts Uvicorn and the Vite preview server from `app/client`, waits for `/health` and the frontend root, and keeps the child processes attached to the launcher lifecycle. Press Ctrl+C to stop both services.

The **Initialize or recreate database** action is the only explicit schema workflow. Without confirmation it initializes a missing SQLite file or checks an existing canonical schema. Typing `RESET` passes `--reset` to `app/scripts/initialize_database.py`, drops the managed local schema, and recreates it from scratch. PostgreSQL database creation and schema reset are also explicit; ordinary startup only checks the canonical `schema_metadata` marker.

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

Backend and frontend output from the launcher is written to `app/resources/runtime/logs`.

The editor reads `/api/metadata` to discover operational execution backends. With zero backends binary export is disabled with an installation hint; exactly one backend is selected automatically; with multiple backends the user must choose TensorFlow, PyTorch, or JAX. The choice is per export request and is not persisted in project data. Backend logs include sanitized requested/public id, worker-reported Keras backend, duration, and outcome.
