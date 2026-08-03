# Startup

Last updated: 2026-08-03

On Windows run `start_on_windows.ps1` and select **Launch application**. The launcher validates local runtimes and dependencies, checks both configured ports, starts Uvicorn and Vite, waits for `/health` and the frontend root, and keeps the child processes attached to the launcher lifecycle. Press Ctrl+C to stop both services.

The **Initialize PostgreSQL** action is the only normal PostgreSQL creation/schema workflow. It is safe to rerun and refuses to run while `DATABASE_EMBEDDED=true`. Ordinary startup assumes PostgreSQL is already initialized.

Manual development startup is available when separate process control is required:

```powershell
cd app/server
$env:PYTHONPATH = (Get-Location).Path
uv run uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Backend and frontend output from the launcher is written to `app/resources/runtime/logs`.
