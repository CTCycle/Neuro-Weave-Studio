# Startup

Last updated: 2026-08-02

On Windows run `start_on_windows.ps1 -Mode Dev`. Backend-only, frontend-only, test, and build modes are available. On Linux or macOS run `scripts/start.sh`. Manual backend startup runs from `app/server` with `PYTHONPATH=. uv run uvicorn app:app --reload --port 8000`; the database-only initializer is `PYTHONPATH=. uv run python ../scripts/initialize_database.py`. Start the frontend separately with `npm run dev`. The Vite development server proxies API paths to port 8000.
