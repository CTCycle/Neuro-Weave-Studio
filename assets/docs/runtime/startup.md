# Startup

Last updated: 2026-07-30

On Windows run `start_on_windows.ps1 -Mode Dev`. Backend-only, frontend-only, test, and build modes are available. On Linux or macOS run `scripts/start.sh`. Manual startup uses `uv run uvicorn neuro_weave.app:app --reload --port 8000` and `npm run dev`. The Vite development server proxies API paths to port 8000.
