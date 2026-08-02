# Startup

Last updated: 2026-08-02

On Windows run `start_on_windows.bat -Mode Launch` for a build-backed FastAPI plus Vite preview. `-Mode Install` provisions pinned project-local Python, uv, and Node.js runtimes; `Dev`, `Backend`, `Frontend`, `Build`, `Test`, and `Stop` are also available. Add `-NoBrowser` for automation. On Linux or macOS run `scripts/start.sh`. Manual startup uses `uv run uvicorn neuro_weave.app:app --host 127.0.0.1 --port 8000` and `npm run dev -- --host 127.0.0.1 --port 5173`. The Vite development and preview servers proxy `/api` without rewriting the prefix.
