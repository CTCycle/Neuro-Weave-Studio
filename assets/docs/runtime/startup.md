# Startup

Last updated: 2026-08-02

On Windows run `start_on_windows.bat -Mode Launch` for a build-backed FastAPI
plus Vite preview. `-Mode Install` provisions pinned project-local runtimes;
`Dev`, `Backend`, `Frontend`, `Build`, `Test`, `Stop`, and
`InitializeDatabase` are also available. Add `-NoBrowser` for automation.

`InitializeDatabase` stops only launcher-owned processes and then performs the
explicit destructive application-schema reset described in
[persistence](../architecture/persistence.md). Do not use it as a normal launch
step.

On Linux or macOS run `scripts/start.sh`. Manual startup uses:

```powershell
cd app/server
uv run uvicorn neuro_weave.app:app --host 127.0.0.1 --port 8000
```

In another terminal, run `npm run dev -- --host 127.0.0.1 --port 5173` from
`app/client`. Vite proxies `/api` without rewriting the prefix.
