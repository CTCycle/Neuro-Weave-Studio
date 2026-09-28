# NW-T3-04 process and fixture cleanup

Date: 2026-09-28
Run: `NW-CV-20260928-T3-04-ADVANCED-SPATIAL-CONVOLUTION`

The task-owned official-launcher process trees were stopped after the browser
rerun and live readiness capture:

- Backend: `uvicorn app:create_app --factory --host 127.0.0.1 --port 9854`
  process tree rooted at PIDs `33580`, `34280`, and `36920`.
- Frontend: `npm run preview -- --host 127.0.0.1 --port 6431 --strictPort`
  process tree rooted at PIDs `33336`, `37332`, `26772`, and `30320`.

The task-created disposable data root
`runtimes/cache/nw-t3-04-advanced-convolution-20260928` was removed after
the final run. A post-cleanup listener check reported no listeners on ports
`9854` or `6431`.

Result: **PASS**. No task-owned server, watcher, or disposable validation
fixture remains running or on disk.
