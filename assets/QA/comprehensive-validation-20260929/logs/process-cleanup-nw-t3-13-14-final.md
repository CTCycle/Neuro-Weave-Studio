# NW-T3-13/NW-T3-14 process cleanup

Date: 2026-09-29

- Before shutdown, the official launcher listeners were backend `127.0.0.1:9750`
  (PID 20248, `uvicorn app:create_app --factory`) and frontend
  `127.0.0.1:6425` (PID 33068, `node tools/preview-server.mjs`).
- The selected task-owned PIDs were stopped after the authoritative browser run.
- A post-stop listener check returned no listeners on ports 9750 or 6425.
- The exact isolated runtime root
  `assets/QA/comprehensive-validation-20260929/nw-t3-13-14-runtime` was removed
  after the services stopped. The first sandbox removal attempt was denied by
  the launcher-created ACL; an exact-path elevated cleanup then succeeded.
- The browser cleanup payload reported `projects_remaining_after_run: 0`.
