# NW-T3-08 cleanup record

Date: 2026-09-29

- Browser finalizers deleted all temporary positive and negative projects.
- Final `GET /api/projects` returned an empty project list.
- The temporary Codex in-app Browser tab was closed.
- Task-owned backend and frontend processes were stopped by their verified
  listening PIDs.
- Ports 9750 and 6425 were rechecked and had no listeners.
- The isolated export basetemp was empty, but removing its directory was
  denied by the managed environment ACL after the test completed. It was not
  force-changed through ownership or permission escalation.
- The protected default `app/server/.pytest_cache` was not modified.
