# NW-T3-09 cleanup record

Date: 2026-09-29

- Final `GET /api/projects` returned an empty project list after browser
  finalizers deleted every fixture.
- The Codex in-app Browser validation tab was closed.
- Task-owned backend/frontend listeners were stopped.
- Ports `9750` and `6425` were checked after shutdown and had zero listeners.
- Structural evidence and launcher runtime logs remain under the dedicated
  `assets/QA/comprehensive-validation-20260929/` roots.
