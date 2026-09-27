# NW-T3-01 cleanup record

- The official backend and frontend processes for ports 9851 and 6427 were stopped after the final browser run.
- Post-stop probes reported both HTTP endpoints stopped; `netstat -ano` showed no `LISTENING` entry for either port, only normal `TIME_WAIT` connections.
- The two disposable browser projects were deleted by the test fixture cleanup path.
- Temporary task-owned Playwright output directories and isolated pytest/launcher cache roots were removed after evidence was copied to the stable `evidence/` paths.
- No unrelated repository cache, process, or workspace path was targeted.
