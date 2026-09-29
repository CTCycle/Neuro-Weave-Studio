# NW-T3-10 process and fixture cleanup

Date: 2026-09-29

- `/api/projects` returned an empty project list after the serial browser run.
- The agent-created Codex in-app Browser tab was closed.
- The task-owned backend root PID `27388` and its child process tree were
  terminated after the final checks.
- The task-owned frontend root PID `10688` and its child process tree were
  terminated after the final checks.
- `Get-NetTCPConnection` confirmed no listeners remained on ports `9750` or
  `6425`.

No unrelated process was targeted. The isolated runtime/cache directories are
retained under the validation QA root for the recorded evidence boundary.
