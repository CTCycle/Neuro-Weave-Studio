# NW-T3-02 process cleanup

Cleanup completed: 2026-09-28T07:14:24.2081736Z UTC

- Closed the task-owned Codex in-app Browser tab.
- Stopped the official launcher processes that owned the validation ports:
  backend PID `18708` on port `9852` and frontend PID `40544` on port `6428`.
- Verified that ports `9852` and `6428` have no listening sockets after
  shutdown.
- Checked for remaining processes using the workspace runtime or the isolated
  `nw-t3-02-recurrent-20260928` run root; none remained beyond the inspection
  shell itself.

The isolated cache/job/export/database roots remain on disk as ignored,
recoverable validation artifacts and were not removed.
