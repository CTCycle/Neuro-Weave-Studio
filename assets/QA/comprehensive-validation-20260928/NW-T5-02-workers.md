# NW-T5-02 disposable worker matrix

Date: 2026-09-28
Branch: `develop`
Workflow: unchanged `app/client/e2e/workflows.spec.ts`
Launcher: official `start_on_windows.ps1 -Action Launch -NoBrowser`
Backend/frontend ports: `9853` / `6429`

## Final status

`NW-T5-02` remains **PARTIAL** and `ISSUE-001` remains open. The matrix did
not reproduce a SQLite lock at workers 1, 2, or 4, but it was not clean at any
worker count. The workflow suite recorded 1, 2, and 3 failures respectively.
Workers 2 and 4 also recorded CPU job-admission `409` responses in the isolated
application logs while the active-job reload workflow timed out. This is a
broader worker/job/UI concurrency limitation and is sufficient to keep the
gate open even though the historical SQLite-lock signature was absent.

| Workers | Result | Duration | Failures | Actual SQLite-lock signatures | Projects after suite | Ports after launcher stop |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1 | 19/20 passed | 100,559 ms | `NW-T4-02` Configure dialog missing while runtime control was still `Checking runtime…` | 0 | 0 | 0 |
| 2 | 18/20 passed | 90,790 ms | `NW-T2-04` `Completed on` timeout with CPU active-job conflict; `NW-T4-02` Configure dialog missing | 0 | 0 | 0 |
| 4 | 17/20 passed | 206,121 ms | `NW-T2-04` active-job/`Completed on` timeout; builder-tab `Untitled model` isolation failure; `NW-T2-05` 180-second timeout and browser-close cleanup error | 0 | 1 before explicit cleanup | 0 |

The lock scan used only each run's isolated `application.log` and the exact
signatures `database is locked`, `sqlite3.OperationalError`, `SQLite objects
created in a thread`, and `database table is locked`. The launcher transcript
also recorded readiness for every run. Workers 2 and 4 logged execution-job
`409` responses; the `422` responses were typed job rejections and were not
counted as SQLite failures.

## Worker-4 cleanup

The worker-4 suite left one disposable tensor-inspection project after its
180-second timeout. A fresh official-launcher instance on the same disposable
root deleted that exact project with HTTP `204`, confirmed `remaining_projects=0`,
stopped the owned backend/frontend listeners, and verified zero remaining
listeners on ports 9853 and 6429. The cleanup transcript is
[official-launcher-nw-t5-02-workers-4-cleanup.txt](logs/official-launcher-nw-t5-02-workers-4-cleanup.txt).

## Evidence

- [Worker-1 launcher](logs/official-launcher-nw-t5-02-workers-1.txt) and [workflow transcript](logs/nw-t5-02-workers-1-workflows.log)
- [Worker-2 launcher](logs/official-launcher-nw-t5-02-workers-2.txt) and [workflow transcript](logs/nw-t5-02-workers-2-workflows.log)
- [Worker-4 launcher](logs/official-launcher-nw-t5-02-workers-4.txt) and [workflow transcript](logs/nw-t5-02-workers-4-workflows.log)
- [Result and lock summary](logs/nw-t5-02-results.log)
- [Cleanup record](logs/process-cleanup-nw-t3-t5.md)
- [Worker-4 active-job failure](evidence/nw-t5-02-workers-4-active-job-failure.md) and [screenshot](evidence/nw-t5-02-workers-4-active-job-failure.png)
- [Worker-4 builder-tab failure](evidence/nw-t5-02-workers-4-builder-tabs-failure.md) and [screenshot](evidence/nw-t5-02-workers-4-builder-tabs-failure.png)
- [Worker-4 tensor-inspection failure](evidence/nw-t5-02-workers-4-tensor-inspection-failure.md) and [screenshot](evidence/nw-t5-02-workers-4-tensor-inspection-failure.png)

No code change was made to the workflow suite or to SQLite behavior in this
slice. Close `ISSUE-001` only after a future unchanged 1/2/4-worker run is
clean at every count, including fixture cleanup and the existing worker/job
admission flows.
