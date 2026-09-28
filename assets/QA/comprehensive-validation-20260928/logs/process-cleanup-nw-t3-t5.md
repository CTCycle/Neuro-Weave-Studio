# Process and fixture cleanup

Date: 2026-09-28

## Recurrent wrapper slice

- Official launcher run root: `runtimes/cache/nw-t3-recurrent-wrappers-20260928`.
- Backend listener `127.0.0.1:9853` and frontend listener `127.0.0.1:6429` were stopped after the final catalog run.
- The final listener check reported zero task-owned listeners.
- Catalog fixtures were deleted by the E2E `finally` paths.

## NW-T5-02 worker matrix

- Worker 1, 2, and 4 runs used separate disposable roots under `runtimes/cache/`.
- Each worker run stopped its exact backend/frontend listener PIDs and reported zero remaining listeners.
- Worker 4 left one disposable tensor-inspection project after its timeout. The explicit cleanup launcher deleted the exact project with HTTP `204`, then the API returned zero projects.
- The cleanup launcher was stopped and the final check reported zero listeners on ports 9853 and 6429.
- The protected canonical Python cache was not targeted; isolated pytest and launcher roots were used.
