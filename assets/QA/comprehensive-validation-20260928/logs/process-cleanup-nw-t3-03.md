# NW-T3-03 process cleanup

Date: 2026-09-28

- The official launcher created task-owned backend and frontend services for ports 9853 and 6430.
- Final service processes observed before cleanup: backend child PID 36248 (uvicorn, parent launcher PID 30396) and frontend child PID 28184 (preview server, parent launcher PID 8932).
- Both task-owned process trees were stopped successfully.
- The isolated data root runtimes/cache/nw-t3-03-convolution-20260928 was removed after the final logs and evidence were copied under assets/QA.
- A final port check found no listeners on 9853 or 6430.
- No task-owned browser, worker, watcher, or helper process was left running.
