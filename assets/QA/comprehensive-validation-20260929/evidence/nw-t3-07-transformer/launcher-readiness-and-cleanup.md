# NW-T3-07 launcher readiness and cleanup

Date: 2026-09-29

- Official launcher: `start_on_windows.ps1 -Action Launch -Backends tensorflow,pytorch,jax -Acceleration Auto -ExportTools -NoBrowser -ResourcesDir runtimes/cache/nw-t3-07-transformer-20260929`
- Backend: `http://127.0.0.1:9750/health` returned `{"status":"ok"}`.
- Frontend: `http://127.0.0.1:6425/` was reachable and rendered the Builder.
- Runtime lane: Python 3.13.14, Keras 3.15.1, TensorFlow 2.21.0 CPU,
  PyTorch 2.13.0+cu126 on NVIDIA GeForce RTX 3060 Laptop GPU, and JAX 0.11.0
  CPU. Native compiler discovery was TensorFlow JIT enabled, PyTorch
  `torch_compile` unavailable, and JAX JIT enabled.
- Isolated database/resource root: `runtimes/cache/nw-t3-07-transformer-20260929`.
- Cleanup: the successful browser fixtures deleted their projects in `finally`
  blocks. One project from the timed-out first repair attempt was identified
  by ID and removed through `DELETE /api/projects/{id}` after a short isolated
  launcher restart; the final database query returned zero projects. The
  task-owned backend listener (9750) and frontend listener (6425) were then
  stopped, and a final listener check found neither port in LISTEN state.
