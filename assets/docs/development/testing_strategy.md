# Testing Strategy

Last updated: 2026-08-20

Backend unit and contract tests cover frozen worker configuration, strict backend-policy validation, framework-free server imports, live metadata, OpenAPI job schemas, per-device lease contention/concurrency, lifecycle transitions, idempotent stop, escalation, crash recovery, shutdown, bounded history, artifact revision conflicts, and temporary-workspace cleanup. Deterministic fake workers exercise pause/crash/cancel paths without optional frameworks. The existing Keras smoke/export checks run through fresh workers and skip only when no operational backend is installed.

Frontend tests cover runtime API/backend/device/policy controls, active-worker diagnostics, stop/switch decisions, job polling, artifact gating, graph validation, and existing Builder/catalog behavior. UI workflows are verified with the in-app Browser at desktop and mobile sizes; captures and lifecycle logs belong under `assets/QA/`.

The real-runtime matrix is conditional: repeat smoke/export cycles and backend/device switching only when the corresponding installed runtime and hardware are available. Native inference, training, datasets, checkpoints, and unsupported TensorFlow/JAX GPU combinations remain explicit skips/deferred outcomes, never passing capability claims. Pytest, Ruff, uv, npm, Vite, TypeScript, lint, Python bytecode, and coverage caches/artifacts are configured below `assets/cache`; `assets/QA` remains reserved for QA evidence. Before handoff run affected backend tests, frontend test/lint/build, focused Ruff/type checks, and a process/port cleanup check.
