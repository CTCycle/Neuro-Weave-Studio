# System Overview

Last updated: 2026-08-20

NeuroWeave Studio contains a flat FastAPI backend in `app/server`, a React TypeScript frontend in `app/client/src`, manifest resources, example projects, and focused documentation under `assets/docs`. The backend owns graph contracts, validation, compilation, persistence, source export, and process-isolated Keras execution. The frontend owns editing state and interaction; backend validation remains authoritative.

The backend is launched through `app:create_app --factory`. Lifespan probes optional runtimes through fresh interpreters, creates the application-scoped job controller, and reaps every active worker before disposing repositories. The server itself stays framework-neutral: Keras/TensorFlow/JAX/PyTorch imports occur only inside a worker process after startup configuration.

Execution is asynchronous. `POST /api/execution/jobs` accepts a discriminated `smoke_inference` or `keras_export` operation and returns `202`; clients list/poll jobs, request an idempotent stop, and retrieve a verified export artifact only after success. Source generation remains synchronous because it does not initialize an ML runtime. Runtime selection is session-scoped and never persisted in project data.

Keras is the operational model API. The `native` API is an explicit deferred capability (`not_implemented`) so the UI and contracts can prepare for future adapters without claiming unsupported inference. Training is outside the current system boundary.
