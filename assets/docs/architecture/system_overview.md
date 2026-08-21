# System Overview

Last updated: 2026-08-21

NeuroWeave Studio contains a flat FastAPI backend in `app/server`, a React TypeScript frontend in `app/client/src`, manifest resources, example projects, and focused documentation under `assets/docs`. The backend owns graph contracts, execution-profile-derived validation, compilation, persistence, source export, import, and process-isolated runtime work. The frontend owns editing state and interaction; backend validation remains authoritative.

The backend is launched through `app:create_app --factory`. Lifespan probes optional runtimes through fresh interpreters, creates the application-scoped job controller, and reaps every active worker before disposing repositories. The server itself stays framework-neutral: Keras/TensorFlow/JAX/PyTorch imports occur only inside a worker process after startup configuration.

Each project persists `execution_backend` and `keras_enabled` as one model execution profile. Keras-enabled graphs use the Keras implementation with the persisted backend; Keras-disabled graphs use a backend-specific native implementation only where the manifest and exporter support it. This profile drives static validation, shape checks, source generation, and runtime routing. Keras is an API layer, not a fourth backend.

Execution is asynchronous. `POST /api/execution/jobs` accepts a discriminated `smoke_inference` or `keras_export` operation and returns `202`; clients list/poll jobs, request an idempotent stop, and retrieve a verified export artifact only after success. Source generation remains synchronous because it does not initialize an ML runtime. Device and memory-policy preferences are session-scoped, while the model backend remains persisted and authoritative.

Keras is the operational model API. Native runtime inference is an explicit deferred capability (`not_implemented`); native models may still validate and generate source for the verified subset without implying internal execution. Training, checkpoints, datasets, and experiment tracking are outside the current system boundary.
