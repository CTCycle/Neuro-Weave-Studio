# System Overview

Last updated: 2026-08-31

NeuroWeave Studio contains a flat FastAPI backend in `app/server`, a standalone Angular TypeScript frontend in `app/client/src/app`, manifest resources, example projects, and focused documentation under `assets/docs`. The backend owns graph contracts, execution-profile-derived validation, compilation, persistence, source export, import, and process-isolated runtime work. The frontend owns editing state and interaction; backend validation remains authoritative.

The backend is launched through `app:create_app --factory`. Lifespan probes optional runtimes through fresh interpreters, creates the application-scoped job controller, and reaps every active worker before disposing repositories. The server itself stays framework-neutral: Keras/TensorFlow/JAX/PyTorch imports occur only inside a worker process after startup configuration.

Each project persists `execution_backend` and `keras_enabled` as one model execution profile. Keras-enabled graphs use the Keras implementation with the persisted backend; Keras-disabled graphs use a backend-specific native implementation only where the manifest and exporter support it. This profile drives static validation, shape checks, source generation, and runtime routing. Keras is an API layer, not a fourth backend.

Execution is asynchronous. `POST /api/execution/jobs` accepts a discriminated `smoke_inference` or `model_export` operation and returns `202`; clients list/poll jobs, request an idempotent stop, and retrieve a verified export artifact only after success. `GET /api/exports/{project_id}/capabilities` is the capability-driven export contract. Source generation remains synchronous because it does not initialize an ML runtime. Device and memory-policy preferences are session-scoped, while the model backend remains persisted and authoritative.

`ExportService` centralizes discovery, compatibility, adapters, bridge conversion, package manifests, structural/reload/inference verification, error mapping, and atomic publication. Keras-enabled exports use the Keras API; native exports use explicit raw framework APIs inside workers, never a Keras fallback. TensorFlow and JAX native execution are enabled only for the verified common 27-node subset; attention and transformer nodes remain Keras-only. All binary methods are initialized, inference-only, and do not preserve training state. Checkpoint formats are weights-only packages, not training pipelines.
