# System Overview

Last updated: 2026-08-05

The repository contains a flat FastAPI backend in `app/server`, a database initializer in `app/scripts`, a single React TypeScript frontend in `app/client/src`, manifest resources in `app/server/resources/nodes`, example projects in `examples`, and focused documentation under `assets/docs`. The frontend build output is generated under `app/client/dist`; when present, the backend serves its assets and SPA fallback. The backend owns graph contracts, validation, compilation, persistence, and exports. The frontend owns editing state and interaction, but backend validation remains authoritative.

The backend is launched through `app:create_app --factory`. Lifespan composition creates the catalog, repository context, and isolated Keras backend runner; no application singleton or mutable process-wide runtime cache is used. Keras is the API layer over independently selectable TensorFlow/Torch runtimes.
