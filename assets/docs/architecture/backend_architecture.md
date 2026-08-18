# Backend Architecture

Last updated: 2026-08-18

The flat `app/server` layout exposes `app.py` as a FastAPI application factory, with `api`, `common`, `configurations`, `domain`, `repositories`, and `services` as the primary layers. NeuroWeave-specific `catalog`, `compilation`, `exporters`, and `validation` packages remain direct server packages so graph behavior stays unchanged.

The request path is strictly endpoint → service → repository. Routers parse Pydantic contracts and map errors; catalog, runtime metadata, validation, compilation, project, and export services own application logic. Pure graph algorithms stay in `validation`. SQLAlchemy repositories persist the canonical relational project graph and revisions through a shared unit of work; services own transaction boundaries.

`create_app()` is the only application construction entry point. Lifespan setup loads the catalog, repository context, and the isolated `KerasBackendRunner` into typed `app.state` dependencies and disposes the SQLAlchemy engine on shutdown. There is no process-global settings, catalog, repository, or compatibility cache.

Keras 3 is the fixed model API layer. The public code targets are `keras`, native `tensorflow`, native `pytorch`, and native `jax`/Flax. Execution backends are independently probed as `tensorflow`, `pytorch`, and `jax`; the public PyTorch id is mapped to Keras' required `KERAS_BACKEND=torch` only inside the worker. Probes, binary construction, and provenance logging run in isolated workers with `KERAS_BACKEND` set before Keras imports.

The API is schema v2: project persistence uses `code_target`, metadata exposes `code_targets` and per-backend `execution_backends`, and unavailable binary export requests return a structured `409 backend_unavailable` response.

The base runtime is pinned to Python 3.13.14 (`>=3.13,<3.14`) because the locked Windows environment contains compatible TensorFlow, PyTorch, and JAX CPython 3.13 wheels. TensorFlow and JAX remain CPU-only on native Windows; PyTorch may use the selected CUDA wheel. Optional framework installation is launcher-managed through locked uv extras, not performed by the running GUI. The launcher records requested/resolved extras and live package/probe status, while `/api/metadata` exposes installed version and operational capability to the frontend.
