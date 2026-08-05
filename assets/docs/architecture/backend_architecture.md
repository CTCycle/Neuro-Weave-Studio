# Backend Architecture

Last updated: 2026-08-05

The flat `app/server` layout exposes `app.py` as a FastAPI application factory, with `api`, `common`, `configurations`, `domain`, `repositories`, and `services` as the primary layers. NeuroWeave-specific `catalog`, `compilation`, `exporters`, and `validation` packages remain direct server packages so graph behavior stays unchanged.

The request path is strictly endpoint → service → repository. Routers parse Pydantic contracts and map errors; catalog, runtime metadata, validation, compilation, project, and export services own application logic. Pure graph algorithms stay in `validation`. SQLAlchemy repositories persist the canonical relational project graph and revisions through a shared unit of work; services own transaction boundaries.

`create_app()` is the only application construction entry point. Lifespan setup loads the catalog, repository context, and `KerasBackendRunner` into typed `app.state` dependencies and disposes the SQLAlchemy engine on shutdown. There is no process-global settings, catalog, repository, or compatibility cache.

Keras is the model API layer; TensorFlow and Torch are selectable runtime backends. Backend probes and binary model construction run in isolated workers with `KERAS_BACKEND` set before the worker imports Keras. Direct PyTorch source generation and TensorFlow-through-Keras source generation remain separate from binary backend selection.
