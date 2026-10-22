# Changelog

Last updated: 2026-10-18

## [Unreleased]

- Launcher serves the frontend from the backend on one origin, stays attached as the lifecycle owner (Enter/Ctrl+C to stop), and no longer starts the Node preview server.
- Fixed Settings section links leaving `/settings` under the backend-served build (`<base href="/">`), `/openapi.json` and `/docs` failing with HTTP 500 (`typing.Self` in `ParameterValueSchema`), and missing static files returning the SPA shell instead of 404.

## [1.0.0] - 2026-10-18

First stable release of NeuroWeave Studio as a locally hosted web application.

### Included

- Angular graph editor and FastAPI backend with a manifest-driven node catalog, typed graph validation, and project persistence.
- Custom node templates, external dataset registration, synthetic Functional Validation, and Tensor Inspection.
- Capability-gated source, project bundle, and model-format exports with isolated worker jobs and revision-safe artifact publication.
- Windows launcher for pinned local runtimes, dependency setup, SQLite initialization and migrations, frontend builds, and service readiness checks.

### Scope

- Functional Validation and Tensor Inspection use synthetic inputs and initialized weights; training and prediction serving are not implemented.
- Dataset definitions and split settings do not download dataset bytes or materialize preprocessing partitions.
- Exports contain untrained model state. Keras architecture import does not import weights or custom code.
