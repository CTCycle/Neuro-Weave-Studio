# System Overview

Last updated: 2026-08-02

## System summary

NeuroWeave is a local-first visual machine-learning architecture editor. The
React and XYFlow client edits a framework-independent graph; the FastAPI backend
owns the canonical contracts, validation, compilation, persistence, and export
boundaries.

## Repository structure

```text
.
|- app/
|  |- client/src/                 React, TypeScript, Zustand, XYFlow UI
|  |- server/neuro_weave/         FastAPI application and domain modules
|  |  |- api/routes/              HTTP routers
|  |  |- catalog/                  manifest loading
|  |  |- compilation/              framework-independent compiler IR
|  |  |- domain/                   graph, tensor, manifest, and result models
|  |  |- exporters/                Keras and PyTorch source adapters
|  |  |- persistence/              database backends, schema, and repositories
|  |  `- validation/               DAG, port, parameter, and shape checks
|  |- resources/examples/          portable example projects
|  `- scripts/initialize_database.py explicit schema reset entry point
|- assets/docs/                    focused project documentation
|- settings/.env.example           canonical runtime and database template
`- start_on_windows.ps1            Windows launcher and maintenance entry point
```

## Runtime topology

In development, the launcher runs Uvicorn on `127.0.0.1:8000` and Vite on
`127.0.0.1:5173`. Vite proxies `/api` to FastAPI. The frontend owns transient
editing state and history; the backend remains authoritative for validation and
persisted project state.

The application database is selected through `settings/.env`. Missing embedded
SQLite files are initialized on first backend startup. Existing files and all
external PostgreSQL schemas require the explicit initializer for destructive
resets.

## Entry points

- Backend: `app/server/neuro_weave/app.py`, exposing `create_app()` and `app`.
- Frontend: `app/client/src/main.tsx`, bootstrapping `App.tsx`.
- Windows: `start_on_windows.ps1` or `start_on_windows.bat`.
- Database reset: `app/scripts/initialize_database.py` or launcher mode
  `InitializeDatabase`.
