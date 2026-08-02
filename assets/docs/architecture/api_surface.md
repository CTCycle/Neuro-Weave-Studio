# API Surface

Last updated: 2026-08-02

All business endpoints are mounted under `/api`. The frontend uses the relative
`/api` base path so development proxying and packaged serving share one boundary.

## Root and health

- `GET /health`: process health check.
- `GET /api/metadata`: application version, framework choices, and model-level
  configuration options.
- `GET /docs`, `GET /redoc`, `GET /openapi.json`: FastAPI documentation.

## Catalog

- `GET /api/nodes/catalog`: all validated built-in node manifests.
- `GET /api/nodes/{definition_id}/{version}`: one manifest by stable identity.

## Projects

- `GET /api/projects`: summaries ordered by most recently updated.
- `POST /api/projects`: create a project and its revision-one snapshot.
- `GET /api/projects/{project_id}`: reopen the current project document.
- `PUT /api/projects/{project_id}`: save a new immutable revision, optionally
  guarded by `expected_revision`.
- `DELETE /api/projects/{project_id}`: delete the project and revision snapshots.
- `POST /api/projects/{project_id}/duplicate`: create an independent copy.
- `GET /api/projects/{project_id}/export`: portable JSON bundle.
- `POST /api/projects/import`: create a project from a portable bundle.

## Validation, compilation, and exports

- `POST /api/validation`: backend graph validation and diagnostics.
- `POST /api/compilation`: framework-independent compilation plus generated
  source for the requested framework.
- `GET /api/exports/{project_id}/source/{framework}`: downloadable generated
  source for a saved project.
- `POST /api/exports/{project_id}/keras-model`: optional initialized `.keras`
  binary export.

The backend remains authoritative: client-side cycle prevention improves
interaction feedback, but persisted saves and source generation still pass
through backend validation and compatibility checks.
