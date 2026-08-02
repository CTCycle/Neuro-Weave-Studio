# Backend Architecture

Last updated: 2026-08-02

## Layer responsibilities

- `api/routes` owns HTTP paths, request/response models, status codes, and
  dependency wiring. Route handlers do not contain graph algorithms or SQL.
- `domain` owns Pydantic graph, tensor, manifest, diagnostic, project, and
  compilation contracts.
- `catalog` loads and validates immutable node manifests from repository
  resources.
- `validation` performs DAG, typed-port, required-connection, parameter,
  compatibility, safe-expression, shape, and parameter-count checks.
- `compilation` turns a validated graph into framework-independent IR.
- `exporters` translate the IR into Keras/TensorFlow-through-Keras and PyTorch
  source, with optional initialized `.keras` export when dependencies are
  installed.
- `services` coordinate validation and persistence for project commands.
- `persistence` owns SQLAlchemy models, the selected database backend, the
  shared `RepositoryContext`, explicit transactions, and project revisions.

## Request flow

```text
HTTP request
  -> FastAPI route and Pydantic contract
  -> service orchestration
  -> validator/compiler/exporter or ProjectRepository
  -> typed response / persisted project state
```

The database is composed once per process. `ProjectRepository` receives the
immutable context explicitly and opens short-lived sessions for reads and
write transactions. Schema creation and reset are confined to
`persistence/initializer.py` and the standalone initializer script.

## Failure boundaries

Validation failures return structured diagnostics. Missing projects return the
route’s documented 404 response. Revision conflicts return HTTP 409. Database
connection or schema failures are startup/dependency failures and must not be
reported as successful in-memory saves.
