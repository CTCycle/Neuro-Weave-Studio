# NeuroWeave Studio

Last updated: 2026-09-01

NeuroWeave Studio is a local-first visual editor for designing machine-learning model architectures without writing code. It uses a framework-independent graph schema, manifest-defined nodes, a FastAPI backend, and an Angular plus Foblex frontend.

The current product release is `1.0.0`, following Semantic Versioning. The frontend package, backend package, runtime API metadata, and lockfiles use this same release version.

The phrase “Django DP” in the original brief was treated as ambiguous and non-essential. FastAPI is the canonical backend bootstrap because it was the explicit technology requirement. No Django dependency or compatibility path is included.

## Implemented MVP

- Searchable catalog with 85 built-in node manifests.
- Drag-and-drop graph editing with typed ports, pan, zoom, minimap, selection, deletion, copy and paste, undo and redo, and automatic layout.
- Client-side cycle prevention and connection checks.
- Backend DAG validation, port validation, parameter validation, compatibility diagnostics, and representative tensor-shape propagation.
- Inference-only model smoke tests with generated dummy tensors, isolated Keras workers, tensor metadata, parameter counts, and actionable runtime errors.
- SQLite project persistence with immutable revision snapshots.
- Project create, save, reopen, duplicate, delete, JSON import, and JSON export.
- Keras-enabled source generation for the documented portable subset, using the model's persisted execution backend.
- Native PyTorch source generation for its verified subset; native TensorFlow and JAX/Flax source capabilities remain unavailable until parity is proven.
- Optional uninitialized Keras `.keras` export for Keras-enabled models when the persisted backend is operational and the graph uses binary-export-supported nodes.
- Source-only `.keras` architecture import through the allowlisted Keras configuration format; imported weights and custom code are never accepted.
- MLP, convolutional classifier, LSTM, and Transformer encoder examples.

## Important terminology

Every model persists one execution profile: `execution_backend` (`tensorflow`, `pytorch`, or `jax`) and `keras_enabled` (`true` or `false`). With Keras enabled, the canonical graph is implemented through Keras and runs against the persisted backend. With Keras disabled, the graph uses the backend-specific native adapter where that adapter is verified. Keras is an API layer, not a fourth backend.

## Requirements

- Python 3.13.14 (the launcher rejects other Python versions).
- Node.js 20 or newer.
- `uv` is recommended for Python dependency management.

## Development startup

### Windows

```powershell
.\start_on_windows.ps1
```

The launcher menu provides these actions:

- Launch application
- Install or repair base dependencies
- Rebuild frontend
- Manage compute backends
- Initialize or upgrade database
- Run the validation suite
- Check for updates
- Update application
- Remove logs
- Clear cache
- Remove all data
- Uninstall application
- Exit

The launcher prepares the repository-local Python, uv, and Node.js runtimes only when they are missing or unusable. Option 2 installs or repairs dependencies and rebuilds the frontend; option 3 rebuilds only the frontend from current source using the existing frontend dependencies; option 1 reuses the existing build and never rebuilds it automatically. If `app/tests/cache/frontend-dist/index.html` is missing, run the standalone rebuild before launching. **Check for updates** performs a read-only remote status check, while **Update application** requires a non-detached, clean checkout of `main` and runs `git pull --ff-only origin main`; it never switches branches or modifies local changes. **Remove all data** deletes generated database/runtime/export/job/log data while preserving application files, settings, runtimes, dependencies, caches, and build output. Use `.\start_on_windows.ps1 -Action RebuildFrontend` to run the standalone rebuild directly. It starts the backend at `http://127.0.0.1:8000` and the frontend at `http://127.0.0.1:5173`, waits for both readiness checks, opens the frontend in the default browser when possible, and then closes the initial PowerShell menu while leaving the independent application process roots running. Use `.\start_on_windows.ps1 -Action Launch -NoBrowser` to skip automatic browser opening.

`settings/.env` is created from `settings/.env.example` on first launcher or direct backend startup and is never overwritten. The default SQLite database is `app/resources/database.db`. Alembic owns schema creation and evolution: installation, explicit initialization, and every backend startup migrate a missing or current-schema database to the current head. Existing databases without the current Alembic history are unsupported and are rejected without legacy adoption or mutation. Keras is core and is installed with the base application. Optional TensorFlow, PyTorch, and JAX installation is explicit through `.\start_on_windows.ps1 -Action ManageBackends`; the interactive prompt offers numbered PyTorch GPU/CUDA and CPU-only choices, along with TensorFlow/JAX CPU-only and all-backend configurations. The launcher reports native-Windows CPU limitations for TensorFlow/JAX and verifies actual worker capability after installation. Runtime/download caches are under `runtimes/cache`; pytest cache and temporary test directories are under `assets/QA`; Ruff, Angular frontend build output, TypeScript metadata, Python bytecode, coverage, Playwright traces, and lint caches are under `app/tests/cache`.

Set `BACKEND_LOGS_VISIBLE=false` in `settings/.env` to keep backend output in the runtime log files instead of opening a dedicated terminal; the default is `true`.

## Manual setup

```powershell
cd app/server
$runtimeCacheRoot = (Resolve-Path ../..).Path + '\\runtimes\\cache'
$toolCacheRoot = (Resolve-Path ../..).Path + '\\app\\tests\\cache'
$env:UV_CACHE_DIR = $runtimeCacheRoot + '\\uv'
$env:PIP_CACHE_DIR = $runtimeCacheRoot + '\\pip'
$env:PYTHONPYCACHEPREFIX = $toolCacheRoot + '\\python-bytecode'
$env:COVERAGE_FILE = $toolCacheRoot + '\\coverage\\.coverage'
$env:PYTHONPATH = (Get-Location).Path
uv sync --extra test
uv run uvicorn app:create_app --factory --reload --port 8000
```

In another terminal:

```bash
cd app/client
npm install
npm run dev
```

## Database configuration

SQLite is the only supported database. Select **Initialize or upgrade database** in the launcher, or run the backend script directly:

```powershell
cd app/server
$env:PYTHONPATH = (Get-Location).Path
uv run python ../scripts/initialize_database.py
```

The operation is repeatable and non-destructive. A missing or empty SQLite file is created by the first Alembic upgrade. Startup and initialization serialize migrations with a database-local lock, apply pending revisions, and verify the single Alembic head. Existing data is preserved by migrations; no reset option is exposed.

For development migration work, use the Alembic environment from `app/server`:

```powershell
uv run alembic revision --autogenerate -m "describe the schema change"
# Review and edit the generated revision before applying it.
uv run alembic upgrade head
uv run alembic current --check-heads
uv run alembic check
```

## Validation

Select **Run validation suite** in the launcher, or run the backend and frontend checks manually:

```powershell
cd app/server
uv run ruff check . ..\tests ..\scripts
uv run pyright
uv run pytest ..\tests

cd ../client
npm run lint
npm run test
npm run build
npm run e2e
```

## Execution profiles, compilation, and artifacts

```bash
cd app/server
uv sync --extra test --extra tensorflow  # or --extra torch-cpu / --extra torch-cu126
```

The Builder persists `execution_backend` and `keras_enabled` with the graph. Validation, source generation, and runtime job submission derive their implementation from that profile; the persisted backend is authoritative for the model. The metadata endpoint reports live backend probes and a catalog-driven compilation configuration. Keras compilation settings are useful for Keras source and binary export, but NeuroWeave does not provide training loops, checkpoints, or native optimizer execution.

Binary export creates an initialized but untrained `.keras` model. It is not a trained model, does not contain learned weights, and supports only the explicitly documented subset. With no operational backend, export is disabled and the API returns a structured unavailable-backend error. Keras architecture import accepts only standard `.keras` configuration metadata, creates a canonical graph, and warns that weights are not imported.

The workspace **Smoke test** is a dry run for Keras-enabled models: it validates the saved graph, generates compatible zero-filled inputs, runs one inference pass, and reports tensor shapes, dtype, parameter count, device, and status. Native-only models retain static, shape, and source validation where supported, while native runtime inference remains explicitly unavailable. No path trains or updates weights.

## Examples

Portable project definitions are under `examples/`. Import them through the editor or POST their project payload to `/api/projects`.

## Documentation

Start with [`assets/docs/project_index.md`](assets/docs/project_index.md). It routes to focused architecture, ontology, node, UI, development, runtime, and decision documents.

## License

A license has not been selected. See `LICENSE.md` before distributing the project.
