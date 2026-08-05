# NeuroWeave Studio

NeuroWeave Studio is a local-first visual editor for designing machine-learning model architectures without writing code. It uses a framework-independent graph schema, manifest-defined nodes, a FastAPI backend, and a React plus XYFlow frontend.

The phrase “Django DP” in the original brief was treated as ambiguous and non-essential. FastAPI is the canonical backend bootstrap because it was the explicit technology requirement. No Django dependency or compatibility path is included.

## Implemented MVP

- Searchable catalog with 85 built-in node manifests.
- Drag-and-drop graph editing with typed ports, pan, zoom, minimap, selection, deletion, copy and paste, undo and redo, and automatic layout.
- Client-side cycle prevention and connection checks.
- Backend DAG validation, port validation, parameter validation, compatibility diagnostics, and representative tensor-shape propagation.
- SQLite project persistence with immutable revision snapshots.
- Project create, save, reopen, duplicate, delete, JSON import, and JSON export.
- Keras, TensorFlow through Keras, and PyTorch source generation for the documented supported subset.
- Optional uninitialized Keras `.keras` export when a selected TensorFlow or Torch backend is operational and the graph uses binary-export-supported nodes.
- MLP, convolutional classifier, LSTM, and Transformer encoder examples.

## Important terminology

Keras is the high-level model API. TensorFlow through Keras uses Keras with the TensorFlow runtime. PyTorch is the framework built on the Torch tensor library. NeuroWeave does not present Torch and PyTorch as independent equivalent backends.

## Requirements

- Python 3.12 or newer.
- Node.js 20 or newer.
- `uv` is recommended for Python dependency management.

## Development startup

### Windows

```powershell
.\start_on_windows.ps1
```

The launcher menu provides these actions:

- Launch application
- Install or repair dependencies
- Initialize or recreate database
- Run the validation suite
- Exit

The launcher prepares the repository-local Python, uv, and Node.js runtimes only when they are missing or unusable. It starts the backend at `http://127.0.0.1:8000` and the frontend at `http://127.0.0.1:5173`, then owns both child processes until Ctrl+C.

`settings/.env` is created from `settings/.env.example` on first launcher or direct backend startup and is never overwritten. The default SQLite database is `app/resources/runtime/neuro_weave.db`. Existing databases must contain the canonical schema marker; use the explicit reset action to recreate a local database from scratch when needed. Keras is core; the launcher install menu selects Keras-only, TensorFlow, Torch, or both runtime extras.

## Manual setup

```bash
cd app/server
uv sync --extra test
PYTHONPATH=. uv run uvicorn app:create_app --factory --reload --port 8000
```

In another terminal:

```bash
cd app/client
npm install
npm run dev
```

## Database configuration

SQLite is the default. Select **Initialize or recreate database** in the launcher. Type `RESET` when prompted to drop and recreate the managed schema. For PostgreSQL, set `DATABASE_EMBEDDED=false` and provide the connection fields in `settings/.env` before running the action. Normal startup only checks the canonical schema marker; it does not repair or migrate databases.

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
```

## Keras backend selection and binary export

```bash
cd app/server
uv sync --extra test --extra tensorflow  # or --extra torch, or both
```

Keras is the model API layer and TensorFlow/Torch are runtime backends. The editor discovers operational backends, auto-selects the only available backend, and requires a selector choice when several are installed. Binary export creates an initialized but untrained `.keras` model. It is not a trained model, does not contain learned weights, and supports only the explicitly documented subset. With no installed backend, export is disabled and the API returns 501 for a valid unavailable-backend request.

## Examples

Portable project definitions are under `examples/`. Import them through the editor or POST their project payload to `/api/projects`.

## Documentation

Start with [`assets/docs/project_index.md`](assets/docs/project_index.md). It routes to focused architecture, ontology, node, UI, development, runtime, and decision documents.

## License

A license has not been selected. See `LICENSE.md` before distributing the project.
