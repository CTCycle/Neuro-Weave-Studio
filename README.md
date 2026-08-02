# NeuroWeave Studio

NeuroWeave Studio is a local-first visual editor for designing machine-learning model architectures without writing code. It uses a framework-independent graph schema, manifest-defined nodes, a FastAPI backend, and a React plus XYFlow frontend.

The phrase “Django DP” in the original brief was treated as ambiguous and non-essential. FastAPI is the canonical backend bootstrap because it was the explicit technology requirement. No Django dependency or compatibility path is included.

## Implemented MVP

- Searchable catalog with 85 built-in node manifests.
- Drag-and-drop graph editing with typed ports, pan, zoom, minimap, selection, deletion, copy and paste, undo and redo, and automatic layout.
- Client-side cycle prevention and connection checks.
- Backend DAG validation, port validation, parameter validation, compatibility diagnostics, and representative tensor-shape propagation.
- SQLite or PostgreSQL project persistence with immutable revision snapshots.
- Project create, save, reopen, duplicate, delete, JSON import, and JSON export.
- Keras, TensorFlow through Keras, and PyTorch source generation for the documented supported subset.
- Optional uninitialized Keras `.keras` export when TensorFlow and Keras optional dependencies are installed and the graph uses binary-export-supported nodes.
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
.\start_on_windows.ps1 -Mode Launch
```

Use `-NoBrowser` for automated validation. Other modes are `Install`, `InitializeDatabase`, `Dev`, `Backend`, `Frontend`, `Build`, `Test`, and `Stop`.

### Linux or macOS

```bash
./scripts/start.sh
```

The frontend runs at `http://127.0.0.1:5173`; the API runs at `http://127.0.0.1:8000`.

## Manual setup

```bash
cd app/server
uv sync --frozen --extra test
uv run uvicorn neuro_weave.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd app/client
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

## Validation

```bash
cd app/server
uv run ruff check .
uv run pytest
uv run python ../scripts/initialize_database.py

cd ../client
npm ci
npm run lint
npm run test
npm run build
```

## Optional Keras binary export

```bash
cd app/server
uv sync --extra keras --extra test
```

Binary export creates an initialized but untrained `.keras` model. It is not a trained model, does not contain learned weights, and supports only the explicitly documented subset.

## Examples

Portable project definitions are under `app/resources/examples/`. Import them through the editor or POST their project payload to `/api/projects`.

Configuration is read from `settings/.env`. `EMBEDDED_DATABASE=true` uses the automatically created SQLite database at `app/resources/runtime/neuro_weave.db`; set it to `false` and provide the `DATABASE_*` PostgreSQL settings for an external database. Generated databases, exports, logs, launcher state, and QA data belong under the ignored `app/resources/runtime/` and `assets/QA/` directories. Run `start_on_windows.ps1 -Mode InitializeDatabase` only when you intentionally want to reset the application schema.

## Documentation

Start with [`assets/docs/project_index.md`](assets/docs/project_index.md). It routes to focused architecture, ontology, node, UI, development, runtime, and decision documents.

## License

A license has not been selected. See `LICENSE.md` before distributing the project.
