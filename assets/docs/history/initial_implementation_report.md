# NeuroWeave Studio Implementation Report

Last updated: 2026-07-30

> Archived from the repository root during the 2026-08-02 NeuroWeave Studio restructuring. The verification notes below describe the pre-restructuring import baseline and are retained for provenance.

## Result

A complete initial repository was created for a local-first visual machine-learning architecture editor. The product uses FastAPI, SQLAlchemy, SQLite, React, TypeScript, Zustand, and XYFlow. Its canonical graph schema is independent of Keras, TensorFlow, and PyTorch.

The ambiguous phrase “Django DP” was treated as non-essential. FastAPI is the only backend bootstrap and there is no Django compatibility path.

## ParaGraph principles reused

ParaGraph was inspected read-only on its `develop` branch. The new repository reuses only architectural principles that fit this product:

- A backend and frontend split with typed service boundaries.
- Manifest-defined nodes loaded from repository resources.
- Explicit graph and portable bundle schema versions.
- Focused domain, service, repository, validation, and adapter modules.
- A root documentation index with narrow leaf documents.
- Windows-first launcher support plus cross-platform commands.

The interface, visual language, model ontology, graph contracts, persistence schema, validation system, and framework exporters are original to NeuroWeave Studio.

## Implemented scope

### Backend

- FastAPI application factory, CORS configuration, health and metadata routes.
- Project CRUD, duplication, optimistic revision checks, JSON import, and JSON export.
- SQLite persistence through SQLAlchemy.
- Immutable project revision snapshots.
- Alembic initial migration targeting the same configured database as the application.
- Repository-root-relative runtime path resolution.
- Pydantic graph, project, manifest, tensor, diagnostic, and compilation contracts.
- Manifest catalog loader with contract validation and duplicate-definition rejection.
- DAG validation, typed port validation, required connection checks, duplicate connection checks, parameter validation, output-name conflict checks, framework compatibility diagnostics, and safe expression validation.
- Shape propagation and representative parameter counts for common input, dense, reshape, flatten, embedding, convolution, pooling, recurrent, attention, Transformer, and merge nodes.
- Framework-independent compilation IR.
- Keras and TensorFlow-through-Keras source generation for the declared supported subset.
- PyTorch `nn.Module` source generation for the declared supported subset.
- Optional initialized but untrained `.keras` binary export.
- Generated source snapshots stored with saved projects when using project export routes.

### Frontend

- React, TypeScript, Vite, Zustand, and XYFlow scaffold.
- Original three-column model-design workspace.
- Searchable and framework-filterable node catalog.
- Drag-and-drop node creation.
- Typed input and output handles.
- Connection creation, deletion, validation, and client-side cycle prevention.
- Node label and parameter inspector.
- Model-level optimizer, learning rate, loss, metric, initializer, regularizer, constraint, and mixed-precision controls.
- Pan, zoom, minimap, grid snapping, selection, copy and paste, undo and redo, and deterministic automatic layout.
- Validation summaries, per-node diagnostics, tensor-shape display, parameter counts, and generated code preview.
- Project create, save, reopen, duplicate, JSON import, JSON export, and Keras binary export controls.
- Typed API client and centralized graph state.

### Catalog and examples

The built-in catalog contains 85 manifest-defined nodes:

- 8 input nodes.
- 11 core nodes.
- 18 convolutional nodes.
- 5 recurrent nodes.
- 10 Transformer and attention nodes.
- 14 merge nodes.
- 12 mathematical nodes.
- 7 output nodes.

Four portable examples are included:

- Multilayer perceptron classifier.
- Convolutional image classifier.
- LSTM sequence model.
- Small Transformer encoder.

## Verification performed

### Passed

- Python syntax compilation for backend source, tests, and migrations.
- Backend test suite: `15 passed`.
- Optional Keras binary test: `1 skipped` because TensorFlow is not installed in the execution environment.
- Alembic migration applied successfully to a clean SQLite database and created `projects`, `project_revisions`, and `alembic_version`.
- Strict offline TypeScript type check completed successfully using temporary declaration stubs for unavailable external packages.
- Real Uvicorn process startup succeeded.
- Health endpoint returned HTTP 200.
- Node catalog returned 85 manifests.
- The MLP example was created, reopened, validated with zero Keras diagnostics, and converted through project routes into Keras and PyTorch source.
- Keras and PyTorch generated source snapshots were persisted and visible after reopening the project.
- Generated PyTorch MLP source was executed with PyTorch and returned an output tensor of shape `(4, 3)` for an input tensor of shape `(4, 20)`.
- All four examples produce parseable Keras source in backend tests.
- All Markdown links in the repository documentation resolve to existing local files.

### Blocked by the execution environment

- `npm install` could not complete because the sandbox npm registry returned HTTP 404 for `@eslint/js`. Consequently, the real Vitest suite, ESLint run, and Vite production build could not be executed here.
- `uv sync --extra test` could not resolve because the sandbox Python registry did not provide Alembic. The backend tests were run successfully with the already installed global packages.
- Ruff was not installed and could not be obtained from the sandbox registry, so Ruff was not executed.
- TensorFlow was not installed, so an actual `.keras` file could not be generated in this environment. The direct exporter is covered by an optional test that runs when TensorFlow and Keras are available.

No claim is made that the blocked checks passed.

## Known limitations

- Training, dataset management, evaluation, experiment tracking, deployment, and learned-weight management are not implemented.
- Project revisions are persisted, but the MVP interface does not yet expose a revision history browser or restore action.
- Node grouping and reusable subgraphs are documented extension points, not implemented editor interactions.
- Custom nodes and plugins are not enabled. The manifest contract anticipates them, but executable third-party code is intentionally excluded from the MVP.
- Many catalog nodes are designable but not source-exportable. Validation blocks source generation when a selected framework mapping is unsupported.
- PyTorch support is intentionally narrower than Keras support. No channels-last to channels-first conversion layer is silently inserted.
- Keras binary export is an initialized, untrained architecture artifact. It contains no learned parameters.
- `.h5`, SavedModel, `state_dict`, full PyTorch pickle serialization, ONNX, TorchScript, TensorFlow Lite, Core ML, OpenVINO, JAX, and Flax exports are not implemented.
- Cross-framework conversion is not implemented.
- Automatic layout is a deterministic layered layout, not an advanced constraint solver.
- Multi-selection copy, edge-preserving subgraph paste, comments, and visual groups are not implemented.
- The safe expression node validates an allowlisted expression language but has no framework exporter in the MVP.

## Recommended next steps

1. Run `uv sync --extra test`, Ruff, and the backend tests in a normal package-network environment.
2. Run `npm install`, ESLint, Vitest, and the Vite production build.
3. Install the optional Keras dependencies and execute the `.keras` binary export test with the MLP and convolutional examples.
4. Add Playwright coverage for drag-and-drop, save and reopen, cycle rejection, property edits, and export controls.
5. Add a revision history and restore interface before expanding persistence behavior.
6. Implement reusable subgraphs only after their serialization and shape-boundary contracts are finalized.
7. Add training as a separate execution domain rather than embedding training state into the architecture graph.
8. Expand framework adapters one verified node family at a time, with executable framework tests for every compatibility claim.
