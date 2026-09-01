# Model Builder final audit

Last updated: 2026-09-01

## Scope

This audit covers the Model Builder graph contract, node catalog, parameter inspector, backend capability filtering, DAG validation, shape propagation, output selection, source/runtime adapters, persistence, and editor interactions. Output-category nodes remain available as optional semantic/task adapters, as requested; they are not required for a graph to have a valid output.

## Canonical contracts

- Graph schema is version 4. Portable project bundles are version 5.
- Nodes persist IDs, definition identity, position, serializable parameters, and `locked` state.
- Edges persist source and target node/port IDs. The graph compiler requires explicit input bindings and preserves port identity for multi-output nodes.
- `output_bindings` stores explicit node-port selections. If it is empty, unconsumed value-producing output ports on non-input nodes are inferred. Explicit selections take precedence.
- Runtime graph readers require the current schema. Database upgrades are forward Alembic migrations; runtime `migrate_legacy_payload` readers, aliases, and legacy implementation fallbacks are not used.

## Catalog and capability audit

The normalized catalog contains 145 manifests across inputs, core, convolutional, recurrent, transformer, merge, mathematical, and outputs categories. Every manifest round-trips through its Pydantic contract, has unique port/parameter IDs, and every supported implementation mapping declares a symbol or adapter.

| Target | Catalog mappings | Supported mappings |
| --- | ---: | ---: |
| Keras API | 145 | 139 |
| Native TensorFlow | 145 | 107 |
| Native PyTorch | 145 | 107 |
| Native JAX / Flax | 145 | 107 |

The inspector receives one unified schema and groups parameters by purpose. It supports typed scalar values, booleans, enums, activation and dtype choices, shapes/tuples/lists, nullable values, structured JSON-safe initializers/regularizers/constraints, conditional visibility/enabling, and target-specific availability. Unsupported mappings remain discoverable and are disabled with an explicit capability message; they do not silently disappear or become identity adapters.

The Keras layer surface includes the applicable constructor controls for Dense, convolution, pooling, dropout, embedding, normalization, recurrent, attention, and transformer nodes, including activation, dtype, initialization, regularization, constraints, trainability, optional state, and advanced controls. Backend capability defaults are used for reset/default duplication when declared.

## Graph and shape validation

The validator reports node/parameter/edge-specific diagnostics for unknown definitions and parameters, duplicate IDs, dangling references, missing required inputs, disabled ports, invalid port kinds, duplicate occupancy, self-loops, cycles, unreachable graph components, invalid explicit outputs, unsupported/unverified implementations, invalid parameter values and combinations, unsafe expressions, and incompatible shapes. Errors block execution; target compatibility diagnostics also block the selected execution profile; warnings and informational model-type inference remain non-binding.

Shape propagation covers graph branches and merges, recurrent sequence/state outputs, attention and transformer paths, dynamic/unknown dimensions, and parameter counts where dimensions are concrete. Matrix multiplication is explicitly batch-first and requires rank-3-or-higher tensors, matching the emitted TensorFlow, PyTorch, and JAX operation semantics.

## Output and editor behavior

- Single terminal workflows receive inferred outputs without an artificial output node.
- Multiple terminal nodes and multiple output ports are represented independently and preserve stable names/selectors.
- Output adapters remain valid and provide optional task metadata for classification, regression, segmentation, sequence, and multi-head workflows.
- Duplicate Exact copies every serialized parameter and receives a new ID; Duplicate with Defaults uses the active implementation defaults; neither copies connections.
- Locked nodes cannot move but remain selectable, configurable, connectable, duplicable, and deletable.
- Empty-canvas context menu: Fit workflow to view, Auto layout, Reset canvas.
- Node context menu: Duplicate with defaults, Duplicate exact, Copy node, Reset parameters, Disconnect all, output toggle, lock toggle, and Delete node.
- Reset canvas always opens a confirmation dialog explicitly stating that all nodes and connections will be removed.

## Verification evidence

The following checks passed during this audit:

- Backend focused suite: `app/server/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp='.pytest-tmp-model-builder-final4' app/tests/test_node_catalog_audit.py app/tests/test_validation.py app/tests/test_core_ml_audit.py app/tests/test_conv1d_and_operations.py`.
- Full backend suite: `app/server/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp='.pytest-tmp-model-builder-full2' ..\\tests` passed; the only reported cases were the repository's existing runtime-dependent skips.
- Frontend unit suite: 3 files, 16 tests passed with `npm run test -- --run`.
- Frontend production build passed with `npm run build`.
- Frontend lint passed with `npm run lint`.
- Touched backend modules passed Ruff with `.venv/Scripts/python.exe -m ruff check common domain compilation exporters models services validation`.
- Representative native matrix multiplication passed for TensorFlow, PyTorch, and JAX in `test_conv1d_and_operations.py`.
- Direct generated Keras catalog audit built 132 supported non-input/non-classification-output manifests under `KERAS_BACKEND=torch`: 132 passed, 0 failed.
- Catalog introspection confirmed zero supported mappings without a symbol or adapter and zero legacy implementation adapter references.
- Browser smoke verification at `http://127.0.0.1:6425/builder` confirmed the populated catalog, grouped inspector, exact duplication, lock/output indicators, node and canvas menus, reset confirmation/cancel/confirm behavior, and disabled native-backend-only parameters when Keras was disabled. Captures are retained in `assets/QA/model-builder-browser-backend-capability.png` and `assets/QA/model-builder-browser-reset-confirmation.png`.

## Intentional limits

Native support is catalog-scoped and currently excludes advanced convolutional/recurrent variants, attention, and transformer adapters. Keras remains the broad graph implementation path. Arbitrary Python callables, custom framework objects, learned weights, and arbitrary code are excluded from the portable graph contract. These limitations are surfaced as capability diagnostics rather than hidden fallbacks.

The repository-wide Pyright baseline still contains unrelated pre-existing typing findings in runtime, Pydantic, and test modules; the production gate for this batch is the successful strict Angular build, frontend lint/tests, focused backend tests, and Ruff check above.
