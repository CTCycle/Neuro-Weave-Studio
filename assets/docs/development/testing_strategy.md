# Testing Strategy

Last updated: 2026-08-31

NeuroWeave's automated suite is intentionally risk-oriented. Tests exist for behavior whose failure can corrupt saved models, produce invalid generated models, violate execution safety, or break an important user workflow. Tooling layout, source-tree shape, presentation details, and framework behavior are not unit-tested merely to increase coverage.

## Backend

Retained regression coverage focuses on:

- graph validation plus shape and parameter inference for DAGs, unsafe expressions, merge, convolution, attention, and unsupported implementation contracts;
- project revision workflows, stale-save rejection, generated-source invalidation, project import/export, and transactional rollback;
- SQLite migration locking and rollback, unsupported database rejection, foreign-key aggregate integrity, and cascades;
- canonical model-configuration round trips, v2-to-v3 migration/downgrade behavior, capability metadata, and explicit invalid native compiler diagnostics;
- safe Keras architecture import through the explicit layer allowlist;
- generated Keras and native TensorFlow/PyTorch/JAX source with backend-specific compiler settings for representative graphs;
- native TensorFlow/PyTorch/JAX parity for deterministic common-subset graphs, including representative multi-input/multi-output cases;
- capability discovery, clean-break `model_export` submission, format option validation, bridge availability, artifact manifests, reload/inference verification, stale revisions, cancellation, conversion failure, and atomic publication;
- runtime discovery skips absent framework packages, records a fingerprinted environment snapshot, reuses matching snapshots, and invalidates them when package state changes;
- binary Keras, native, ONNX, SafeTensors, and checkpoint export reload/restore coverage where the optional dependency is operational;
- runtime preflight validation, device lease exclusivity, idempotent cancellation, and worker/workspace cleanup.

Real framework availability is conditional. Runtime tests skip when the required execution backend is not installed and operational. Controller lifecycle tests use deterministic fake worker processes so lease, cancellation, and cleanup behavior remains testable without optional ML frameworks.

## Frontend

Frontend tests are deliberately limited to state and interaction contracts that can corrupt or block model editing:

- graph cycle prevention plus preservation of graph metadata and undo/redo history;
- isolated multi-tab editor sessions;
- description metadata editing, capability-driven compilation configuration, backend/Keras switching, and minimap/focus-mode state contracts;
- the Node Catalog drag payload consumed by the model canvas;
- Model Export page capability grouping, explicit disabled reasons, format-specific options, job polling, cancellation, retry, and verified download;
- Builder keeps graph save/validation separate from Compilation configuration and source generation, while CodePreview remains inspection/copy-only there.

Presentation details, button inventories, clipboard status, basic filtering, and collapsible UI behavior are covered by static checks and product QA rather than dedicated unit tests.

## Verification

Backend changes should run pytest, Ruff, and Pyright. Frontend changes should run Vitest, ESLint, the Angular production build, and Playwright E2E. Vitest includes only `src/**/*.test.ts` and `src/**/*.spec.ts`; Playwright specs live under `e2e/` and are run through the dedicated `npm run e2e` command.

Coverage percentage is not a release goal. Regression signal, business-risk protection, execution safety, and maintainability are the criteria for retaining tests.

The native compilation parity gate covers PyTorch and Keras-on-PyTorch representative dense, classification, recurrent, LSTM, and convolution graphs. It verifies catalog mappings, generated helper construction/evaluation, task/target/activation diagnostics, compilation summaries, forward shapes and dtypes, and supported native export reloads. Runtime evidence must state that helper verification is inference-only; TensorFlow and JAX validation remain out of scope for this gate. Browser acceptance uses the official launcher and records graph creation, validation, compilation, inference, export capability results, logs, network/console failures, and cleanup under `assets/QA/`.
