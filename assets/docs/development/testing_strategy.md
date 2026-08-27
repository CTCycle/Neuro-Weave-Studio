# Testing Strategy

Last updated: 2026-08-27

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
- binary Keras, native, ONNX, SafeTensors, and checkpoint export reload/restore coverage where the optional dependency is operational;
- runtime preflight validation, device lease exclusivity, idempotent cancellation, and worker/workspace cleanup.

Real framework availability is conditional. Runtime tests skip when the required execution backend is not installed and operational. Controller lifecycle tests use deterministic fake worker processes so lease, cancellation, and cleanup behavior remains testable without optional ML frameworks.

## Frontend

Frontend tests are deliberately limited to state and interaction contracts that can corrupt or block model editing:

- graph cycle prevention plus preservation of graph metadata and undo/redo history;
- isolated multi-tab editor sessions;
- description metadata editing, capability-driven compilation configuration, backend/Keras switching, and minimap/focus-mode state contracts;
- the Node Catalog drag payload consumed by the model canvas;
- Model Export center capability grouping, explicit disabled reasons, format-specific options, job polling, cancellation, retry, and verified download.

Presentation details, button inventories, clipboard status, basic filtering, and collapsible UI behavior are covered by static checks and product QA rather than dedicated unit tests.

## Verification

Backend changes should run pytest, Ruff, and Pyright. Frontend changes should run Vitest, ESLint, and the TypeScript/Vite production build.

Coverage percentage is not a release goal. Regression signal, business-risk protection, execution safety, and maintainability are the criteria for retaining tests.
