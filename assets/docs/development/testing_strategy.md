# Testing Strategy

Last updated: 2026-09-10

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
- runtime discovery is on demand per backend/API pair, skips absent framework packages, records a schema-versioned fingerprinted environment snapshot, deduplicates concurrent requests, bypasses the snapshot for explicit refresh, and invalidates matching results when package or environment identity changes;
- binary Keras, native, ONNX, SafeTensors, and checkpoint export reload/restore coverage where the optional dependency is operational;
- runtime preflight validation, device lease exclusivity, idempotent cancellation, and worker/workspace cleanup.
- dataset metadata round trips, CRUD/restart persistence, no-content-copy guarantees, streamed CSV/TSV inspection, Parquet metadata, NPY/NPZ headers, image/text enumeration, exact composed mappings, role conflicts, subset bounds, and unavailable external sources.

Real framework availability is conditional. Runtime tests skip when the required execution backend is not installed and operational. Controller lifecycle tests use deterministic fake worker processes so lease, cancellation, and cleanup behavior remains testable without optional ML frameworks.

## Frontend

Frontend tests are deliberately limited to state and interaction contracts that can corrupt or block model editing:

- graph cycle prevention plus preservation of graph metadata and undo/redo history;
- isolated multi-tab editor sessions;
- description metadata editing, capability-driven compilation configuration, backend/Keras switching, and minimap/focus-mode state contracts;
- the Node Catalog drag payload consumed by the model canvas;
- Model Export page capability grouping, explicit disabled reasons, format-specific options, job polling, cancellation, retry, and verified download;
- Builder keeps graph save/validation separate from Compilation configuration and source generation, while CodePreview remains inspection/copy-only there.
- dataset capability choices, disabled Auto Detect, persisted resource refresh, wizard role/subset/validation transitions, catalog selection/deletion, unavailable-source rendering, and future-only Processing/Split routes.
- browser file-input import of a `.keras` architecture, including persisted project retrieval and downstream route navigation;
- exact runtime device/memory-policy local-storage round trips after a browser reload, using native select change events;
- keyboard graph traversal, source/target port activation, authoritative connection validation, and accessible status feedback;
- visible-control naming, document landmarks, language/title, and focus-visible baseline checks across the core browser routes.

Presentation details, button inventories, clipboard status, basic filtering, and collapsible UI behavior are covered by static checks and product QA rather than dedicated unit tests.

## Verification

Backend changes should run pytest, Ruff, and Pyright. Frontend changes should run Vitest, ESLint, the Angular production build, and Playwright E2E. Vitest includes only `src/**/*.test.ts` and `src/**/*.spec.ts`; Playwright specs live under `e2e/` and are run through the dedicated `npm run e2e` command.

Coverage percentage is not a release goal. Regression signal, business-risk protection, execution safety, and maintainability are the criteria for retaining tests.

The repository reference path has two tiers. Hermetic tests stub the Hugging Face resolver and retain explicit outage and malformed-commit cases. The live backend round trip is opt-in with `NEURO_WEAVE_RUN_LIVE_REPOSITORY_TESTS=1`; the live Nodes UI round trip is opt-in with `NEURO_WEAVE_RUN_LIVE_REPOSITORY_E2E=1`. Both verify a public commit is pinned and that the definition remains reference-only. They are intentionally excluded from the offline default suite because external availability is not a deterministic CI dependency.

The browser file-input test exercises Playwright's browser input path; it does not certify every native file chooser implementation. The accessibility baseline is not a substitute for full WCAG conformance or a screen-reader/assistive-technology matrix. When those environments are available, record the manual result under `assets/QA/` with browser, OS, assistive technology, and observed interaction details.

The native compilation parity gate covers PyTorch and Keras-on-PyTorch representative dense, classification, recurrent, LSTM, and convolution graphs. It verifies catalog mappings, generated helper construction/evaluation, task/target/activation diagnostics, compilation summaries, forward shapes and dtypes, and supported native export reloads. Runtime evidence must state that helper verification is inference-only; TensorFlow and JAX validation remain out of scope for this gate. Browser acceptance uses the official launcher and records graph creation, validation, compilation, inference, export capability results, logs, network/console failures, and cleanup under `assets/QA/`.
