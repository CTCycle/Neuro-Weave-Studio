# Model Builder end-to-end evaluation

Last updated: 2026-09-03

## Outcome

The Model Builder validation, graph interaction, shape propagation, inspector, catalog, and responsive layout refinements are implemented and pushed to `origin/develop` through these incremental checkpoints:

- `d057da2` — server-authoritative connection validation.
- `f221178` — graph visuals, catalog placement, inspector clarity, and Metrics-before-Environment navigation.
- `6d001d5` — shape diagnostics and propagation regression coverage.
- `08bd883` — transient feedback isolation, compact validation metrics, and repeatable visual captures.
- `3782a8b` — typed diagnostics default factories.

## Significant issues and root causes

- Connection acceptance could diverge between the browser and the actual graph validator. The server now validates every proposed edge against the canonical graph, including port kind, enabled conditions, capacity, duplicates, cycles, and shape compatibility. The browser keeps only enough local filtering to provide connectability feedback.
- Derived client metadata could remain stale after graph edits or asynchronous validation responses. Graph revisions now invalidate derived state, shape summaries are applied only to the revision that requested them, and connection responses are ignored when the graph, backend, or Keras setting has changed.
- Reshape inference accepted empty or element-count-incompatible targets. It now returns a blocking diagnostic and identifies the failing node, parameter, and reason where the source can be determined.
- Bezier paths and Foblex’s default handle markers made connection geometry visually unstable and introduced arbitrary black dots. Persistent and provisional edges now use one stable straight-path treatment, with integrated colored port targets and restrained connectability/error states.
- New drafts could retain a previous transient connection-error banner. Feedback is revision-scoped and reset on graph/validation changes.
- Validation metrics used nested single-purpose cards and the inspector port copy was dense and cryptic. Metrics now use a compact ruled row; inspector connection rows use explicit `INPUT`/`OUTPUT`, required/optional, cardinality, and readable shape descriptions.

## Implemented changes

- Added server-side proposed-edge validation at `/api/validation/connection`.
- Made graph commit, shape-summary propagation, output detection, and validation state revision-aware.
- Added topology-derived effective input/output behavior, including inferred terminal outputs and explicit output bindings.
- Added targeted shape diagnostics for impossible reshapes, merge incompatibility, and upstream shape context.
- Reworked graph edges and connection previews to stable straight paths.
- Removed default black connector handles and replaced them with accessible, larger, direction-aware port targets.
- Improved catalog placement so newly added nodes do not overlap.
- Moved `Metrics` before `Environment` in the primary navigation.
- Simplified inspector connection language and collapsed advanced sections by default.
- Added responsive Catalog and Inspector drawers with verified no-overflow behavior.
- Added regression coverage for topology validation, residual/merge shapes, impossible reshape parameters, and downstream recomputation after Dense unit changes.
- Added repeatable Playwright screenshot capture for clean, valid, and invalid Builder states.

## Browser workflows exercised

The application was run in the in-app browser from clean drafts and saved-model fixtures. The following were exercised and visually inspected:

- Empty Builder state, catalog category grouping/search, and no-selection inspector.
- Inspector selection and parameter editing for input and Dense nodes.
- Valid input → Dense → Dense → classification output graph.
- Fan-out from one input into two branches, followed by a valid merge/add node.
- Invalid input → Conv2D rank mismatch, with rejected edge and specific feedback.
- Invalid merge mutation by changing one Dense branch from 32 to 16 units, followed by restoration to a Ready graph.
- Cycle creation attempt, rejected without adding an edge.
- Edge selection, deletion, undo, node movement, zoom, pan, fit, minimap, canvas lock, focus mode, and tab isolation.
- New-model reset after connection rejection; transient error feedback was confirmed cleared.
- Inspector drawer and Catalog drawer at responsive widths.
- Viewports: default 1280×720, Playwright 1280×800, 768×900, and 390×844. Document and content horizontal overflow were absent at the responsive sizes.

## Graph architectures used

- Linear MLP: `input.tabular → layer.dense → layer.dense → output.classification`.
- Valid residual-style merge: one tabular input fans out to two Dense branches, both connect to `merge.add`, and the merge is the inferred terminal output.
- Invalid residual merge: the same graph with one branch changed to 16 units while the other remains 32; merge shape incompatibility is blocking until restored.
- Invalid convolution connection: rank-2 tabular output proposed to a rank-4 Conv2D input.
- Cycle candidate: downstream Dense output proposed back into an upstream merge input.
- Disconnected fixture: valid branch plus unconnected output/mask-target nodes to verify graph-level reachability diagnostics.

## Visual evidence

These final PNGs were captured by the Playwright workflow and inspected after the run:

- [Empty Builder](screenshots/final-builder-empty.png) — grouped compact catalog, clear empty-canvas start point, no selected node.
- [Valid graph](screenshots/final-builder-valid.png) — clean straight edges, colored ports, Ready state, and compact validation metrics.
- [Validation feedback](screenshots/final-builder-validation.png) — cycle rejection feedback, affected graph state, and validation surface kept below the canvas without flooding it with labels.

## Automated verification

- Backend: `176 passed, 6 skipped` from the complete repository-root pytest run. Six skips are conditional optional-framework coverage.
- Focused backend validation: `20 passed` for validation and connection-validation regressions.
- Backend Ruff on all touched backend and test files: passed.
- Focused Pyright on all touched backend and test files: `0 errors`.
- Frontend Vitest: `6 files, 29 tests passed`.
- Frontend ESLint: passed with zero warnings.
- Angular production build: passed.
- Playwright: `13 passed` in 55 seconds.
- In-app browser smoke check: Builder loaded with meaningful content, no error overlay, no captured browser console errors, and no document overflow.

## Remaining limitations

- Repository-wide Ruff still reports 16 pre-existing findings in unrelated files, primarily import ordering plus one existing `exec` check. Repository-wide Pyright still reports 289 pre-existing findings outside the audited paths; touched paths pass focused Pyright.
- The Playwright web-server log emitted a Vite `ResizeObserver loop completed with undelivered notifications` notice during a responsive interaction. It did not produce a browser console error, error overlay, or test failure, but remains a low-priority runtime warning for a future cleanup pass.
- Six backend tests remain skipped when optional ML frameworks are unavailable in the local environment; this does not affect the pure graph/shape validation coverage.
