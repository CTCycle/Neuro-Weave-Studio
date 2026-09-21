# Focused CNN and activation Builder matrix

Date: 2026-09-21
Repository: `CTCycle/Neuro-Weave-Studio`
Evaluation PR: [#15](https://github.com/CTCycle/Neuro-Weave-Studio/pull/15)
Validated head: `8129768d313de33db952aac05f3208625699a9f8`
Base: `develop` at `82dffda5ba5b2a5542f21ee57c62e0ec943fd722`

## Remote evidence

GitHub Actions run [35542542213](https://github.com/CTCycle/Neuro-Weave-Studio/actions/runs/35542542213) completed the focused Playwright job successfully. The run reported `10 passed (4.1m)` for `app/client/e2e/node-functional-matrix.spec.ts`:

- CNN 1D: Conv1D -> Average Pooling 1D -> Global Pooling 1D, Keras/PyTorch.
- CNN 2D: Conv2D -> Max Pooling 2D -> Global Pooling 2D, Keras/PyTorch.
- CNN 3D: Conv3D -> Average Pooling 3D -> Global Pooling 3D, Keras/PyTorch.
- CNN 2D native: Conv2D -> Max Pooling 2D -> Global Pooling 2D, native PyTorch.
- Activation values: `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, and `linear`, Keras/PyTorch.

Each case used the Builder catalog for the target nodes, connected ports through the Builder's keyboard path, observed successful server-side connection validation, reached a ready static validation result, saved a valid graph, submitted Functional Validation, and observed a succeeded worker job. The run recorded no browser `pageerror` or `console.error` events.

Supporting checks in the same run passed: frontend ESLint, 100 Vitest tests, Angular production build, FastAPI startup, Keras/PyTorch discovery, Chromium installation, and native TensorFlow, PyTorch, and JAX parity jobs. The separate Python quality job still failed at the repository's existing Ruff gate; its backend tests passed, and that failure is not attributed to this matrix.

## Evidence boundary

This is a bounded browser/runtime regression matrix, not full catalog certification. It does not validate every built-in node, every parameter option, TensorFlow/JAX Builder execution, export, or training. Native parity jobs are supporting forward-execution evidence only and do not replace the missing Builder evidence for those profiles. The `ui.builder.catalog-coverage` ledger entry therefore remains `UNVALIDATED`.

## Integration

The focused spec is retained as a permanent Playwright regression file. CI runs the complete `npm run e2e` suite again; frontend diagnostics upload runs on every completed frontend job so successful matrix screenshots remain available for inspection. Browser errors are asserted as failures by the integrated spec rather than only logged.
