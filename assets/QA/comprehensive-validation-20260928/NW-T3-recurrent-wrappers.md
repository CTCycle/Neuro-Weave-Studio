# Bounded recurrent wrapper continuation

Date: 2026-09-28
Run ID: `NW-CV-20260928-T3-RECURRENT-WRAPPERS`
Implementation revision: `38c9ccc` (`test: validate recurrent wrapper catalog slice`)
Branch: `develop`

## Final status

The selected recurrent-wrapper workflow is **PARTIAL**, not a catalog-coverage
PASS. The live Builder/runtime behavior passed on the exact
`execution_backend=pytorch`, `keras_enabled=true`, Keras API / PyTorch backend,
CPU lane: the live catalog remained at 149 definitions with the frozen
fingerprint, all six catalog E2E cases passed, both wrapper graphs validated,
persisted at revision 2, executed successfully, and emitted empty browser-error
evidence. The requested parameter acceptance values (`64` for the
bidirectional LSTM and `68` for the two-layer LSTM stack) are not compatible
with the current LSTM contract or the real Keras worker.

For the fixture input width 3 and units 4, one Keras LSTM has
`4 * (3 * 4 + 4 * 4 + 4) = 128` parameters. The bidirectional wrapper therefore
has `256`, and the two-layer stack has `128 + 144 = 272`. The same values were
returned by static shape inference and the Keras/PyTorch CPU worker. Values
`64` and `68` are the corresponding Simple RNN-style counts, not LSTM counts.
The implementation and documentation already use the standard LSTM gate
multiplier, so the test harness records the verified semantic values instead of
claiming an invalid acceptance PASS.

## Bounded result

| Case | Static shape | Runtime shape | Static/runtime parameters | Persistence and diagnostics | Status |
| --- | --- | --- | ---: | --- | --- |
| Bidirectional LSTM, units 4, `return_sequences=true`, `merge_mode=concat` | `[null, 4, 8]` | `[1, 4, 8]` | `256 / 256` | Revision 2, valid diagnostics `[]`, runtime `succeeded`, browser errors `[]` | PASS for behavior; PARTIAL for requested count criterion |
| Two-layer LSTM stack, units 4, `return_sequences=true` | `[null, 4, 4]` | `[1, 4, 4]` | `272 / 272` | Revision 2, valid diagnostics `[]`, runtime `succeeded`, browser errors `[]` | PASS for behavior; PARTIAL for requested count criterion |
| Bidirectional merge shapes | `sum/mul/ave=[null,4,4]`; `concat=[null,4,8]` | Static focused inference | `256` for each mode | Valid response and empty browser errors | PASS |
| Invalid stack repair | Blocked at `layers=0`; valid after `layers=2` | Repaired graph executed | `272` after repair | Diagnostic captured; save/reopen and runtime succeeded | PASS |
| Neighbor coverage | Existing Simple RNN, GRU, LSTM, and Concatenate cases retained | Existing Keras/PyTorch CPU cases | Existing contract values | Six-case serial catalog spec passed | PASS within named slice |

The live catalog count and fingerprint were rechecked against
`evidence/catalog-runtime-freeze.json`; the runtime identity was rechecked
against `evidence/runtime-readiness.json`. The official launcher reached the
frontend and backend readiness gates before the browser run.

## Repair and visual evidence

The invalid stack path set `layers=0`, observed Builder `Blocked`, opened the
diagnostic, and captured `Recurrent stack layers must be positive`. It then set
`layers=2`, saved/reopened the graph at revision 2, revalidated, and completed
functional execution on CPU.

- [Bidirectional merge-mode payload](evidence/nw-t3-wrapper-bidirectional-merge-modes.json)
- [Bidirectional merge-mode screenshot](evidence/nw-t3-wrapper-bidirectional-merge-modes.png)
- [Bidirectional LSTM payload](evidence/nw-t3-wrapper-bidirectional-lstm.json)
- [Bidirectional Builder-ready screenshot](evidence/nw-t3-wrapper-bidirectional-lstm-builder-ready.png)
- [Bidirectional runtime screenshot](evidence/nw-t3-wrapper-bidirectional-lstm-runtime-succeeded.png)
- [Bidirectional browser errors](evidence/nw-t3-wrapper-bidirectional-lstm-browser-errors.json)
- [Invalid stack payload](evidence/nw-t3-wrapper-stack-invalid.json)
- [Invalid stack screenshot](evidence/nw-t3-wrapper-stack-invalid.png)
- [Stack payload](evidence/nw-t3-wrapper-stack-lstm.json)
- [Stack Builder-ready screenshot](evidence/nw-t3-wrapper-stack-lstm-builder-ready.png)
- [Stack runtime screenshot](evidence/nw-t3-wrapper-stack-lstm-runtime-succeeded.png)
- [Stack browser errors](evidence/nw-t3-wrapper-stack-browser-errors.json)
- [In-app Browser observation](evidence/nw-t3-wrapper-inapp-browser-observation.md)

The authoritative browser command was:

```text
PLAYWRIGHT_BASE_URL=http://127.0.0.1:6429 PLAYWRIGHT_WORKERS=1 npm run e2e -- e2e/catalog-risk-slice.spec.ts --workers=1 --reporter=list
```

It completed `6 passed` in approximately `1.3m`. The final run is recorded in
[catalog-risk-slice-recurrent-wrappers-final.log](logs/catalog-risk-slice-recurrent-wrappers-final.log).

## Quality gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Focused backend validation/export/native parity/catalog/parameter tests | PASS | [backend-recurrent-wrappers-current.log](logs/backend-recurrent-wrappers-current.log) |
| Changed-file Ruff | PASS | [ruff-recurrent-wrappers-current.log](logs/ruff-recurrent-wrappers-current.log) |
| Changed catalog E2E lint | PASS | [catalog-risk-slice-eslint-current.log](logs/catalog-risk-slice-eslint-current.log) |
| Frontend unit tests | PASS, 20 files / 105 tests | [frontend-unit-current-nw-t3-recurrent-wrappers.log](logs/frontend-unit-current-nw-t3-recurrent-wrappers.log) |
| Frontend lint | PASS, zero warnings | [frontend-lint-current-nw-t3-recurrent-wrappers.log](logs/frontend-lint-current-nw-t3-recurrent-wrappers.log) |
| Production build | PASS | [frontend-build-current-nw-t3-recurrent-wrappers.log](logs/frontend-build-current-nw-t3-recurrent-wrappers.log) |
| Full strict Pyright | PARTIAL, 58 errors / 0 warnings / 0 informations | [pyright-current-nw-t3-recurrent-wrappers.log](logs/pyright-current-nw-t3-recurrent-wrappers.log) |

## L0-L6 boundary

| Level | Final status | Evidence and limitation |
| --- | --- | --- |
| L0 catalog inventory | PASS | 149-node catalog and frozen fingerprint match; this does not close node-level coverage. |
| L1 single-node behavior | PASS for the named recurrent wrappers and retained neighbors | The remaining catalog definitions and profile-specific enable/disable behavior remain open. |
| L2 parameters | PARTIAL | Shapes, merge modes, layer repair, and standard LSTM parameter counts pass; the plan's `64/68` LSTM count criterion remains unresolved because it conflicts with the declared Keras semantics. |
| L3 ports and edges | PASS for the selected sequence input/wrapper/output graphs | Exhaustive catalog edge generation and incompatible-pair coverage remain open. |
| L4 graph composition | PASS for the selected bidirectional and stack graphs | This is not full graph-family coverage. |
| L5 runtime/profile | PASS for the selected Keras/PyTorch CPU lane | The other five live profiles, source/export closure, and positive PyTorch Inductor path remain unpromoted. |
| L6 persistence/recovery | PASS for the invalid-stack repair, save/reopen, validation, runtime, and cleanup path | Matrix-wide recovery and failure cleanup remain open. |

Formal basic-convolution `NW-T3-03` is explicitly **UNTESTED**; this run
selected the prepared recurrent-wrapper continuation. Training, dataset
materialization, assistive technology, all-six-profile runtime/export closure,
the remaining live profiles, and full catalog coverage were not promoted.
