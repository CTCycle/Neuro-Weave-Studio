# Backend Compatibility Consolidation

Date: 2026-09-13
Scope: six static execution profiles, verified-only node and parameter mappings, repair-safe graph editing, compilation/export gates, and runtime job validation.

## Consolidated policy

The selected profile is one of:

| Profile | Mapping key | API identity |
| --- | --- | --- |
| Keras over TensorFlow | `keras` plus TensorFlow eligibility | `keras` / `tensorflow` |
| Keras over PyTorch | `keras` plus PyTorch eligibility | `keras` / `pytorch` |
| Keras over JAX | `keras` plus JAX eligibility | `keras` / `jax` |
| Native TensorFlow | `tensorflow` | `native` / `tensorflow` |
| Native PyTorch | `pytorch` | `native` / `pytorch` |
| Native JAX | `jax` | `native` / `jax` |

A node or parameter is available for authoring only when the exact selected mapping is present, supported, and `status: verified`. A missing or absent parameter capability inherits the verified node mapping; an explicit parameter capability can further restrict support, values, constraints, defaults, or editability. Keras backend exclusions are a separate compatibility reason. Runtime and compiler availability remain separate gates.

Incompatible catalog entries stay discoverable with a stable reason code and are disabled for add, drag/drop, paste, duplication, materialization, output creation, and new connections. Existing incompatible nodes, edges, and stored values remain in the editor for repair. Existing nodes remain selectable and deletable; their parameters remain visible and disabled, while malformed stored values can be edited back to a valid value. Profile switching revalidates without pruning or normalizing the graph. Explicit reset applies the selected profile's effective defaults.

## Static compatibility matrix

The server matrix covers all six profiles (`tests/test_backend_compatibility_consolidation.py`, 50 tests in the dedicated matrix). The frontend resolver/store/materializer suites cover strict verified-only availability, atomic mutations, invalid-value retention, reset, profile switching, and custom-template preflight.

| Check | Keras/TF | Keras/PT | Keras/JAX | Native TF | Native PT | Native JAX |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Common Dense chain | pass | pass | pass | pass | pass | pass |
| `recurrent.simple_rnn` | pass | pass | pass | blocked | blocked | blocked |
| Dense `kernel_initializer=he_normal` | pass | pass | pass | blocked | blocked | blocked |
| Dense default initializer `glorot_uniform` | pass | pass | pass | pass | pass | pass |
| LSTM `unit_forget_bias=false` | pass | pass | pass | blocked | pass | blocked |
| Keras backend exclusion reason | enforced | enforced | enforced | n/a | n/a | n/a |
| Synthetic unverified node mapping | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Synthetic unverified parameter mapping | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

Stable compatibility reason codes include `missing_implementation`, `unsupported_implementation`, `unverified_implementation`, `backend_excluded`, `unsupported_parameter`, `unverified_parameter`, `unsupported_parameter_option`, and `invalid_parameter_value`.

## Validation and mutation gates

- Canonical client resolution: `app/client/src/app/domain/types.ts`.
- Canonical server resolution: `app/server/validation/profile_compatibility.py`.
- Server `ValidationService` and `ConnectionValidationService` use the same selected mapping rule and expose `reason_code` details.
- Direct store paths are guarded atomically: add, update, duplicate-with-defaults, duplicate-exact, paste, connection commit, and profile-aware custom materialization.
- Existing invalid values are retained; supported fields on an otherwise-invalid node can still be repaired; rejected updates leave graph state, history, and revision unchanged.
- Compilation/source generation and execution jobs validate the complete graph/profile first. Native source generation has no Keras fallback.
- Export capability responses keep invalid graphs and unavailable runtime/compiler capabilities visible with explicit reasons; job submission rejects invalid graphs, mismatched worker backend/API, unavailable formats, and invalid export options before starting a worker.

## Automated validation

- Frontend: `npm run test` — 14 files, 78 tests passed.
- Frontend lint: `npm exec -- eslint . --max-warnings 0 --no-cache` — passed.
- Frontend production build: `npm exec -- ng build --verbose` — passed.
- Server compatibility/validation/connection/compiler/catalog subset — 76 tests passed.
- Server API export-boundary checks — 2 tests passed, including no job creation for worker-profile mismatch and no artifact publication after invalid-graph rejection.
- Narrowed Ruff on changed server files/tests — passed.
- Narrowed Pyright on changed server files/tests — 0 errors, 0 warnings.
- Browser workflows: `npm exec -- playwright test e2e/workflows.spec.ts --workers=1` — 14 passed. The two new compatibility workflows also pass in their focused parallel run.

## Browser and correlated evidence

Permanent Playwright evidence is stored beside this report:

- [profile-incompatible-builder.png](backend-compatibility-consolidation/profile-incompatible-builder.png) — native PyTorch profile retains Simple RNN, marks it repair-required, disables the catalog entry, ports, and parameter controls.
- [invalid-compilation.png](backend-compatibility-consolidation/invalid-compilation.png) — compile action is disabled and validation reports the native PyTorch Simple RNN incompatibility.
- [invalid-export-capabilities.png](backend-compatibility-consolidation/invalid-export-capabilities.png) — project/source/native export actions remain visible with explicit invalid-profile/runtime reasons and cannot run.
- [smoke-job-succeeded.png](backend-compatibility-consolidation/smoke-job-succeeded.png) — compatible Keras/PyTorch smoke-test job reaches a verified result.

The manual CUA pass used the official launcher (backend `9750`, preview `6425`) and correlated browser/API state:

1. A saved draft switched from Keras to native PyTorch without pruning the retained `recurrent.simple_rnn` node. The persisted response was revision 2, `backend=pytorch`, `keras_enabled=false`, `validation_status=invalid`, with all three original node IDs still present.
2. The compilation surface reported `Simple RNN is not supported by native pytorch` and left source compilation disabled.
3. The export surface showed disabled Project bundle, Generated source, Keras, PyTorch, ONNX, and checkpoint actions with profile/runtime explanations.
4. A compatible Keras/PyTorch smoke job progressed `starting` → `running` → `succeeded` on `cuda:0` / NVIDIA GeForce RTX 3060 Laptop GPU, reporting 1 input, 1 output, 161 parameters, and 4120.8 ms. The permanent Playwright workflow independently polls the submitted job to `succeeded`.
5. Backend API logs for the pass are in `app/resources/runtime/logs/backend.log`; the correlated requests include `/api/validation`, `/api/exports/.../capabilities`, `/api/execution/jobs`, and repeated `/api/execution/jobs/{id}` polling.

## Runtime and compiler status

- VERIFIED live: PyTorch/Keras inference on the RTX 3060 smoke-test path.
- VERIFIED static: all six profile mapping/parameter/connection/compiler/export-boundary contracts listed above.
- UNVERIFIED: live TensorFlow execution/export and live JAX execution/export were not run in this pass. Their static profiles, mapping diagnostics, and compiler capability surfaces remain explicit; no native TensorFlow/JAX optimizer, training, or exporter parity is claimed.
- RUNTIME-DEPENDENT: optional ONNX/checkpoint availability remains dependent on the runtime discovery/dependency response and is displayed with its returned reason.
- SKIPPED as a gate: the broader live API artifact test's fixed 10-second discovery helper expired while PyTorch/Keras discovery took about 13 seconds; the backend log subsequently recorded the probe as available. The focused invalid-graph/no-artifact and worker-profile mismatch tests pass independently, and the browser smoke job provides live execution proof.

## Documentation alignment

The compatibility, manifest, editor-interaction, execution-profile, compiler, export, error-pattern, and testing documents were updated on 2026-09-13 to use the same six-profile, verified-only, repair-retaining, no-fallback terminology.
