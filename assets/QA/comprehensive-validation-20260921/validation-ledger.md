# NeuroWeave Comprehensive Validation Ledger

Last updated: 2026-09-21

Run ID: `NW-CV-20260921-T0-01`

Repository: `develop` at `752ab151978b56dd2c20568f63434e346dcf6ea8`

Campaign status: `PARTIAL` — Tier 0 was executed. The campaign is not a full
application certification; all later tiers remain explicit validation work.

## Status summary

| Tier | Scope | PASS | PARTIAL | FAIL | BLOCKED | UNTESTED | UNKNOWN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 0 | Environment and startup | 3 | 1 | 0 | 0 | 0 | 0 |
| Tier 1 | Application foundations | 0 | 0 | 0 | 0 | 5 | 0 |
| Tier 2 | Core workflows | 0 | 0 | 0 | 0 | 9 | 0 |
| Tier 3 | Model Builder catalog | 0 | 0 | 0 | 0 | 19 | 0 |
| Tier 4 | Runtime and integrations | 0 | 0 | 0 | 0 | 4 | 0 |
| Tier 5 | Resilience, usability, closure | 0 | 0 | 0 | 0 | 5 | 0 |
| **Total** | **46 slices** | **3** | **1** | **0** | **0** | **42** | **0** |

`UNTESTED` means that the slice is in campaign scope but was not run in this
revision-specific pass. It does not mean that the feature is absent or broken.

## Ledger schema

Each row keeps these decisions separate: feature existence, whether the exact
scenario was exercised, and the resulting status. The full recommended record
also includes run ID, capability, revision, environment, scenarios, profiles,
issues, failure class, fixes, adjacent regression, evidence, remaining gaps,
timestamp, and catalog fingerprint where the slice is catalog-related.

## Slice ledger

| Slice ID | Capability | Status | Feature exists | Exercised | Revision | Failure class | Evidence | Remaining gap / next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NW-T0-01` | Freeze validation baseline and live catalog | PASS | Yes | Yes | `752ab151978b56dd2c20568f63434e346dcf6ea8` | test/environment | [run manifest](run-manifest.json), [baseline snapshot](t0-baseline-snapshot.json), [catalog](catalog/live-catalog.json), [catalog hash](catalog/catalog-sha256.txt) | Repeat whenever branch, dependencies, catalog, or runtime identity changes. |
| `NW-T0-02` | Official launcher startup and owned-process shutdown | PASS | Yes | Yes | `752ab151978b56dd2c20568f63434e346dcf6ea8` | configuration / test-environment | [launcher log](logs/t0-02-launch.log), [post-launch process tree](logs/t0-02-processes-after-launch.txt), [pre-shutdown process tree](logs/t0-02-processes-before-shutdown.txt), [shutdown port check](logs/t0-02-listeners-after-shutdown.txt), [backend log](logs/backend.log) | Repeat after launcher, port, dependency, runtime, or startup-log changes. |
| `NW-T0-03` | Repository quality-gate baseline | PARTIAL | Yes | Yes | `752ab151978b56dd2c20568f63434e346dcf6ea8` | code-quality / test-environment / performance | [quality results](t0-quality-results.json), [backend tests](logs/t0-03-backend-pytest.log), [exact-test rerun](logs/t0-03-backend-failing-test-rerun.log), [Ruff](logs/t0-03-ruff.log), [Pyright](logs/t0-03-pyright.log), [frontend logs](logs/t0-03-frontend-lint.log) | Ruff and Pyright remain nonzero; the full suite had one discovery polling timeout, while the exact test passed in an isolated rerun. |
| `NW-T0-04` | Direct deep links and refresh | PASS | Yes | Yes | `752ab151978b56dd2c20568f63434e346dcf6ea8` | — | [full shell suite](logs/t0-04-shell-e2e.log), [deep-link repetition](logs/t0-04-shell-route-repeat3.log), [route evidence](t0-route-evidence.json) | Preserve route coverage and add richer page/console/request capture if the shell test contract changes. |
| `NW-T1-01` | Shell navigation and route boundaries | UNTESTED | Not assessed in this run | No | — | — | — | Run route transitions, wildcard/missing-model routes, anchors, Back/Forward, and nested dataset/model routes. |
| `NW-T1-02` | Database initialization, migrations, and integrity | UNTESTED | Not assessed in this run | No | — | — | — | Run clean/current/older upgrades, concurrency, rollback, schema rejection, foreign keys, and restart using disposable databases. |
| `NW-T1-03` | Project lifecycle, revisions, and canonical bundle | UNTESTED | Not assessed in this run | No | — | — | — | Exercise create/open/edit/save/duplicate/delete/import/export, stale writes, restart, and Builder reopening. |
| `NW-T1-04` | Runtime preference persistence | UNTESTED | Not assessed in this run | No | — | — | — | Validate backend/API/device/memory selections, malformed storage, unavailable devices, reload, and storage failure. |
| `NW-T1-05` | Settings and Training boundaries | UNTESTED | Boundary documented | No | — | — | — | Confirm no editable global settings and the intentional session-only/disabled Training behavior. |
| `NW-T2-01` | Builder editing, history, and persistence | UNTESTED | Not assessed in this run | No | — | — | — | Cover editor mechanics, dirty/repairable invalid drafts, save/reopen, and history. |
| `NW-T2-02` | Connection, shape, and invalid-state recovery | UNTESTED | Not assessed in this run | No | — | — | — | Cover compatible/incompatible ports, cycles, shape propagation, and repair without silent pruning. |
| `NW-T2-03` | Compilation configuration and generated source | UNTESTED | Not assessed in this run | No | — | — | — | Verify exact profile/source identity, freshness, and unsupported compiler diagnostics. |
| `NW-T2-04` | Functional Validation worker lifecycle | UNTESTED | Not assessed in this run | No | — | — | — | Verify submit/poll/cancel/reload/cleanup and exact worker profile without fallback. |
| `NW-T2-05` | Tensor inspection | UNTESTED | Implemented; not exercised here | No | — | — | — | Run Builder/overview entry points, traces, statistics, slices, expiry, cancellation, and fail-closed checks. |
| `NW-T2-06` | Dataset registration and workspace | UNTESTED | Not assessed in this run | No | — | — | — | Cover source registration, bounded inspection, roles, persistence, relink, and deletion. |
| `NW-T2-07` | Dataset processing and split configuration | UNTESTED | Declarative boundary exists | No | — | — | — | Validate persisted processing/split metadata without treating unimplemented materialization as a defect. |
| `NW-T2-08` | Custom metric and loss lifecycle | UNTESTED | Implemented; not exercised here | No | — | — | — | Cover create/edit/validation/save/reload/task compatibility/delete. |
| `NW-T2-09` | Safe custom-node authoring | UNTESTED | Implemented; not exercised here | No | — | — | — | Cover mathematical and graph-template materialization, validation, persistence, and reference-only repositories. |
| `NW-T3-01` | Inputs and basic shape/core nodes | UNTESTED | Catalog slice in scope | No | — | — | — | Close L0-L6 evidence for the live family and supported profiles. |
| `NW-T3-02` | Dense, embedding, dropout, normalization | UNTESTED | Catalog slice in scope | No | — | — | — | Close parameters, ports, Builder, static, persistence, source, and runtime evidence. |
| `NW-T3-03` | Basic convolution and pooling | UNTESTED | Catalog slice in scope | No | — | — | — | Extend current CNN evidence beyond the named canary cases. |
| `NW-T3-04` | Advanced and spatial convolution | UNTESTED | Catalog slice in scope | No | — | — | — | Validate uncommon ranks, spatial parameters, and profile-specific capability. |
| `NW-T3-05` | Recurrent nodes | UNTESTED | Catalog slice in scope | No | — | — | — | Use bounded recurrent graphs and immediate adjacent regressions. |
| `NW-T3-06` | Attention primitives | UNTESTED | Catalog slice in scope | No | — | — | — | Validate masks, heads, sequence shapes, and unsupported combinations. |
| `NW-T3-07` | Transformer blocks | UNTESTED | Catalog slice in scope | No | — | — | — | Validate block composition, parameters, shapes, and supported profiles. |
| `NW-T3-08` | Arithmetic merge and fan-in | UNTESTED | Catalog slice in scope | No | — | — | — | Cover fan-in cardinality, broadcasting/shape diagnostics, and execution. |
| `NW-T3-09` | Structural merge and branching | UNTESTED | Catalog slice in scope | No | — | — | — | Cover concatenate/stack/branch/fan-out and invalid edge recovery. |
| `NW-T3-10` | Unary mathematical operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-11` | Rounding and trigonometric operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-12` | Hyperbolic and activation operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-13` | Comparison and logical operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-14` | Reduction and aggregate operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-15` | Arithmetic, dot, and matrix operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-16` | Shape, cast, repeat, stack, transpose, and tile operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster. |
| `NW-T3-17` | Index, slice, expression, and stop-gradient operations | UNTESTED | Catalog slice in scope | No | — | — | — | Validate bounded operation group and reconcile live roster; missing live definitions are debt. |
| `NW-T3-18` | Output and task contracts | UNTESTED | Catalog slice in scope | No | — | — | — | Cover inferred/explicit outputs, task compatibility, multi-head outputs, and unsupported mappings. |
| `NW-T3-19` | Generated ports and edge coverage | UNTESTED | Catalog slice in scope | No | — | — | — | Generate bounded compatible/incompatible candidates from the frozen live catalog. |
| `NW-T4-01` | Fresh six-profile live matrix | UNTESTED | Runtime lanes declared | No | — | — | — | Run each Keras/native TensorFlow/PyTorch/JAX profile independently through Builder, save, source, worker, and reload. |
| `NW-T4-02` | Native compiler capability gates | UNTESTED | Capability gates declared | No | — | — | — | Validate XLA, JIT, and `torch.compile`; unavailable toolchains must remain explicit and fail closed. |
| `NW-T4-03` | Export artifact matrix | UNTESTED | Export paths declared | No | — | — | — | Verify capability gating, artifact identity, hashes, structural checks, reload/forward checks, and browser download. |
| `NW-T4-04` | Live Hugging Face repository reference | UNTESTED | Opt-in path exists | No | — | — | — | Run only with explicit live-network opt-in; verify commit pinning, outage handling, and no code execution. |
| `NW-T5-01` | Failure and recovery matrix | UNTESTED | Recovery contracts declared | No | — | — | — | Exercise one subsystem failure at a time and verify typed errors, no partial writes, retry, and usability. |
| `NW-T5-02` | SQLite concurrency and Playwright parallelism | UNTESTED | Harness constraint documented | No | — | — | — | Compare workers 1/2/4 with disposable data and close or document lock behavior. |
| `NW-T5-03` | Responsive layout and accessibility | UNTESTED | Automated baseline exists | No | — | — | — | Cover required viewports, keyboard/focus/modal behavior, overflow, graph accessibility, and manual assistive technology separately. |
| `NW-T5-04` | Performance-sensitive workflows | UNTESTED | Measurement scope declared | No | — | — | — | Collect current medians on equivalent hardware/runtime and classify only clear regressions. |
| `NW-T5-05` | Final regression and documentation reconciliation | UNTESTED | Closure process declared | No | — | — | — | Reconcile catalog, status entries, evidence links, code/docs drift, and temporary artifacts after all prior tiers. |

## Tier 0 execution record

### Baseline (`NW-T0-01`)

- Branch: `develop`.
- Revision: `752ab151978b56dd2c20568f63434e346dcf6ea8`.
- Tracked working-tree and staged changes: none before campaign files were added.
- Live catalog: 149 definitions; category counts are 8 inputs, 14 core, 18 convolutional, 5 recurrent, 10 transformer, 14 merge, 73 mathematical, and 7 outputs.
- Catalog fingerprint: `b92166f9e933c1db7bc95e7d48b0d083b570e2f551e573d668cb5d368fd6f7dd`.
- Isolated database: `artifacts/tier0-neuroweave.db` during execution; the disposable database is not required for future runs.

### Launcher (`NW-T0-02`)

The official `start_on_windows.ps1 -Action Launch -NoBrowser` path used isolated
ports 9849 and 6426. Frontend readiness, backend health, migration startup,
catalog loading, and clean termination of the verified launcher-owned process
trees passed. `/api/metadata` remained `unchecked` for optional runtime
discovery during startup, as required by the startup contract.

### Quality (`NW-T0-03`)

| Gate | Result |
| --- | --- |
| Locked dependency sync | PASS |
| Backend compileall | PASS |
| Ruff | FAIL: existing repository-wide findings; see log |
| Pyright | FAIL: 370 errors and 2 warnings; see log |
| Backend pytest | PARTIAL: one export test timed out waiting 10 seconds for discovery; exact isolated rerun passed |
| `npm ci` | PASS |
| Frontend lint | PASS |
| Frontend Vitest | PASS: 20 files, 100 tests |
| Frontend production build | PASS |

The backend timeout is recorded as a test/environment/performance finding, not
as a confirmed application defect: the failing full-suite run observed the
probe complete at approximately 11.85 seconds, and the exact test passed in a
fresh isolated rerun.

### Browser routes (`NW-T0-04`)

The full shell file passed 5/5. The deep-link test then passed three repeated
runs, each covering direct navigation and one refresh for all eight required
routes. The in-app Browser also rendered the `/models` shell against the
officially launched preview before the automated run. The standard shell spec
does not persist a trace or custom console/request log on passing runs; that
observability gap is noted for future browser-slice improvements.

## Evidence and environment

The machine-readable [run manifest](run-manifest.json) records OS/runtime
versions, commands, ports, isolated paths, catalog identity, and cleanup. The
catalog and metadata responses are retained under `catalog/`; complete command
output is under `logs/`. No application source files were changed by this
campaign.

Future agents must start by reading this ledger and the
[validation strategy](../../docs/development/validation_strategy.md), then
continue with `NW-T1-01` and `NW-T1-02` only after inspecting the current
project status ledger and freezing a new run identity.
