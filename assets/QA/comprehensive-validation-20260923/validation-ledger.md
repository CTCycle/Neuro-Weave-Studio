# NeuroWeave Comprehensive Validation Ledger

Last updated: 2026-09-24

Run ID: `NW-CV-20260924-T2-03-04-T4-02`

Repository: `develop`; NW-T2-03, NW-T2-04, and NW-T4-02 source validated at `6c1687457822380f471c7ce8129b93632af96972`. The run evidence and source revision are recorded in the [2026-09-24 run manifest](../comprehensive-validation-20260924/NW-T2-03-04-T4-02-run-manifest.json).

Historical Tier 0 evidence remains at `752ab151978b56dd2c20568f63434e346dcf6ea8`; the current NW-T0-03 revalidation is linked below.

Campaign status: **PARTIAL**. NW-T2-03 and NW-T2-04 now PASS. NW-T4-02 remains PARTIAL because TensorFlow/XLA is blocked by the missing TensorFlow package and PyTorch `torch.compile` is unavailable in the current launcher environment. This is not a full application certification.

## Aggregate status

| Tier | Scope | PASS | PARTIAL | FAIL | BLOCKED | UNTESTED |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tier 0 | Environment and startup | 3 | 1 | 0 | 0 | 0 |
| Tier 1 | Application foundations | 5 | 0 | 0 | 0 | 0 |
| Tier 2 | Core workflows | 4 | 0 | 0 | 0 | 5 |
| Tier 3 | Model Builder catalog | 0 | 0 | 0 | 0 | 19 |
| Tier 4 | Runtime and integrations | 0 | 1 | 0 | 0 | 3 |
| Tier 5 | Resilience, usability, closure | 0 | 0 | 0 | 0 | 5 |
| Total | 46 slices | 12 | 2 | 0 | 0 | 32 |

UNTESTED means that the slice remains in campaign scope but was not run in
this revision-specific pass. It does not mean the feature is absent or broken.

## Slice ledger

| Slice ID | Capability | Status | Revision | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- |
| NW-T0-01 | Freeze validation baseline and live catalog | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Repeat when branch, dependencies, catalog, or runtime identity changes. |
| NW-T0-02 | Official launcher startup and owned-process shutdown | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Repeat after launcher, port, dependency, runtime, or startup-log changes. |
| NW-T0-03 | Repository quality-gate baseline | PARTIAL | `912181fa92b8a616fd1cc12665f2567c817c88dc` | [2026-09-23 revalidation](NW-T0-03-revalidation-20260923.md), [backend suite](logs/nw-t0-03-backend-pytest-20260923.log), [Ruff](logs/nw-t0-03-ruff-final-20260923.log), [Pyright](logs/nw-t0-03-pyright-final-20260923.log) | Full backend suite and Ruff pass; the prior discovery timeout is resolved. Strict Pyright still reports 52 errors across five test modules. |
| NW-T0-04 | Direct deep links and refresh | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Preserve route coverage when the shell contract changes. |
| NW-T1-01 | Shell navigation and route boundaries | PASS | `50d54116c4a007d1342ff99dc7e11e9e6e2f41d8` | [NW-T1-01 report](../comprehensive-validation-20260922/NW-T1-01.md), browser diagnostics | Preserve route coverage when shell or route contracts change. |
| NW-T1-02 | Database migrations and persistence | PASS | `a30eefc93658138400e4061f4af07eed4c021985` | [NW-T1-02 report](../comprehensive-validation-20260922/NW-T1-02.md), 10 focused tests | Repeat migration and persistence checks after schema or repository changes. |
| NW-T1-03 | Project lifecycle, revisions, JSON bundles, stale writes, restart, and Keras import | PASS | `d42bda2330581461fff2e52663780d0ea3fa5e53` | [NW-T1-03 report](NW-T1-03.md), [run manifest](NW-T1-03-run-manifest.json), [backend log](logs/nw-t1-03-backend-pytest.log), [serial browser log](logs/nw-t1-03-playwright-cold.log), [Builder screenshot](evidence/nw-t1-03-project-bundle-builder.png) | Repeat after project-schema, bundle, runtime-discovery, or import changes. |
| NW-T1-04 | Runtime API/backend/device/memory preference persistence, stale and malformed storage, reload, unavailable storage | PASS | `02b9d24edb5fcfe536ab5a74f8fde0b94c03bd13` | [NW-T1-04 report](NW-T1-04.md), [run manifest](NW-T1-04-run-manifest.json), [10 focused tests](logs/nw-t1-04-focused-vitest.log), [6 serial browser workflows](logs/nw-t1-04-playwright.log), [post-reload screenshot](evidence/nw-t1-04-active-preference-reload.png) | Repeat after preference storage/schema or runtime metadata contract changes; this slice does not claim runtime execution. |
| NW-T1-05 | Settings informational boundary and Training prerequisites, session-only draft, disabled start, SPA retention, reload loss | PASS | `cc0f937b1f52d06fa36328e083be163021a3829b` | [NW-T1-05 report](NW-T1-05.md), [run manifest](NW-T1-05-run-manifest.json), [serial browser log](logs/nw-t1-05-playwright.log), [in-app Browser observation](evidence/nw-t1-05-inapp-browser-observation.md), [diagnostics](evidence/nw-t1-05-browser-diagnostics.json), [screenshots](evidence/nw-t1-05-training-session-draft.png) | Repeat after Settings routes, training prerequisites, or draft lifecycle change; training execution remains unimplemented. |
| NW-T2-01 | Builder selection, Inspector editing/reset, undo/redo, dirty state, save/reopen, invalid-draft repair | PASS | `912181fa92b8a616fd1cc12665f2567c817c88dc` | [NW-T2-01 report](NW-T2-01.md), [run manifest](NW-T2-01-run-manifest.json), [serial Playwright](logs/nw-t2-01-playwright.log), [valid/negative neighbors](logs/nw-t2-01-valid-negative-neighbors.log), [focused unit tests](logs/nw-t2-01-vitest.log), [Browser diagnostics](evidence/nw-t2-01-browser-diagnostics.json), [rendered evidence](evidence/nw-t2-01-reset-and-history.png) | Repeat after Builder history, Inspector, parameter, or project persistence changes. |
| NW-T2-02 | Builder connection candidates, port/cardinality behavior, shape inference, cycles, incompatible edges, repair | PASS | `66d04b3c788baf8ba2921a7f3c1cd4c8541e1e44` | [NW-T2-02 report](NW-T2-02.md), [run manifest](NW-T2-02-run-manifest.json), [13 backend tests](logs/nw-t2-02-backend-focused-final.log), [39 frontend tests](logs/nw-t2-02-frontend-final.log), [16 serial browser workflows](logs/nw-t2-02-workflows-final-current.log), [3 serial repeats](logs/nw-t2-02-connection-repeat-final-current.log), [2-worker run](logs/nw-t2-02-two-workers-final-current.log), [in-app Browser observation](evidence/nw-t2-02-inapp-browser-observation.md), [cycle rejection](evidence/nw-t2-02-final-browser-run/nw-t2-02-cycle-rejection.png), [inferred merge shape](evidence/nw-t2-02-final-browser-run/nw-t2-02-valid-merge-shape.png) | The focused 2-worker recheck did not reproduce a lock; `NW-T5-02` remains open for the full 1/2/4-worker SQLite matrix. |
| NW-T2-03 | Profile-aware compilation, source identity/freshness, and unsupported compiler behavior | PASS | `6c1687457822380f471c7ce8129b93632af96972` | [combined run report](../comprehensive-validation-20260924/NW-T2-03-04-T4-02.md), [19 workflow cases](../comprehensive-validation-20260924/logs/nw-t2-03-04-t4-02-workflows-final-20260924.log), [JAX source screenshot](../comprehensive-validation-20260924/evidence/nw-t2-03-jax-source-revision.png), [focused JAX E2E](../comprehensive-validation-20260924/logs/nw-t2-03-jax-profile-e2e-focused-final-20260924.log) | Repeat after compiler-capability or source-freshness contract changes; this pass covers the selected Keras/PyTorch and native JAX profiles only. |
| NW-T2-04 | Functional Validation polling, cancellation, cleanup, exact profile identity, and reload restore | PASS | `6c1687457822380f471c7ce8129b93632af96972` | [combined run report](../comprehensive-validation-20260924/NW-T2-03-04-T4-02.md), [19 workflow cases](../comprehensive-validation-20260924/logs/nw-t2-03-04-t4-02-workflows-final-20260924.log), [restored result/cancel screenshot](../comprehensive-validation-20260924/evidence/nw-t2-04-validation-restored-after-reload.png), [terminal cleanup record](../comprehensive-validation-20260924/evidence/nw-t2-04-cancel-cleanup.json), [native JAX reload observation](../comprehensive-validation-20260924/evidence/nw-t2-03-jax-inapp-browser-observation.md) | Repeat after job lifecycle, session reference, or worker cleanup changes; all-profile and full-catalog runtime coverage remains separate. |
| NW-T4-02 | TensorFlow XLA, JAX JIT, and PyTorch `torch.compile` capability gates without fallback | PARTIAL | `6c1687457822380f471c7ce8129b93632af96972` | [combined run report](../comprehensive-validation-20260924/NW-T2-03-04-T4-02.md), [live metadata](../comprehensive-validation-20260924/evidence/nw-t4-02-live-metadata.json), [runtime snapshot](../comprehensive-validation-20260924/evidence/nw-t4-02-runtime-discovery.json), [compiler-gate E2E](../comprehensive-validation-20260924/logs/nw-t4-02-compiler-gates-e2e-focused-20260924.log), [backend rejection tests](../comprehensive-validation-20260924/logs/nw-t4-02-compiler-gates-pytest-final-20260924.log) | JAX JIT executed successfully. PyTorch compiler is unavailable and rejected; TensorFlow is not installed, so XLA discovery/execution cannot be completed here. |

Five other Tier 2 slices, all Tier 3 and Tier 5 slices, and three Tier 4
slices remain UNTESTED. NW-T4-02 remains PARTIAL as recorded above. The next
ordered action is NW-T2-05.

## Evidence

- Detailed slice report: [NW-T1-03.md](NW-T1-03.md)
- Per-slice manifest: [NW-T1-03-run-manifest.json](NW-T1-03-run-manifest.json)
- Detailed NW-T1-04 report: [NW-T1-04.md](NW-T1-04.md)
- NW-T1-04 run manifest: [NW-T1-04-run-manifest.json](NW-T1-04-run-manifest.json)
- NW-T1-04 launcher, unit, build, Playwright, and cleanup logs: `logs/`
- NW-T1-04 browser diagnostics and screenshot: `evidence/`
- Detailed NW-T1-05 report: [NW-T1-05.md](NW-T1-05.md)
- NW-T1-05 run manifest: [NW-T1-05-run-manifest.json](NW-T1-05-run-manifest.json)
- NW-T1-05 launcher, Playwright, fixture-cleanup, and process-cleanup logs: `logs/`
- NW-T1-05 Browser diagnostics, in-app observation, and screenshots: `evidence/`
- Detailed NW-T2-02 report: [NW-T2-02.md](NW-T2-02.md)
- NW-T2-02 run manifest: [NW-T2-02-run-manifest.json](NW-T2-02-run-manifest.json)
- NW-T2-02 launcher, backend/frontend tests, serial/two-worker browser runs, shell lifecycle, and cleanup logs: `logs/`
- NW-T2-02 in-app Browser observation, catalog snapshot, and rendered screenshots: `evidence/`
- Detailed 2026-09-24 report: [NW-T2-03, NW-T2-04, and NW-T4-02](../comprehensive-validation-20260924/NW-T2-03-04-T4-02.md)
- 2026-09-24 run manifest: [NW-T2-03-04-T4-02-run-manifest.json](../comprehensive-validation-20260924/NW-T2-03-04-T4-02-run-manifest.json)
- 2026-09-24 selected workflow, backend, frontend, build, lint, runtime, and cleanup evidence: `../comprehensive-validation-20260924/logs/` and `../comprehensive-validation-20260924/evidence/`
- 2026-09-24 disposable project, process, port, and isolated cache cleanup: [fixture cleanup](../comprehensive-validation-20260924/logs/nw-t2-03-04-fixture-cleanup-20260924.log), [process cleanup](../comprehensive-validation-20260924/logs/nw-t2-03-04-process-cleanup-20260924.log)
- Detailed NW-T2-01 report: [NW-T2-01.md](NW-T2-01.md)
- NW-T2-01 run manifest: [NW-T2-01-run-manifest.json](NW-T2-01-run-manifest.json)
- NW-T2-01 launcher, Playwright, neighbor, focused unit, build, fixture-cleanup, and process-cleanup logs: `logs/`
- NW-T2-01 Browser diagnostics, in-app observation, and screenshots: `evidence/`
- NW-T0-03 revalidation report and manifest: [NW-T0-03-revalidation-20260923.md](NW-T0-03-revalidation-20260923.md), [run manifest](NW-T0-03-revalidation-20260923-run-manifest.json)
- Backend and launcher logs: `logs/`
- Browser diagnostics and screenshots: `evidence/`
- Separate full Model Builder catalog/profile work remains pending in the [Model Builder validation ledger](../model-builder-validation-ledger.md).

The 2026-09-22 ledger remains historical and unchanged. This ledger preserves
the revision and evidence boundary for every completed slice.

## Harness note

The first exploratory Keras browser attempt had no completed runtime discovery
in its isolated state and stopped before the import request. The browser flow
now checks Keras on the Environment page before importing. A clean-cache rerun
passed, including an available live PyTorch/Keras probe and import response
201. This was a harness precondition, not a reproduced product failure.

Earlier focused NW-T1 runs used the official preview and therefore did not
exercise Playwright's configured `webServer` lifecycle. NW-T2-02 ran the
default serial shell suite without a base URL override; all ten tests passed,
the Playwright command exited naturally, and port 6426 had no listener. This
closes `ISSUE-006` for the current Windows workflow. The full worker-count
matrix for shared SQLite remains open as `NW-T5-02`/`ISSUE-001`; its focused
two-worker recheck passed without reproducing a lock. Strict Pyright remains
at 52 errors (`ISSUE-003`).

## Next action

Continue with NW-T2-05, the next ordered untested Tier 2 workflow: tensor inspection lifecycle and fail-closed runtime checks.
