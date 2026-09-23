# NeuroWeave Comprehensive Validation Ledger

Last updated: 2026-09-23

Run ID: `NW-CV-20260923-T1-04`

Repository: `develop` at base revision `db3351ed62f3e397c5dd09210181a9ab921492e0`; NW-T1-04 source validated at `02b9d24edb5fcfe536ab5a74f8fde0b94c03bd13`.

Historical Tier 0 evidence remains at `752ab151978b56dd2c20568f63434e346dcf6ea8`.

Campaign status: **PARTIAL**. NW-T1-04 adds one PASS slice to the existing Tier 0, NW-T1-01, NW-T1-02, and NW-T1-03 results. This is not a full application certification.

## Aggregate status

| Tier | Scope | PASS | PARTIAL | FAIL | BLOCKED | UNTESTED |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tier 0 | Environment and startup | 3 | 1 | 0 | 0 | 0 |
| Tier 1 | Application foundations | 4 | 0 | 0 | 0 | 1 |
| Tier 2 | Core workflows | 0 | 0 | 0 | 0 | 9 |
| Tier 3 | Model Builder catalog | 0 | 0 | 0 | 0 | 19 |
| Tier 4 | Runtime and integrations | 0 | 0 | 0 | 0 | 4 |
| Tier 5 | Resilience, usability, closure | 0 | 0 | 0 | 0 | 5 |
| Total | 46 slices | 7 | 1 | 0 | 0 | 38 |

UNTESTED means that the slice remains in campaign scope but was not run in
this revision-specific pass. It does not mean the feature is absent or broken.

## Slice ledger

| Slice ID | Capability | Status | Revision | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- |
| NW-T0-01 | Freeze validation baseline and live catalog | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Repeat when branch, dependencies, catalog, or runtime identity changes. |
| NW-T0-02 | Official launcher startup and owned-process shutdown | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Repeat after launcher, port, dependency, runtime, or startup-log changes. |
| NW-T0-03 | Repository quality-gate baseline | PARTIAL | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Ruff, Pyright, and the full-suite discovery timeout remain tracked debt. |
| NW-T0-04 | Direct deep links and refresh | PASS | `752ab151978b56dd2c20568f63434e346dcf6ea8` | 2026-09-21 ledger | Preserve route coverage when the shell contract changes. |
| NW-T1-01 | Shell navigation and route boundaries | PASS | `50d54116c4a007d1342ff99dc7e11e9e6e2f41d8` | [NW-T1-01 report](../comprehensive-validation-20260922/NW-T1-01.md), browser diagnostics | Preserve route coverage when shell or route contracts change. |
| NW-T1-02 | Database migrations and persistence | PASS | `a30eefc93658138400e4061f4af07eed4c021985` | [NW-T1-02 report](../comprehensive-validation-20260922/NW-T1-02.md), 10 focused tests | Repeat migration and persistence checks after schema or repository changes. |
| NW-T1-03 | Project lifecycle, revisions, JSON bundles, stale writes, restart, and Keras import | PASS | `d42bda2330581461fff2e52663780d0ea3fa5e53` | [NW-T1-03 report](NW-T1-03.md), [run manifest](NW-T1-03-run-manifest.json), [backend log](logs/nw-t1-03-backend-pytest.log), [serial browser log](logs/nw-t1-03-playwright-cold.log), [Builder screenshot](evidence/nw-t1-03-project-bundle-builder.png) | Repeat after project-schema, bundle, runtime-discovery, or import changes. |
| NW-T1-04 | Runtime API/backend/device/memory preference persistence, stale and malformed storage, reload, unavailable storage | PASS | `02b9d24edb5fcfe536ab5a74f8fde0b94c03bd13` | [NW-T1-04 report](NW-T1-04.md), [run manifest](NW-T1-04-run-manifest.json), [10 focused tests](logs/nw-t1-04-focused-vitest.log), [6 serial browser workflows](logs/nw-t1-04-playwright.log), [post-reload screenshot](evidence/nw-t1-04-active-preference-reload.png) | Repeat after preference storage/schema or runtime metadata contract changes; this slice does not claim runtime execution. |

NW-T1-05, all Tier 2, Tier 3, Tier 4, and Tier 5 slices remain UNTESTED. The
next action is NW-T1-05, Settings and session-only Training boundaries.

## Evidence

- Detailed slice report: [NW-T1-03.md](NW-T1-03.md)
- Per-slice manifest: [NW-T1-03-run-manifest.json](NW-T1-03-run-manifest.json)
- Detailed NW-T1-04 report: [NW-T1-04.md](NW-T1-04.md)
- NW-T1-04 run manifest: [NW-T1-04-run-manifest.json](NW-T1-04-run-manifest.json)
- NW-T1-04 launcher, unit, build, Playwright, and cleanup logs: `logs/`
- NW-T1-04 browser diagnostics and screenshot: `evidence/`
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

`ISSUE-006` remains a separate historical Playwright process-cleanup issue; the
focused run returned with port 6426 free. It does not classify route or import
behavior as a product failure.

The NW-T1-04 six-test serial browser run also exited successfully and released
port 6426. `ISSUE-006` remains open because its broader Windows shell-gate
reproductions are still unresolved.

## Next action

Continue with NW-T1-05 using a fresh run identity: Settings non-feature
boundaries and the intentionally disabled/session-only Training workflow.
