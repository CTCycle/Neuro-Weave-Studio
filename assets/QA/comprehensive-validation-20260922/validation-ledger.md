# NeuroWeave Comprehensive Validation Ledger

Last updated: 2026-09-22

Run ID: NW-CV-20260922-T1-02

Repository: develop at a30eefc93658138400e4061f4af07eed4c021985

Historical Tier 0 evidence remains at 752ab151978b56dd2c20568f63434e346dcf6ea8.

Campaign status: PARTIAL. This run adds one PASS slice to the existing Tier 0
and NW-T1-01 results. The campaign is not a full application certification.

## Aggregate status

| Tier | Scope | PASS | PARTIAL | FAIL | BLOCKED | UNTESTED |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tier 0 | Environment and startup | 3 | 1 | 0 | 0 | 0 |
| Tier 1 | Application foundations | 2 | 0 | 0 | 0 | 3 |
| Tier 2 | Core workflows | 0 | 0 | 0 | 0 | 9 |
| Tier 3 | Model Builder catalog | 0 | 0 | 0 | 0 | 19 |
| Tier 4 | Runtime and integrations | 0 | 0 | 0 | 0 | 4 |
| Tier 5 | Resilience, usability, closure | 0 | 0 | 0 | 0 | 5 |
| Total | 46 slices | 5 | 1 | 0 | 0 | 40 |

UNTESTED means that the slice remains in campaign scope but was not run in this
revision-specific pass. It does not mean the feature is absent or broken.

## Slice ledger

| Slice ID | Capability | Status | Revision | Evidence | Remaining gap |
| --- | --- | --- | --- | --- | --- |
| NW-T0-01 | Freeze validation baseline and live catalog | PASS | 752ab151978b56dd2c20568f63434e346dcf6ea8 | 2026-09-21 ledger | Repeat when branch, dependencies, catalog, or runtime identity changes. |
| NW-T0-02 | Official launcher startup and owned-process shutdown | PASS | 752ab151978b56dd2c20568f63434e346dcf6ea8 | 2026-09-21 ledger | Repeat after launcher, port, dependency, runtime, or startup-log changes. |
| NW-T0-03 | Repository quality-gate baseline | PARTIAL | 752ab151978b56dd2c20568f63434e346dcf6ea8 | 2026-09-21 ledger | Ruff, Pyright, and the full-suite discovery timeout remain tracked debt. |
| NW-T0-04 | Direct deep links and refresh | PASS | 752ab151978b56dd2c20568f63434e346dcf6ea8 | 2026-09-21 ledger | Preserve route coverage when the shell contract changes. |
| NW-T1-01 | Shell navigation and route boundaries | PASS | 50d54116c4a007d1342ff99dc7e11e9e6e2f41d8 | NW-T1-01 report, run manifest, browser diagnostics | Preserve route coverage when shell or route contracts change. |
| NW-T1-02 | Database migrations and persistence | PASS | a30eefc93658138400e4061f4af07eed4c021985 | [NW-T1-02 report](NW-T1-02.md), [run manifest](NW-T1-02-run-manifest.json), [focused pytest log](logs/nw-t1-02-focused-pytest.log) | Repeat migration and persistence checks after schema or repository changes. |

NW-T1-03 through NW-T1-05, all Tier 2, Tier 3, Tier 4, and Tier 5 slices
remain UNTESTED in this ledger.

## Evidence

- Detailed reports: NW-T1-01.md and NW-T1-02.md
- Per-slice manifests: run-manifest.json (NW-T1-01), NW-T1-02-run-manifest.json
- Browser diagnostics: evidence/
- Logs: logs/
- Screenshots: screenshots/

The evidence is revision and environment scoped. The existing 2026-09-21
ledger was not overwritten.

## Harness note

The shell and focused tests passed, but Playwright did not release its
run-owned Angular dev-server process automatically in this Windows environment.
The process was identified and stopped after each successful run, and ports
6426 and 9849 were verified free. This is tracked as ISSUE-006 in the project
status ledger. It does not classify the route behavior as a product failure.

## Next action

Continue with NW-T1-03 using a fresh run identity and the current revision after
inspecting this ledger and the project status ledger.
