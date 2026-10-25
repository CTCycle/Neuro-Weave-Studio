# P0/P1 Validation Receipt

Date: 2026-10-25  
Scope: only P0 and P1 findings from the NeuroWeave validation review.

## P0 — CI and reported end-to-end regressions

Pushed commits:

- `543bcd1` — sorted the 69 Ruff I001 findings, fixed the repository Protocol typing error, and corrected the persistence architecture note.
- `0ea67a1` — installed all supported runtime extras in the CI browser job, fixed settings test fixture selection, and fixed the attention-group and merge-validation E2E races.

Checks completed on the pushed source:

- Ruff: passed (`ruff check . ../tests ../scripts --no-cache`).
- Pyright: passed with 0 errors across 154 files.
- Backend suite: passed with one expected skip.
- Frontend: 104 unit tests passed; lint and production build passed.
- Native parity: TensorFlow 39, PyTorch 39, and JAX 39 passed.
- Targeted browser regressions: NW-T3-06 7/7, NW-T4-01 six-profile matrix 1/1, NW-T1-05 1/1, and NW-T3-08 11/11 passed.

The first clean 180-case Playwright replay completed with 177 passed, 1 skipped, 1 failed, and 1 not run. That sole failure came from the local launch harness setting `NW_SETTINGS_E2E_PROFILE` to a new empty directory; `database settings.spec` could not find the directory's `database-config.json`. With the override unset, those database settings tests passed 2/2. The corrected CI-configured full replay used a fresh SQLite resource root and `CI=true`; it completed 179 passed, 1 skipped, 0 failed in 45.2 minutes. NW-T3-06, NW-T4-01, NW-T1-05, and NW-T3-08 all passed in that run.

The GitHub combined-status lookup returned no status entries for the pushed revisions, so no remote CI result is claimed here.

## P1 — NW-T5-01 scope

NW-T5-01 is closed as PASS for explicitly bounded controlled-failure and recovery behavior. The focused tests passed for (1) an injected worker import failure that fails closed, preserves service health and cleans the failed workspace, followed by successful same-profile inference, and (2) an injected repository `URLError` that returns 503 without persisting a definition, followed by successful resolution. The full backend suite also passed these tests.

A later isolated attempt at the repository-outage test produced no test result and was stopped after it stalled in the test harness. This adds no new pass or failure evidence. Naturally occurring import failures, crashes, and outages are excluded from NW-T5-01 acceptance criteria because no stable, reproducible trigger is available; the PASS claim covers controlled injections and recovery only.

