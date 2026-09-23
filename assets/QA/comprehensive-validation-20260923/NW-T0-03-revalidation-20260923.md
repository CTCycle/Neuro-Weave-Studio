# NW-T0-03 repository quality-gate revalidation

| Field | Value |
| --- | --- |
| Run ID | `NW-CV-20260923-T0-03-RECHECK` |
| Date | 2026-09-23 |
| Branch | `develop` |
| Base revision | `155f845e7bbf085ea1ce865f52fd062dd950db33` |
| Validated source revision | `912181fa92b8a616fd1cc12665f2567c817c88dc` |
| Status | **PARTIAL** |

## Scope

This rechecks the Tier 0 repository quality baseline after the Builder
validation changes. The previous ledger recorded a full-suite discovery
timeout and nonzero Ruff and Pyright results. The timeout and Ruff result are
resolved in this run. Strict Pyright remains nonzero, so NW-T0-03 and the
quality-gate baseline remain partial.

The project-export discovery helper previously polled for up to ten seconds.
It now uses a 30-second monotonic deadline. The exact export/discovery test
passed on rerun and the full backend suite completed without a failure. Ruff's
`B009` finding in the test helper was corrected. The Pyright run retains a
narrow `reportUnknownMemberType` ignore on `pytest.approx` because the
installed pytest typing surface reports its `approx` parameters as unknown.

## Results

- Full backend pytest suite: **531 passed, 131 skipped, 0 failed**, exit code 0.
- Exact discovery/export regression: passed after the longer bounded polling
  wait.
- Full Ruff check across the server, scripts, and tests: **0 findings**, exit
  code 0.
- Full strict Pyright check: **52 errors, 0 warnings**, across five test
  modules: 21 in `test_node_execution_matrix.py`, 12 in `test_metrics.py`, 9
  in `test_parameter_surface_parity.py`, 6 in
  `test_compilation_capabilities.py`, and 4 in `test_native_parity.py`.
  These findings keep the quality gate partial.
- Full frontend Vitest suite: **104 tests in 20 files passed**. Frontend lint
  and the production build also passed after the Builder source change.

The run did not broaden into a cleanup of the remaining test typing debt.
There is no evidence of a new application API or runtime regression from this
revalidation. The test discovery timeout is no longer outstanding in the
current suite result; repository-wide Pyright findings remain open.

## Evidence

- [Full backend suite](logs/nw-t0-03-backend-pytest-20260923.log)
- [Exact discovery/export test rerun](logs/nw-t0-03-discovery-rerun-20260923.log)
- [Full Ruff result](logs/nw-t0-03-ruff-final-20260923.log)
- [Full strict Pyright result](logs/nw-t0-03-pyright-final-20260923.log)
- [Full frontend suite](logs/nw-t0-03-frontend-vitest-final-20260923.log)
- [Frontend lint after Builder change](logs/nw-t2-01-frontend-lint.log)
- [Production build after Builder change](logs/nw-t2-01-frontend-build.log)
- [Per-check manifest](NW-T0-03-revalidation-20260923-run-manifest.json)

## Remaining boundary

NW-T0-03 stays **PARTIAL** until the 52 strict Pyright errors are triaged and
resolved or explicitly baselined. This recheck does not clear the separate
parallel E2E fixture and Windows dev-server lifecycle issues.
