# Develop reconciliation QA — 2026-09-02

## Delivery

- PR #1, `Improve model builder boundaries and compilation workflow`, was reviewed against `new-features` tip `6511395ae19ee5c81378eb7d123e0c7352716077`.
- `origin/develop` was `102ba3463ac90ab6b276e926614bfef34df55354`, an ancestor of the PR tip. The local `develop` history was fast-forwarded to the PR tip; the completion changes in this worktree are being delivered as a follow-up commit on `develop`.
- No force-push or worktree was used. The remote `new-features` branch remains unchanged.

## Implementation review

- Graph schema v5 and project bundle format v6 are canonical. Persisted and exported input shapes are feature-only; runtime validation prepends the unknown batch dimension.
- The `20260902_runtime_batch_shapes` migration covers project nodes and revision snapshots, removes effectively empty quantization configuration, and restores the legacy leading-null shape on downgrade.
- Keras `input_shape` remains feature-only. `batch_shape` and `batch_input_shape` discard their leading batch value, warn for a concrete batch size, select the input manifest from normalized rank, and reject empty or malformed feature shapes.
- Effective graph outputs are resolved from the active catalog before compilation validation. Selectors, mappings, task inference, activation checks, and diagnostics are output-specific; native configuration remains backend-specific.
- The Inspector batch control is presentation-only. Catalog filters use manifest capabilities, model I/O counts are graph-derived, the Inference tab remains absent from workspace navigation, and KaTeX fitting has no nested horizontal scroll region.
- Component-local styles use repository tokens and the existing IBM Plex font stack. Documentation dates and the requested serialization, shape, UI, and QA records use 2026-09-02.

## Local validation evidence

- Backend compileall: passed.
- Ruff: passed (`All checks passed!`).
- Full backend pytest: passed at 100% with the repository's existing skipped cases. The cache-disabled invocation reports only the known `cache_dir` PytestConfigWarning.
- Full strict Pyright: not green; the repository reports 288 errors, 0 warnings, and 0 informations, including pre-existing strict typing/stub findings outside this reconciliation. This remains an explicit baseline limitation.
- Frontend Vitest: passed, 6 files and 29 tests.
- Frontend ESLint with `--no-cache`: passed.
- Production Angular build with `--progress=false`: passed; output was generated under `app/tests/cache/frontend-dist`.
- Playwright E2E: passed serially, 12 tests. A parallel run was separately blocked by SQLite write contention (`database is locked`); the serial run completed successfully.

## Browser evidence

At the narrow 560px browser viewport, the Builder catalog and Inspector collapsed to their mobile controls, the progressive filter panel exposed Category, Compatibility, Port type, Family, and Operation choices, the batch preview was unchecked by default and showed `runtime: [batch, ...]` only after interaction, model cards/workspace showed graph-derived counts, workspace navigation omitted Inference, the compilation dialog rendered formulas without a horizontal scrollbar, and evaluated document/body/nested overflow widths were all zero.

## Remote CI status at review

The latest GitHub Actions run for the PR tip (`33645341098`) concluded `failure` immediately with no jobs or logs and GitHub classified it as a workflow-file issue. An earlier `develop` run had the same no-job failure, so this predates the reconciliation and is not evidence against the code changes. PR #1 exposed GitGuardian Security Checks as passing; the full repository CI result was not available as a code-level result.
