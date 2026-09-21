# Comprehensive Validation Strategy

Last updated: 2026-09-21

## Purpose

This document is the durable digest of the NeuroWeave comprehensive end-to-end
validation roadmap. It defines the order, evidence boundary, and closure rules
for future validation work without treating source presence, catalog metadata,
or historical reports as execution proof.

The revision-specific campaign ledger is the detailed execution record:

- [2026-09-21 campaign ledger](../../QA/comprehensive-validation-20260921/validation-ledger.md)
- [campaign run manifest](../../QA/comprehensive-validation-20260921/run-manifest.json)

The high-level operational status remains canonical in the
[project status ledger](../project_status_ledger.md).

## Evidence rules

Use these fields independently for every slice:

- `Feature exists`: the capability is present or intentionally bounded in the
  current product.
- `Exercised`: the named scenario actually ran against a stated revision and
  environment.
- `Status`: the result of the required evidence, not an inference from the
  first two fields.

Use `PASS`, `PARTIAL`, `FAIL`, `BLOCKED`, `UNTESTED`, and `UNKNOWN` in the
campaign ledger. Keep `BLOCKED` for a correctly detected unavailable
environment/toolchain capability; use `FAIL` for an expected path or expected
diagnostic that does not behave correctly. A catalog mapping, metadata response,
retrieval result, or static test is not runtime/render evidence by itself.

Every positive claim must identify the exact revision, environment, execution
profile, scenarios, and evidence paths. Profiles are part of test identity:
never accept a fallback backend or API as proof for the selected profile.

## Universal slice contract

Each slice follows the same bounded loop:

```text
Inspect expected behavior
→ establish isolated state and fixtures
→ execute the exact scenarios
→ collect browser/API/backend/persistence evidence
→ reproduce and classify failures
→ apply one narrow fix only when requested and justified
→ rerun the failure, one valid neighbor, and one negative neighbor
→ update the campaign ledger
→ update the project status ledger only when the evidence supports it
```

Browser slices should retain a final rendered result, page/console/request
diagnostics where available, and the relevant backend log section. Persistence
slices additionally record the database path and before/after state. Runtime
and export slices record profile identity, worker/job state, artifact manifest,
and hashes where applicable.

## Ordered campaign

### Tier 0 — Environment and startup

This is the first executable tier and must precede feature-heavy workflows.

| Slice | Focus | Closure evidence |
| --- | --- | --- |
| `NW-T0-01` | Freeze branch, commit, working tree, dependency versions, live catalog, and database/cache identity. | Run manifest, catalog JSON, deterministic catalog hash, version record. |
| `NW-T0-02` | Official Windows launcher readiness, migration-on-start, frontend/backend handoff, and owned-process shutdown. | Launcher output, health/metadata, process tree before/after, port check, isolated database. |
| `NW-T0-03` | Repository quality baseline: sync, compile, Ruff, Pyright, backend tests, frontend lint/test/build. | Complete command logs with separate functional and static-analysis outcomes. |
| `NW-T0-04` | Direct navigation and refresh for `/datasets`, `/training`, `/environment`, `/metrics`, `/nodes`, `/settings`, `/builder`, and `/models/missing/compilation`. | Browser test output, route list, rendered shell assertions, retry/repetition result. |

The 2026-09-21 run completed this tier as `PARTIAL`: baseline, launcher
startup/shutdown, and deep-link repetition passed; the quality baseline retains
Ruff/Pyright debt and one non-reproducible full-suite discovery timeout. See the
campaign ledger for exact evidence.

### Tier 1 — Application foundations

Run these after Tier 0 and before core product workflows.

| Slice | Focus |
| --- | --- |
| `NW-T1-01` | Shell navigation, wildcard and missing-model routes, nested routes, anchors, and browser Back/Forward. |
| `NW-T1-02` | Clean/current/older migration paths, concurrent initialization, rollback, schema rejection, foreign keys, and restart persistence. |
| `NW-T1-03` | Project create/open/edit/save/revision/duplicate/delete, canonical JSON import/export, stale writes, restart, and Keras import. |
| `NW-T1-04` | Runtime API/backend/device/memory preference persistence, malformed/stale storage, reload, and unavailable storage. |
| `NW-T1-05` | Settings non-feature boundary plus the intentionally disabled/session-only Training boundary. |

### Tier 2 — Core workflows

| Slice | Focus |
| --- | --- |
| `NW-T2-01` | Builder editing, undo/redo, selection, inspector reset, dirty state, save/reopen, and invalid-draft recovery. |
| `NW-T2-02` | Connection candidates, ports, shape inference, cycles, incompatible edges, and repairable invalid state. |
| `NW-T2-03` | Profile-aware compilation configuration, source identity/freshness, and unsupported compiler behavior. |
| `NW-T2-04` | Functional Validation submission, polling, cancellation, worker cleanup, exact profile identity, and reload. |
| `NW-T2-05` | Tensor inspection entry points, traces, statistics, slicing, expiry/cancellation, and fail-closed runtime checks. |
| `NW-T2-06` | Dataset registration, source inspection, roles, bounded metadata, persistence, relink, and deletion. |
| `NW-T2-07` | Dataset processing and split configuration as the current declarative boundary. |
| `NW-T2-08` | Custom metric/loss create, edit, validation, save/reload, task compatibility, and deletion. |
| `NW-T2-09` | Safe mathematical and graph-template custom-node authoring, materialization, validation, persistence, and reference-only repositories. |

### Tier 3 — Model Builder catalog campaign

Use the live catalog fingerprint captured by `NW-T0-01` and preserve the
existing L0-L6 Model Builder ledger. Do not claim full coverage from a handful
of representative graphs.

| Slice | Bounded family |
| --- | --- |
| `NW-T3-01` | Input and basic shape/core nodes. |
| `NW-T3-02` | Dense, embedding, dropout, and normalization. |
| `NW-T3-03` | Basic convolution and pooling. |
| `NW-T3-04` | Advanced/spatial convolution. |
| `NW-T3-05` | Recurrent nodes. |
| `NW-T3-06` | Attention primitives. |
| `NW-T3-07` | Transformer blocks. |
| `NW-T3-08` | Arithmetic merge and fan-in. |
| `NW-T3-09` | Structural merge and branching. |
| `NW-T3-10` | Unary mathematical operations: absolute, sign, exponential/logarithmic, and square-root families. |
| `NW-T3-11` | Rounding, trigonometric, and elementary scalar mathematical operations. |
| `NW-T3-12` | Hyperbolic, activation, and probability operations. |
| `NW-T3-13` | Equality, ordering, and logical operations. |
| `NW-T3-14` | Reductions and aggregate operations. |
| `NW-T3-15` | Arithmetic, dot, and matrix operations. |
| `NW-T3-16` | Cast, clip, concatenate, dimension, reshape, repetition, stack, transpose, and tiling. |
| `NW-T3-17` | Indexing, slicing, expressions, stop-gradient, and any live definitions not otherwise rostered. |
| `NW-T3-18` | Output/task contracts and explicit versus inferred outputs. |
| `NW-T3-19` | Generated compatible ports/edges, cardinality, state, cycles, dangling references, and unreachable components. |

Each node-family slice closes only when Builder materialization, parameters,
ports/shapes, static validation, persistence/recovery, source, and applicable
runtime evidence are recorded per supported profile. Unsupported paths need
explicit fail-closed evidence.

### Tier 4 — Runtime and integration-heavy validation

| Slice | Focus |
| --- | --- |
| `NW-T4-01` | Fresh six-profile live matrix: Keras TensorFlow/PyTorch/JAX and native TensorFlow/PyTorch/JAX. |
| `NW-T4-02` | TensorFlow XLA, JAX JIT, and PyTorch `torch.compile` capability gates without fallback. |
| `NW-T4-03` | Capability-gated export artifacts, manifests, hashes, structural checks, reload/forward verification, and browser downloads. |
| `NW-T4-04` | Opt-in public Hugging Face repository reference, resolved commit pin, reload/edit/delete, outage handling, and no code execution. |

### Tier 5 — Resilience, usability, and closure

| Slice | Focus |
| --- | --- |
| `NW-T5-01` | Failure/recovery matrix for startup, payloads, revisions, graphs, runtimes, workers, exports, inspection, and external references. |
| `NW-T5-02` | SQLite behavior at Playwright worker counts 1, 2, and 4 using disposable data; document any intentional harness constraint. |
| `NW-T5-03` | Responsive and accessibility coverage at 390×844, 768×900, 1280×720, and 1440×900, including keyboard/manual assistive-technology boundaries. |
| `NW-T5-04` | Median cold-start, catalog, Builder, validation, discovery, Functional Validation, export, and dataset inspection timings. |
| `NW-T5-05` | Final regression, live catalog reconciliation, evidence-link repair, documentation reconciliation, and temporary-artifact audit. |

## Failure and remediation policy

Reproduce a failure twice interactively or consistently in automation before
calling it a product defect. Classify it as one primary category: functional,
frontend, backend, integration, persistence/state, performance, UX/workflow,
configuration, or test/environment. Trace the first incorrect state transition
from user action through UI/store, API, service, repository/runtime/exporter,
and persisted side effect. Fix narrowly, then run the exact failure, a valid
neighbor, and a negative neighbor before changing status.

The minimum adjacent regression follows the changed boundary: routing includes
shell plus Builder and nested model routes; persistence includes migrations and
CRUD; Builder changes include focused store tests and the relevant workflow;
compilation/runtime/export changes include a representative source or worker
path; accessibility styling includes the accessibility suite and affected E2E.

## Completion boundary

NeuroWeave remains `validation in progress` until the current revision has a
green or explicitly qualified foundation, core workflow, catalog, profile,
export, resilience, and evidence-integrity record. Every PASS must link to
current evidence; no high-priority feature may remain `PARTIAL`, `FAIL`,
`UNTESTED`, or `UNKNOWN`. A genuine unavailable toolchain may remain
`BLOCKED` only when the product detects and reports it clearly.
