# NW-T3-03 current-checkout revalidation

Date: 2026-09-29
Repository: `CTCycle/Neuro-Weave-Studio`
Branch: `develop`
Implementation HEAD: `42666a8374599e27367fcc2c9ae2a1ac77798ee7`

## Decision

**PASS, bounded.** The four authoritative convolution workflows passed on the current checkout using the unchanged `app/client/e2e/node-functional-matrix.spec.ts`. No convolution manifest, shape-inference, exporter, runtime, or catalog implementation file was changed for this run.

The working tree contained only the scoped T3-06 test additions/changes while this revalidation ran. The dedicated launcher used a fresh resource root, database, and ports: `runtimes/cache/nw-t3-03-revalidation-20260929`, FastAPI `9854`, and UI `6431`.

## Frozen environment

- Catalog: 149 definitions; category counts and all definition IDs are in [catalog-freeze.json](evidence/catalog-freeze.json).
- Catalog fingerprint: `306042a67a66852161de34e2c72c35bccb509a80b1245172af11f53180b500d1`.
- Python `3.13.14`; Keras `3.15.1`; TensorFlow `2.21.0`; PyTorch `2.13.0+cu126`; JAX `0.11.0`.
- TensorFlow/JAX ran on CPU; PyTorch discovery reported the NVIDIA GeForce RTX 3060 Laptop GPU, while these cases explicitly selected CPU.

## Browser/runtime cases

| Case | Static output / params | Runtime output / params | Worker | Result |
| --- | --- | --- | --- | --- |
| Conv1D → AvgPool1D → GlobalPool1D | `[null,32]` / `320` | `[1,32]` / `320` | Keras/PyTorch/CPU | PASS |
| Conv2D → MaxPool2D → GlobalPool2D | `[null,32]` / `896` | `[1,32]` / `896` | Keras/PyTorch/CPU | PASS |
| Conv3D → AvgPool3D → GlobalPool3D | `[null,32]` / `896` | `[1,32]` / `896` | Keras/PyTorch/CPU | PASS |
| Conv2D → MaxPool2D → GlobalPool2D | `[null,32]` / `896` | `[1,32]` / `896` | native/PyTorch/CPU | PASS |

Each graph used five materialized nodes and four Builder-created edges. The existing spec verified successful connection responses, static validation, exact shape and parameter summaries, save/reload node rendering, exact Functional Validation worker identity, succeeded runtime state, empty browser-error payloads, and fixture deletion. Builder-ready and runtime-succeeded screenshots are retained under [evidence/nw-t3-03-revalidation](evidence/nw-t3-03-revalidation).

## Checks and boundaries

The focused backend boundary suite passed, as did Ruff, frontend lint, 105 frontend unit tests, and the production build. Strict Pyright remains nonzero at the documented 58-error baseline. This result does not claim the remaining convolution definitions, all six live profiles, source/export, training, accessibility, or worker-matrix closure.

The exact command/result and job identifiers are recorded in [NW-T3-03-revalidation-run-manifest.json](NW-T3-03-revalidation-run-manifest.json).
