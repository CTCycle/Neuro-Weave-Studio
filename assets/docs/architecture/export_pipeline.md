# Export Pipeline

Last updated: 2026-08-29

`ExportService` is the single capability and publication boundary. `GET /api/exports/{project_id}/capabilities` validates the current graph and returns only profile-compatible methods, runtime/dependency reasons, option definitions, payload/conversion badges, and limitations. The dedicated `/models/:modelId/export` page consumes that matrix and owns project/source downloads and binary export jobs. Compilation owns source generation and source inspection/copy; it does not download artifacts. Project JSON and generated source remain the canonical synchronous representations.

`POST /api/execution/jobs` accepts the clean-break `model_export` operation with a persisted project revision, server-validated filename stem, format, options, and immutable worker selection. `RuntimeController` discovers installed Keras and native APIs independently, skips absent frameworks, and reuses a matching environment snapshot; explicit backend management or a changed package/hardware/configuration fingerprint refreshes discovery. It then starts an isolated worker, reports preflight/building/serializing/validating/publishing phases, and keeps the prior published artifact when any check, cancellation, revision guard, or packaging step fails. Publication is staged and atomic; the returned `ExportArtifactManifest` contains deterministic file paths, hashes, signatures, dependency versions, and verification results.

The model profile determines the implementation: Keras-enabled projects use the Keras API against the selected TensorFlow, PyTorch, or JAX backend; native projects use raw TensorFlow, PyTorch, or JAX/Flax code. Keras is an API layer, not a fourth backend, and a native worker never falls back to Keras. TensorFlow and JAX native manifests are restricted to the verified common 27-node subset: inputs/outputs, dense, activation, dropout, flatten, embedding, batch/layer normalization, Conv2D, MaxPool2D, LSTM, GRU, add/residual/skip, and concatenate. Attention and transformer nodes remain Keras-only.

Supported artifact families are:

| Category | Formats | Contract |
| --- | --- | --- |
| Project/source | `.neuro-weave.json`, `.py` | Canonical graph/profile or generated source; no weights in the project bundle |
| Keras | `.keras`, `.weights.h5` | Keras-enabled profiles; initialized inference model or weights-only |
| Native | `.savedmodel.zip`, `.pt2`, `.pt` | TensorFlow SavedModel; PyTorch ExportedProgram; legacy trace-based TorchScript |
| Interchange | `.onnx`, `.onnx.zip` | Named inputs/outputs, default opset 18, dynamic batch, optional deterministic external-data package |
| Weights/checkpoint | `.safetensors.zip`, `.tf-checkpoint.zip`, `.flax-checkpoint.zip` | Explicit weights-only package with graph/profile manifest; not universally loadable |

All binary artifacts are initialized and untrained. No method claims optimizer or training-state preservation. ONNX uses a verified native PyTorch path, TensorFlow `tf2onnx` bridge, or Keras path; native JAX ONNX remains disabled until its `jax2tf`/`tf2onnx` bridge passes parity verification. Missing optional tooling disables only its affected capability with an actionable reason.

`.keras` architecture import remains separate from export. `POST /api/imports/keras-architecture` reads allowlisted configuration metadata, maps it to canonical nodes and edges, and never deserializes arbitrary Python, custom code, or weights.

Native PyTorch source and project bundles include the canonical compilation configuration so an export can be audited for optimizer, loss, metric, and target-contract intent. Binary artifacts remain initialized inference artifacts: compilation metadata is not training state, and export never performs an optimizer step or claims trained weights.
