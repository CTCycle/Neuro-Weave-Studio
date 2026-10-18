# Export Pipeline

Last updated: 2026-10-18

`ExportService` is the single capability and publication boundary. `GET /api/exports/{project_id}/capabilities` validates the current graph and returns only profile-compatible methods, runtime/dependency reasons, option definitions, payload/conversion badges, and limitations. The dedicated `/models/:modelId/export` page consumes that matrix and owns project/source downloads and binary export jobs. Compilation owns source generation and source inspection/copy; it does not download artifacts. Project JSON and generated source remain the canonical synchronous representations. Revision-scoped verification and open coverage boundaries are summarized in the [project status ledger](../project_status_ledger.md). Revision-scoped verification and open coverage boundaries are summarized in the [project status ledger](../project_status_ledger.md).

Capability availability and job validation are the same contract: both use the saved graph revision, exact `execution_backend`, exact Keras/native API, runtime requirements, and server-owned option choices. An invalid graph makes source and artifact capabilities unavailable and prevents job creation; a worker with the wrong backend or API is rejected before the controller can create a job or publish an artifact. Unavailable runtime/compiler and dependency reasons are displayed, not converted into a fallback path.

`POST /api/execution/jobs` accepts the clean-break `model_export` operation with a persisted project revision, server-validated filename stem, format, options, and immutable worker selection. The UI completes on-demand discovery for the persisted Keras/native API pair before submission; direct requests with an unchecked pair start or join discovery and return structured `runtime_discovery_pending`. `RuntimeController` probes only the requested API, skips absent frameworks, and validates or reuses a matching environment snapshot; explicit refresh or a changed package/hardware/configuration fingerprint refreshes discovery. It then starts an isolated worker, reports preflight/building/serializing/validating/publishing phases, and keeps the prior published artifact when any check, cancellation, revision guard, or packaging step fails. Publication is staged and atomic; the returned `ExportArtifactManifest` contains deterministic file paths, hashes, signatures, dependency versions, and verification results.

The model profile determines the implementation: Keras-enabled projects use the Keras API against the selected TensorFlow, PyTorch, or JAX backend; native projects use raw TensorFlow, PyTorch, or JAX/Flax code. Keras is an API layer, not a fourth backend, and a native worker never falls back to Keras. Verified native mappings cover the common input/layer, mathematical, merge, and output-adapter subset declared by the catalog. Advanced convolutional variants, recurrent variants, attention, and transformer nodes remain explicitly Keras-only until their native adapters are verified.

Supported artifact families are:

| Category | Formats | Contract |
| --- | --- | --- |
| Project/source | `.neuro-weave.json`, `.py` | Canonical graph/profile or generated source; no weights in the project bundle |
| Keras | `.keras`, `.weights.h5` | Keras-enabled profiles; initialized model or weights-only |
| Native | `.savedmodel.zip`, `.pt2` | TensorFlow SavedModel; PyTorch ExportedProgram |
| Interchange | `.onnx`, `.onnx.zip` | Named inputs/outputs, default opset 18, dynamic batch, optional deterministic external-data package |
| Weights/checkpoint | `.safetensors.zip`, `.tf-checkpoint.zip`, `.flax-checkpoint.zip` | Explicit weights-only package with graph/profile manifest; not universally loadable |

All binary artifacts are initialized and untrained. No method claims optimizer or training-state preservation. ONNX uses verified native PyTorch and direct native JAX (`jax2onnx`) paths, the TensorFlow `tf2onnx` bridge, or the Keras path. Native JAX export is available when `jax2onnx` and the ONNX runtime dependencies are installed. Missing optional tooling disables only its affected capability with an actionable reason.

`.keras` architecture import remains separate from export. `POST /api/imports/keras-architecture` reads allowlisted configuration metadata, maps it to canonical nodes and edges, and never deserializes arbitrary Python, custom code, or weights.

Native PyTorch source and project bundles include the canonical compilation configuration so an export can be audited for optimizer, loss, metric, and target-contract intent. Binary artifacts remain initialized model artifacts: compilation metadata is not training state, and export never performs an optimizer step or claims trained weights.
