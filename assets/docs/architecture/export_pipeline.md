# Export Pipeline

Last updated: 2026-08-20

Project JSON export is always available for a valid schema. Source export requires graph and code-target compatibility validation. Portable Keras uses the backend-neutral Keras 3 API; native PyTorch has an explicit validated exporter, while native TensorFlow and JAX/Flax source capabilities remain disabled until runtime parity is proven. Keras is the fixed model abstraction, not a fourth execution backend.

The core dependency is `keras`; `tensorflow`, `torch`, and the bounded `jax`/`flax` extra are selected by the launcher. `RuntimeController` probes each execution backend in a short-lived worker and starts a new worker for every export. `POST /api/execution/jobs` carries a discriminated `keras_export` operation plus immutable runtime selection, so concurrent jobs do not share mutable server-wide backend state. Unknown policies, unavailable API/backend/device, occupied leases, missing projects, and incompatible graphs return structured submission errors; accepted failures remain on the job record.

Native PyTorch source export records inferred `TensorSpec` summaries in the canonical IR. The generated model derives dense/recurrent feature widths from connected shapes, uses explicit NHWC/NCHW conversion around Conv2D and pooling, and supports rank-aware normalization, embedding, dropout, and common merge nodes. Recurrent `input_size` is derived from the connected tensor and is not a user-authored v2 parameter.

Binary `.keras` export creates an initialized, untrained Keras model through the documented supported subset. The worker compiles the model from validated configuration, saves to a unique temporary artifact, reloads it with `keras.saving.load_model(..., compile=True)`, runs inference, and the controller atomically replaces the destination only after those checks and a final project-revision check succeed. `.h5`, weights, trained artifacts, ONNX, cross-framework conversion, and native-framework inference are not claimed by the MVP.
