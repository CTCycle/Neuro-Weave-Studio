# Export Pipeline

Last updated: 2026-08-05

Project JSON export is always available for a valid schema. Source export requires graph and code-target compatibility validation. Portable Keras uses the backend-neutral Keras 3 API; native TensorFlow, PyTorch, and JAX/Flax targets use explicit native exporters. Keras is the fixed model abstraction, not a fourth execution backend.

The core dependency is `keras`; `tensorflow`, `torch`, and the bounded `jax`/`flax` extra are selected by the launcher. `KerasBackendRunner` probes each execution backend during application lifespan in an isolated worker with `KERAS_BACKEND` set before importing Keras. The binary export request carries `execution_backend`, so concurrent exports do not share mutable server-wide backend state. An unknown backend returns HTTP 422; a known but unavailable backend returns structured HTTP 409 `backend_unavailable`; an incompatible graph returns HTTP 422.

Binary `.keras` export creates an initialized, untrained Keras model through the documented supported subset. `.h5`, weights, trained artifacts, ONNX, and cross-framework conversion are not claimed by the MVP.
