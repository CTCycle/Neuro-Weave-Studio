# Export Pipeline

Last updated: 2026-08-05

Project JSON export is always available for a valid schema. Source export requires graph and framework compatibility validation. Keras and TensorFlow-through-Keras use the Keras functional API adapter. PyTorch generates an `nn.Module` directly through its API. Keras is a layer over its configured backend, not a fourth runtime.

The core dependency is `keras`; `tensorflow` and `torch` are bounded optional extras selected by the launcher. `KerasBackendRunner` probes each installed backend during application lifespan and performs binary construction in a dedicated worker process with an explicit `KERAS_BACKEND` environment value before importing Keras. The binary export request carries its backend, so concurrent exports do not share mutable server-wide backend state. A valid but unavailable backend returns HTTP 501; an incompatible graph returns HTTP 422.

Binary `.keras` export creates an initialized, untrained Keras model through the documented supported subset. `.h5`, weights, trained artifacts, ONNX, and cross-framework conversion are not claimed by the MVP.
