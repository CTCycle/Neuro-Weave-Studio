# Testing Strategy

Last updated: 2026-08-05

Backend unit tests cover manifest loading, DAG validation, shape propagation, parameter counts, serialization, typed OpenAPI schemas, required revisions, service-only dependencies, and Keras error contracts. Ruff and strict Pyright run over server, scripts, and backend tests. Keras binary artifact tests run for each installed operational backend and are explicitly skipped when no TensorFlow/Torch extra is installed. Frontend tests cover graph-cycle logic, catalog filtering, and zero/one/multiple backend selector states; the local UI zero-backend state is verified through the in-app Browser.
