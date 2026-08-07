# Testing Strategy

Last updated: 2026-08-07

Backend unit tests cover manifest loading, DAG validation, shape propagation, parameter counts, serialization, typed OpenAPI schemas, enriched catalog summaries, revision-specific compiled targets, required revisions, service-only dependencies, code-target compatibility, backend availability contracts, and Keras error contracts. Ruff and strict Pyright run over server, scripts, and backend tests. Keras binary artifact tests run for each installed operational TensorFlow, PyTorch, or JAX backend and are explicitly skipped when no optional runtime is installed. Frontend tests cover graph-cycle logic, focused Builder commands, catalog filtering, route-level workspace states, and zero/one/multiple backend selector states; the local UI route and responsive states are verified through the in-app Browser.
