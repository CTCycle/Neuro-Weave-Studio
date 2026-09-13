# Graph Compiler

Last updated: 2026-09-13

Compilation starts only after the server validates the complete graph against the exact selected one of the six execution profiles. The compiler topologically orders the DAG and creates a canonical IR containing stable node identities, sanitized variable names, ordered input dependencies, parameters, model inputs, and model outputs. Exporters consume this IR only after validation succeeds; invalid nodes, parameters, connections, or compilation settings produce no source or artifact. Framework symbols are not stored in node instances, which prevents TensorFlow, Keras, or PyTorch concerns from leaking into the project graph. Native exporters are selected by the exact native backend and never fall back to a Keras exporter.
