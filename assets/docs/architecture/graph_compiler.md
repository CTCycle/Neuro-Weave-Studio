# Graph Compiler

Last updated: 2026-07-30

Compilation starts only after validation. The compiler topologically orders the DAG and creates a canonical IR containing stable node identities, sanitized variable names, ordered input dependencies, parameters, model inputs, and model outputs. Exporters consume this IR. Framework symbols are not stored in node instances, which prevents TensorFlow, Keras, or PyTorch concerns from leaking into the project graph.
