# Training Configuration

Last updated: 2026-07-30

Optimizer, loss, metric, learning-rate, regularizer, initializer, constraint, and mixed-precision concepts are represented in model-level configuration panels, not the tensor graph. This prevents configuration edges from being mistaken for tensors and supports multiple output heads through output-name mappings. Training execution is not implemented; the persisted configuration is an extension boundary.
