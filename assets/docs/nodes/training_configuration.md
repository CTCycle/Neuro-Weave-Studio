# Training Configuration

Last updated: 2026-08-21

Optimizer, loss, metric, learning-rate, regularizer, initializer, constraint, random-seed, and mixed-precision concepts are represented in model-level configuration panels, not the tensor graph. This prevents configuration edges from being mistaken for tensors and supports multiple output heads through `__default__`, output-ID, and output-label mappings.

When `keras_enabled=true`, validated optimizer, loss, metric, initializer, regularizer, constraint, seed, and mixed-precision settings are applied when Keras source or binary models are built. When `keras_enabled=false`, native implementations do not consume these Keras compilation settings. Dataset handling, `fit`, training loops, and weight updates remain out of scope; the future-training configuration is persisted metadata only.

The Builder presents this configuration as a compact compilation summary with a metadata-driven modal. The backend catalog is authoritative for supported values, parameter ranges, and trusted formula metadata; Apply updates the graph configuration once. This is an extension point for future training and does not imply that training is implemented.
