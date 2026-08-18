# Training Configuration

Last updated: 2026-08-18

Optimizer, loss, metric, learning-rate, regularizer, initializer, constraint, random-seed, and mixed-precision concepts are represented in model-level configuration panels, not the tensor graph. This prevents configuration edges from being mistaken for tensors and supports multiple output heads through `__default__`, output-ID, and output-label mappings.

For the Keras target, validated optimizer, loss, metric, initializer, regularizer, constraint, seed, and mixed-precision settings are applied when source or binary models are built. Dataset handling, `fit`, training loops, and weight updates remain out of scope; the future-training configuration is persisted metadata only.
