# Training Configuration

Last updated: 2026-08-29

Optimizer, loss, metric, learning-rate, regularizer, initializer, constraint, random-seed, and mixed-precision concepts are represented in the Keras branch of one model-level configuration value, not the tensor graph. The same canonical value contains explicit native compiler branches for TensorFlow, PyTorch, and JAX. This prevents configuration edges from being mistaken for tensors and supports multiple output heads through `__default__`, output-ID, and output-label mappings.

When `keras_enabled=true`, validated optimizer, loss, metric, initializer, regularizer, constraint, seed, and mixed-precision settings are applied when Keras source or binary models are built. When `keras_enabled=false`, native implementations consume only their selected backend branch: TensorFlow `jit_compile`, PyTorch `torch.compile` options, or JAX `jit_enabled`. Environment device, memory, and precision preferences remain session-scoped worker configuration. Dataset handling, `fit`, training loops, and weight updates remain out of scope; the future-training configuration is persisted metadata only.

The Compilation page presents this configuration through a capability-driven, metadata-driven modal. The backend capability tree is authoritative for supported values, parameter ranges, compiler availability, and trusted formula metadata; Apply validates the proposed graph/profile and performs one revision-aware replacement of the canonical graph configuration. This is an extension point for future training and does not imply that training is implemented.

For native PyTorch, the same Compilation modal exposes optimizer parameters plus output-selected losses and metrics from the filtered server catalog. Generated helpers resolve those selectors to trusted pure-torch implementations and report compilation metadata during isolated smoke execution. The helpers verify construction and finite evaluation only: no optimizer step, gradient update, dataset, or training loop is performed.
