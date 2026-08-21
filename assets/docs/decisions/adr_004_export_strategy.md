# ADR 004: Export Strategy

Last updated: 2026-08-21

Status: accepted. NeuroWeave JSON is the canonical project representation. It stores the graph, model-level configuration, and the execution profile (`execution_backend` plus `keras_enabled`). Generated source is derived from that profile: Keras-enabled models use the Keras exporter against the persisted backend, while Keras-disabled models use a verified native exporter such as the current PyTorch subset. The compiler and validator must never reinterpret Keras as a fourth backend or fall back from an unavailable native implementation.

Binary export is opt-in and capability-gated. `.keras` is implemented for a restricted initialized and untrained subset and is verified by reload and inference before publication. `.keras` architecture import is a separate source-only path: it reads allowlisted configuration metadata, creates canonical nodes and edges, and does not import weights or custom Python. `.h5`, standalone weights, trained artifacts, PyTorch full-object pickle, arbitrary cross-framework conversion, and native-framework runtime inference are not default outputs.

Native TensorFlow and JAX/Flax source capabilities remain unavailable until dependency-backed exporter and runtime parity is proven. Future training and inference attach after static validation and the Keras/native adapter boundary; this ADR does not authorize datasets, training loops, checkpoints, or experiment tracking.
