# Framework Compatibility

Last updated: 2026-08-18

A node definition contains declarative mappings for the v2 code targets `keras`, `tensorflow`, `pytorch`, and `jax`. Keras 3 is the model API; the latter three are native ecosystem targets. The checked-in manifest is the source of target and parameter capability truth; the loader must not promote nodes into a portable set. Native TensorFlow and JAX/Flax node capabilities are currently disabled pending runtime parity tests. The native PyTorch core currently includes dense, dropout, flatten, embedding, LSTM/GRU, Conv2D, MaxPool2D, rank-aware BatchNorm/LayerNorm, and common add/residual/skip/concatenate merges. Unsupported node/parameter combinations produce typed compatibility diagnostics and block compilation; there is no identity fallback.

The 2026-08-18 local parity audit found Keras 3.15.1 and PyTorch 2.13.0+cpu in `app/server/.venv`, while TensorFlow, JAX, and Flax were not importable (`dependency_missing` from the isolated worker probes). Native TensorFlow and JAX therefore remain disabled for all manifest nodes; no source-target capability is promoted from source inspection alone. The complete inventory and commands are recorded in `assets/QA/native-target-audit-2026-08-18.md`.
