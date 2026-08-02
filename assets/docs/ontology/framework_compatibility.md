# Framework Compatibility

Last updated: 2026-07-30

A node definition contains mappings for `keras`, `tensorflow_keras`, and `pytorch`. Compatibility is evaluated per node and whole graph. Keras and TensorFlow-through-Keras share generated Keras APIs but remain separate target declarations because runtime and packaging expectations can differ. Torch is not a separate backend from PyTorch. Unsupported nodes produce compatibility diagnostics and block compilation.
