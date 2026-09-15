# Core Layers

Last updated: 2026-09-12

Core manifests cover Dense or Linear, Activation, Parametric ReLU, Dropout, batch, layer, group, RMS, and unit normalization, Flatten, Reshape, Permute, Repeat, and Embedding. Keras mappings expose the serializable constructor surface, including initializers, regularizers, constraints, dtype, trainability, and applicable advanced controls. Dense replaces only the final dimension and can count parameters when input features are concrete. Shape-changing rules retain unknown dimensions when static inference is not possible; native availability remains target-specific and is surfaced as a compatibility diagnostic.

RMS Normalization, Parametric ReLU, and Unit Normalization are verified through the Keras API on the supported TensorFlow, PyTorch, and JAX execution backends. Their direct native-source adapters remain explicitly unavailable until backend-specific parity is verified; no native fallback is used.
