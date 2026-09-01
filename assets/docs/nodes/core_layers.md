# Core Layers

Last updated: 2026-09-01

Core manifests cover Dense or Linear, Activation, Dropout, batch, layer, and group normalization, Flatten, Reshape, Permute, Repeat, and Embedding. Keras mappings expose the serializable constructor surface, including initializers, regularizers, constraints, dtype, trainability, and applicable advanced controls. Dense replaces only the final dimension and can count parameters when input features are concrete. Shape-changing rules retain unknown dimensions when static inference is not possible; native availability remains target-specific and is surfaced as a compatibility diagnostic.
