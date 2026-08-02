# Core Layers

Last updated: 2026-07-30

Core manifests cover Dense or Linear, Activation, Dropout, batch, layer, and group normalization, Flatten, Reshape, Permute, Repeat, and Embedding. Dense replaces only the final dimension and can count parameters when input features are concrete. Reshape and permutation require stricter production inference before arbitrary export support is expanded.
