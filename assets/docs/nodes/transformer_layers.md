# Transformer Layers

Last updated: 2026-09-01

Attention manifests include multi-head, self, and cross-attention, positional encoding, encoder and decoder blocks, feed-forward blocks, residual links, masks, and complete stacks. Attention inputs require rank three. Concrete embedding dimensions must be divisible by the number of heads.

The Keras adapter implements multi-head attention, self-attention, cross-attention, sinusoidal positional encoding, residual/mask paths, feed-forward blocks, encoder and decoder blocks, and stacks. These nodes expose conditional parameters and preserve multi-port relationships in the canonical graph. Native TensorFlow, PyTorch, and JAX mappings currently report explicit compatibility diagnostics for transformer nodes; they do not fall back to identity behavior.
