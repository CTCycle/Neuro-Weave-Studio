# Transformer Layers

Last updated: 2026-07-30

Attention manifests include multi-head, self, and cross-attention, positional encoding, encoder and decoder blocks, feed-forward blocks, residual links, masks, and complete stacks. Attention inputs require rank three. Concrete embedding dimensions must be divisible by the number of heads.

The MVP Keras adapter implements multi-head attention, self-attention, sinusoidal positional encoding, and a pre-normalized Transformer encoder block. The bundled Transformer encoder example validates and generates Keras source. Decoder blocks, complete stacks, cross-attention, and PyTorch Transformer adapters remain cataloged design nodes with explicit compatibility diagnostics rather than identity fallbacks.
