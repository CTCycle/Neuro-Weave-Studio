# Validation Rules

Last updated: 2026-07-30

Blocking errors include cycles, missing model inputs or outputs, missing required connections, dangling edges, unknown ports or definitions, duplicate connection contracts, conflicting outputs, invalid parameters, explicit nulls for non-nullable parameters, undeclared v2 parameters, tensor-rank contradictions, merge incompatibility, residual-shape mismatch, attention-dimension mismatch, unsafe mathematical expressions, and unsupported framework features.

Warnings identify suspicious but potentially compilable graphs, including unknown optional parameters and incomplete shape information. Compatibility notices describe target-specific limitations. Informational diagnostics carry model summaries such as inferred tensor shapes and parameter counts.

The representative inference subset covers inputs, identity layers, dense layers, reshape, flatten, embedding, one-dimensional through three-dimensional convolution, pooling, global pooling, recurrent layers, attention, Transformer encoder blocks, and common merges. Unknown dimensions propagate symbolically or as `null` rather than being fabricated.
