# Shape Inference

Last updated: 2026-07-30

Shape inference traverses nodes in topological order and maps manifest shape-rule kinds to deterministic functions. It currently covers inputs, dense, flatten, reshape, convolution rank checks, pooling passthrough, recurrent layers, embeddings, merge compatibility, concatenation, and attention divisibility. Unknown or partly symbolic dimensions remain unknown. Failed inference is a blocking diagnostic, not a guessed result.
