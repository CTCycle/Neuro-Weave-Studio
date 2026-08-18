# Shape Inference

Last updated: 2026-08-18

Shape inference traverses nodes in topological order and maps manifest shape-rule kinds to deterministic functions. It currently covers inputs, dense, flatten, reshape, convolution rank checks, pooling defaults and data formats, recurrent layers, embeddings, merge compatibility, concatenation axes, and Keras attention dimensions. Multi-port nodes use manifest port order, and graph validation rejects components that are not on an input-to-output path. Unknown or partly symbolic dimensions remain unknown. Failed inference is a blocking diagnostic, not a guessed result.
