# Mathematical Operations

Last updated: 2026-09-12

The catalog includes absolute value, exponential, logarithm, power, square root, softmax, argmax, reductions, slicing, indexing, transpose, stop-gradient, and a safe expression node.

Stop Gradient is a shape-preserving gradient barrier: the forward tensor is unchanged while gradient propagation through that graph path is disabled. Keras source uses `keras.ops.stop_gradient`; direct native source uses `tf.stop_gradient`, `torch.detach`, or `jax.lax.stop_gradient` for TensorFlow, PyTorch, and JAX respectively. These mappings remain part of the canonical manifest contract and are covered by operation-source tests.

The safe expression validator parses expressions with Python's AST module but never executes them. The allowed variable is `x`. Allowed functions are `abs`, `clip`, `exp`, `log`, `maximum`, `minimum`, `pow`, `softmax`, and `sqrt`. Arithmetic, unary arithmetic, comparisons, and boolean operators are accepted. Attribute access, imports, subscripts, comprehensions, lambdas, assignment, string literals, keyword arguments, and unknown names are rejected.

Mathematical nodes that do not have a verified exporter mapping remain available for architecture design but report a blocking code-target compatibility diagnostic during source generation.
