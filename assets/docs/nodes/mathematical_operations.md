# Mathematical Operations

Last updated: 2026-09-12

The catalog includes absolute value, exponential, logarithm, power, square root, softmax, argmax, reductions, slicing, indexing, transpose, stop-gradient, and a safe expression node.

Stop Gradient is a shape-preserving gradient barrier: the forward tensor is unchanged while gradient propagation through that graph path is disabled. It is implemented with the backend-neutral Keras operation and is available with the supported Keras execution backends. Direct native-source mappings remain unavailable until backend-specific gradient semantics are covered by native compilation tests.

The safe expression validator parses expressions with Python's AST module but never executes them. The allowed variable is `x`. Allowed functions are `abs`, `clip`, `exp`, `log`, `maximum`, `minimum`, `pow`, `softmax`, and `sqrt`. Arithmetic, unary arithmetic, comparisons, and boolean operators are accepted. Attribute access, imports, subscripts, comprehensions, lambdas, assignment, string literals, keyword arguments, and unknown names are rejected.

Mathematical nodes that do not have a verified exporter mapping remain available for architecture design but report a blocking code-target compatibility diagnostic during source generation.
