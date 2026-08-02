# Mathematical Operations

Last updated: 2026-07-30

The catalog includes absolute value, exponential, logarithm, power, square root, softmax, argmax, reductions, slicing, indexing, transpose, and a safe expression node.

The safe expression validator parses expressions with Python's AST module but never executes them. The allowed variable is `x`. Allowed functions are `abs`, `clip`, `exp`, `log`, `maximum`, `minimum`, `pow`, `softmax`, and `sqrt`. Arithmetic, unary arithmetic, comparisons, and boolean operators are accepted. Attribute access, imports, subscripts, comprehensions, lambdas, assignment, string literals, keyword arguments, and unknown names are rejected.

Mathematical nodes that do not have a verified exporter mapping remain available for architecture design but report a blocking framework compatibility diagnostic during source generation.
