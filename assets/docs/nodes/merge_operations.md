# Merge Operations

Last updated: 2026-07-30

Merge nodes include arithmetic elementwise operations, concatenate, stack, split, average, maximum, minimum, dot product, matrix multiplication, residual, and skip connections. Elementwise merges require compatible shapes. Concatenation requires equal rank and compatible non-concatenated dimensions; the current MVP validates rank and computes the selected axis size when concrete.
