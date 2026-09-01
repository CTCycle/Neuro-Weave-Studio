# Merge Operations

Last updated: 2026-09-01

Merge nodes include arithmetic elementwise operations, concatenate, stack, split, average, maximum, minimum, dot product, matrix multiplication, residual, and skip connections. Elementwise merges require compatible shapes. Concatenation requires equal rank and compatible non-concatenated dimensions; shape inference computes the selected axis size when concrete. Matrix multiplication follows the batch-first rank contract used by generated framework code, and incompatible rank/inner dimensions are blocking validation errors.
