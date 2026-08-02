# Tensor and Shape System

Last updated: 2026-07-30

Tensor Specification contains an ordered shape, data type, and optional name. A dimension is a positive integer, an unknown value, or a symbolic dimension. Rank is the number of dimensions. Unknown dimensions are compatible unless a concrete contradiction is known. The leading dimension conventionally represents batch size. Shape inference never treats `-1` as a concrete positive dimension in persisted canonical output.
