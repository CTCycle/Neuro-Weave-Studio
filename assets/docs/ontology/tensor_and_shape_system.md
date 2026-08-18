# Tensor and Shape System

Last updated: 2026-08-18

Tensor Specification contains an ordered shape, data type, and optional name. A dimension is a positive integer, an unknown value, or a symbolic dimension. Rank is the number of dimensions. Tensor compatibility requires matching dtypes and ranks; unknown dimensions remain compatible unless a concrete contradiction is known. The leading dimension conventionally represents batch size. Shape inference never treats `-1` as a concrete positive dimension in persisted canonical output.
