# ADR 004: Export Strategy

Last updated: 2026-07-30

Status: accepted. JSON project export is canonical. Generated Python is the primary framework artifact. Binary export is opt-in and capability-gated. `.keras` is implemented for a restricted initialized and untrained subset. `.h5`, PyTorch full-object pickle, weights without training, and arbitrary cross-framework conversion are not default outputs. Future conversion should use a validated intermediate such as ONNX only where operator compatibility is proven.
