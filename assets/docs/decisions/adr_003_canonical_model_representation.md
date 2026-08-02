# ADR 003: Canonical Model Representation

Last updated: 2026-07-30

Status: accepted. Projects persist a versioned framework-independent graph. Node instances reference manifests and contain only IDs, positions, labels, and serializable parameters. Edges reference stable port IDs. Framework adapters consume validated graphs through canonical IR. No Keras Layer, TensorFlow Operation, PyTorch Module, executable callable, or serialized framework object appears in the project schema.
