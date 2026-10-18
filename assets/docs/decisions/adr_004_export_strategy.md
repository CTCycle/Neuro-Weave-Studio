# ADR 004: Export Strategy

Last updated: 2026-10-18

Status: accepted. NeuroWeave JSON is the canonical project representation. It stores the graph, model-level configuration, and the execution profile (`execution_backend` plus `keras_enabled`). Generated source and binary export are derived from that profile: Keras-enabled models use the Keras API against the persisted backend, while Keras-disabled models use a verified raw native exporter. The compiler, validator, and workers must never reinterpret Keras as a fourth backend or fall back from an unavailable native implementation.

Export is capability-gated through `ExportService` and `GET /api/exports/{project_id}/capabilities`. The accepted matrix includes project/source files, `.keras`, `.weights.h5`, TensorFlow SavedModel packages, ONNX, PyTorch ExportedProgram, SafeTensors, TensorFlow checkpoints, and Flax checkpoints. Each artifact is staged, structurally checked, reloaded where applicable, forward-pass verified, recorded in a manifest, and published atomically only if the project revision is unchanged. External data and directories are deterministic ZIP packages; partial output must never replace a previous artifact.

Native TensorFlow, PyTorch, and JAX/Flax capabilities are limited to the verified subset declared by the catalog. Advanced convolutional/recurrent variants, attention, and transformer nodes remain Keras-only until native adapters are verified. ONNX bridges are enabled only after verification; native JAX ONNX is currently withheld while its `jax2tf`/`tf2onnx` path is unverified. All outputs are initialized and untrained; weights/checkpoint methods are explicitly weights-only and no method preserves optimizer or training state. `.h5` whole-model, pickle-based `torch.save`, OpenVINO/LiteRT, arbitrary cross-framework conversion, and compatibility aliases for the removed `keras_export` operation are excluded.

`.keras` architecture import remains a separate allowlisted metadata path and never executes arbitrary Python or imports weights. Future training workflows may attach after this boundary without changing the export contract.
