# Serialization and Versioning

Last updated: 2026-08-27

The canonical graph is version 3 and the portable NeuroWeave project bundle is version 4. A graph stores one `model_configuration` value with explicit `keras` and `native` branches; native TensorFlow, PyTorch, and JAX compiler settings are independent and disabled by default. A project bundle also stores `execution_backend`, `keras_enabled`, the graph, and revision-independent export settings. Node definitions have independent stable IDs and integer versions. Readers accept only the current graph and bundle versions and reject unsupported future versions; the forward Alembic migration `20260827_canonical_model_configuration` moves the previous flattened settings columns into the canonical JSON value. There are no dual reads, aliases, or compatibility shims. Project revisions are application history, not schema versions.

Generated source snapshots retain the model execution profile used for the current revision. A `.keras` binary is an initialized artifact, not a portable project bundle. A `.keras` architecture import reads only allowlisted configuration metadata and creates a new canonical graph; weights, optimizer state, and custom Python objects are not serialized into the NeuroWeave project.
