# Serialization and Versioning

Last updated: 2026-08-21

The canonical graph and portable NeuroWeave project bundle are both version 2. A project bundle stores `execution_backend`, `keras_enabled`, the graph, model-level compilation configuration, and revision-independent export settings. Node definitions have independent stable IDs and integer versions. Readers accept only the current graph and bundle versions and reject unsupported future versions; there is no version-1 migration reader. Project revisions are application history, not schema versions.

Generated source snapshots retain the model execution profile used for the current revision. A `.keras` binary is an initialized artifact, not a portable project bundle. A `.keras` architecture import reads only allowlisted configuration metadata and creates a new canonical graph; weights, optimizer state, and custom Python objects are not serialized into the NeuroWeave project.
