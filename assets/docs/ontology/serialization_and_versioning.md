# Serialization and Versioning

Last updated: 2026-07-30

Canonical graph schema version starts at 1. Node definitions have independent stable IDs and integer versions. Portable bundle version also starts at 1. Readers must reject unsupported future bundle versions. Schema migrations must be explicit, deterministic, tested on saved fixtures, and never silently discard parameters, positions, ports, or configuration. Project revisions are application history, not schema versions.
