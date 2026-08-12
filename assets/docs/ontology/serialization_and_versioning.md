# Serialization and Versioning

Last updated: 2026-08-12

The canonical graph and portable project bundle are both version 2. Node definitions have independent stable IDs and integer versions. Readers accept only the current graph and bundle versions and reject unsupported future versions; there is no version-1 migration reader. Project revisions are application history, not schema versions.
