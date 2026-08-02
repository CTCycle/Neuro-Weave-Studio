# Projects and Exports

Last updated: 2026-08-02

## Save and reopen

Saving creates or updates the project’s current document and appends an
immutable revision snapshot. Reopening reads the current persisted graph and
restores its framework, validation status, metadata, and generated source
snapshots.

The update request may include `expected_revision`. If another save has already
advanced the project, the backend returns HTTP 409. Reopen the project, review
the newer state, and save again; do not retry blindly.

## Duplicate and portable JSON

Duplicate creates an independent project with a new identifier. JSON export is
portable and contains a format identifier, bundle version, export timestamp, and
the project creation payload. Import creates a new project from that bundle and
does not copy database revision identifiers.

## Source and binary exports

Saved projects can export Keras/TensorFlow-through-Keras or PyTorch source. The
optional `.keras` endpoint creates an initialized, untrained model only when
TensorFlow and Keras are installed and the graph uses the documented binary
subset. It does not contain learned weights and is not a trained model.

Generated artifacts belong under the configured export root. Keep exported code
and binaries under normal project or release review; the database is the source
of truth for project metadata and revision history.
