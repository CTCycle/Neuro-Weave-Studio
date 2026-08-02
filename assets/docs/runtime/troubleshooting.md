# Troubleshooting

Last updated: 2026-07-30

A blank catalog usually means node JSON failed Pydantic validation; inspect backend startup logs. Save conflicts return HTTP 409 and require reopening before overwrite. Keras export returning 501 means optional dependencies are absent or a node is outside the binary subset. A 422 compilation response means validation or compatibility failed. Delete `runtime/neuro_weave.db` only when intentionally resetting local data.
