# Troubleshooting

Last updated: 2026-08-17

A blank catalog usually means node JSON failed Pydantic validation; inspect the backend log under `app/resources/runtime/logs`. If a configured port is occupied, the launcher reports the listener PID and leaves that process untouched. Save conflicts return HTTP 409 and require reopening before overwrite. Keras binary export returning structured HTTP 409 `backend_unavailable` means the requested TensorFlow, PyTorch, or JAX backend is not operational; select an installed backend or install one through the launcher. A 422 binary export response means the request or graph is invalid. Invalid or missing backend request bodies are standard FastAPI 422 responses.

If startup reports that the configured SQLite database does not contain the canonical NeuroWeave schema, the database uses the previous schema or is incomplete. Select **Initialize or recreate database**, type `RESET`, and confirm that the configured local data may be discarded. There is no automatic migration or compatibility reader.
