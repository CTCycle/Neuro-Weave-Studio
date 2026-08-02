# Configuration

Last updated: 2026-08-02

Configuration is loaded through `pydantic-settings` from environment variables
and `settings/.env`. The launcher copies `settings/.env.example` only when
`settings/.env` is absent; it never overwrites an existing local file.

## Database

```dotenv
EMBEDDED_DATABASE=true
DATABASE_URL=
DATABASE_ENGINE=postgresql+psycopg
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=neuro_weave
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=
DATABASE_SSL=false
DATABASE_SSL_CA=
DATABASE_CONNECT_TIMEOUT=30
DATABASE_INSERT_BATCH_SIZE=1000
DATABASE_SQLITE_PATH=
```

`EMBEDDED_DATABASE=true` uses SQLite. A missing SQLite file is initialized on
startup. Existing SQLite files are checked but not modified. Set it to `false`
for PostgreSQL; `DATABASE_URL` can seed connection fields and explicit
`DATABASE_*` values take precedence. PostgreSQL schema creation or reset is
explicit through the initializer.

`DATABASE_SQLITE_PATH` is optional and is useful for isolated QA databases. The
default is `app/resources/runtime/neuro_weave.db`.

## Application and frontend

The remaining settings retain the `NEURO_WEAVE_` namespace:

- `NEURO_WEAVE_CORS_ORIGINS`
- `NEURO_WEAVE_LOG_LEVEL`
- `NEURO_WEAVE_EXPORT_ROOT`
- `VITE_API_BASE_URL` (normally `/api`)

Relative SQLite and export paths are anchored to the repository root, not the process working directory. This keeps the application, launcher, initializer, and manual startup commands on one canonical runtime location. The default database is `app/resources/runtime/neuro_weave.db`; generated artifacts use `app/resources/runtime/exports`. The default CORS origins are `http://127.0.0.1:5173` and `http://localhost:5173`.
