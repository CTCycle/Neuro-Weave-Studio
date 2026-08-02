from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from neuro_weave.config import DatabaseSettings, get_database_settings
from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import ModelGraph
from neuro_weave.paths import REPOSITORY_ROOT
from neuro_weave.persistence.backend import (
    build_database_backend,
    build_postgres_connect_args,
    build_postgres_url,
)
from neuro_weave.persistence.initializer import (
    ensure_database_ready,
    initialize_database,
)
from neuro_weave.persistence.models import ProjectRecord
from neuro_weave.persistence.repositories import ProjectRepository
from neuro_weave.persistence.session import RepositoryContext, build_session_factory
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session


def _clear_database_cache() -> None:
    get_database_settings.cache_clear()


def test_embedded_database_settings_resolve_relative_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    monkeypatch.setenv("DATABASE_SQLITE_PATH", "assets/QA/contract.db")
    _clear_database_cache()

    settings = get_database_settings()

    assert settings.embedded_database is True
    assert settings.sqlite_path == (REPOSITORY_ROOT / "assets/QA/contract.db").resolve()


def test_embedded_database_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "sometimes")
    _clear_database_cache()

    with pytest.raises(ValueError, match="must be boolean"):
        get_database_settings()


def test_ssl_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "true")
    monkeypatch.setenv("DATABASE_SSL", "maybe")
    _clear_database_cache()

    with pytest.raises(ValueError, match="must be boolean"):
        get_database_settings()


def test_external_database_url_is_overridden_by_explicit_values(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://url_user:url%20password@url.example:6543/url_db?sslmode=require&connect_timeout=17",
    )
    monkeypatch.setenv("DATABASE_HOST", "explicit.example")
    monkeypatch.setenv("DATABASE_NAME", "explicit_db")
    monkeypatch.setenv("DATABASE_USERNAME", "explicit_user")
    _clear_database_cache()

    settings = get_database_settings()

    assert settings.embedded_database is False
    assert settings.engine == "postgresql"
    assert settings.host == "explicit.example"
    assert settings.port == 6543
    assert settings.database_name == "explicit_db"
    assert settings.username == "explicit_user"
    assert settings.password == "url password"
    assert settings.ssl is True
    assert settings.connect_timeout == 17


def test_external_database_requires_connection_identity(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    for key in (
        "DATABASE_URL",
        "DATABASE_HOST",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
    ):
        monkeypatch.delenv(key, raising=False)
    _clear_database_cache()

    with pytest.raises(ValueError, match="requires DATABASE_HOST"):
        get_database_settings()


def test_old_namespaced_database_key_is_not_a_compatibility_path(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDED_DATABASE", raising=False)
    monkeypatch.setenv("NEURO_WEAVE_DATABASE_URL", "postgresql://legacy.example/legacy")
    monkeypatch.delenv("DATABASE_SQLITE_PATH", raising=False)
    _clear_database_cache()

    assert get_database_settings().embedded_database is True


def test_postgres_url_and_connect_args_escape_credentials() -> None:
    settings = DatabaseSettings(
        embedded_database=False,
        engine="postgresql+psycopg",
        host="db.example",
        port=5432,
        database_name="neuro_weave",
        username="user@example",
        password="p@ss word",
        ssl=True,
        ssl_ca="C:/certs/root.pem",
        connect_timeout=19,
    )

    assert build_postgres_url(settings) == (
        "postgresql+psycopg://user%40example:p%40ss+word@db.example:5432/neuro_weave"
    )
    assert build_postgres_connect_args(settings) == {
        "connect_timeout": 19,
        "client_encoding": "utf8",
        "sslmode": "require",
        "sslrootcert": "C:/certs/root.pem",
    }


def test_external_database_rejects_uninstalled_driver() -> None:
    settings = DatabaseSettings(
        embedded_database=False,
        engine="postgresql+psycopg2",
        host="db.example",
        port=5432,
        database_name="neuro_weave",
        username="postgres",
    )

    with pytest.raises(ValueError, match="Unsupported database engine"):
        build_database_backend(settings)


def test_missing_sqlite_file_is_initialized_and_explicit_reset_removes_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "database.db"
    settings = DatabaseSettings(embedded_database=True, sqlite_path=db_path)

    backend = ensure_database_ready(settings)
    assert db_path.exists()
    assert {"projects", "project_revisions"}.issubset(
        inspect(backend.engine).get_table_names()
    )
    with Session(backend.engine) as session:
        session.add(
            ProjectRecord(
                name="reset me",
                description="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                schema_version=1,
                target_framework="keras",
                current_revision=1,
                validation_status="valid",
                graph_json="{}",
                generated_code_json="{}",
            )
        )
        session.commit()

    backend.engine.dispose()
    reset_backend = initialize_database(settings, drop_existing=True)
    reset_factory = build_session_factory(reset_backend.engine)
    with reset_factory() as session:
        assert session.scalar(select(ProjectRecord)) is None
    reset_backend.engine.dispose()


def test_existing_sqlite_schema_is_not_reset_on_normal_startup(tmp_path: Path) -> None:
    db_path = tmp_path / "existing.db"
    settings = DatabaseSettings(embedded_database=True, sqlite_path=db_path)
    backend = initialize_database(settings)
    with Session(backend.engine) as session:
        session.add(
            ProjectRecord(
                name="keep me",
                description="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                schema_version=1,
                target_framework="keras",
                current_revision=1,
                validation_status="valid",
                graph_json="{}",
                generated_code_json="{}",
            )
        )
        session.commit()
    backend.engine.dispose()

    reopened = ensure_database_ready(settings)
    with Session(reopened.engine) as session:
        assert session.scalar(select(ProjectRecord).where(ProjectRecord.name == "keep me"))
    reopened.engine.dispose()


def test_postgres_project_persistence_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    get_database_settings.cache_clear()
    settings = get_database_settings()
    backend = initialize_database(settings, drop_existing=True)
    context = RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )
    repository = ProjectRepository(context)
    now = datetime.now(timezone.utc)
    try:
        created = repository.create(
            name="PostgreSQL contract",
            description="",
            target_framework=TargetFramework.KERAS,
            graph=ModelGraph(),
            timestamp=now,
            validation_status="valid",
        )
        reopened = repository.get(created.id)
        assert reopened is not None
        assert reopened.current_revision == 1
    finally:
        context.dispose()
        get_database_settings.cache_clear()


def test_existing_stale_sqlite_schema_fails_without_recreating(tmp_path: Path) -> None:
    db_path = tmp_path / "stale.db"
    db_path.touch()
    settings = DatabaseSettings(embedded_database=True, sqlite_path=db_path)

    with pytest.raises(RuntimeError, match="missing required table"):
        ensure_database_ready(settings)
