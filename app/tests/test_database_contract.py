from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from configurations.settings import (
    REPOSITORY_ROOT,
    DatabaseSettings,
    get_database_settings,
)
from domain.enums import ExecutionBackend, ValidationStatus
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from repositories.backend import (
    build_session_factory,
)
from repositories.base import Base
from repositories.initializer import HEAD_REVISION, initialize_database
from repositories.models import (
    GeneratedCodeSnapshotRecord,
    ProjectEdgeRecord,
    ProjectGraphSettingsRecord,
    ProjectNodeRecord,
    ProjectRecord,
    ProjectRevisionRecord,
)
from repositories.session import RepositoryContext

###############################################################################
def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path)

###############################################################################
def _context(path: Path) -> RepositoryContext:
    backend = initialize_database(_settings(path))
    return RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )

###############################################################################
def test_sqlite_database_settings_resolve_relative_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", "assets/QA/contract.db")
    settings = get_database_settings()

    assert settings.sqlite_path == (REPOSITORY_ROOT / "assets/QA/contract.db").resolve()

###############################################################################
def test_sqlite_database_settings_use_the_canonical_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_SQLITE_PATH", "")

    settings = get_database_settings()

    assert settings.sqlite_path == (
        REPOSITORY_ROOT / "app/resources/database.db"
    ).resolve()

###############################################################################
def test_database_timeout_is_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT", "0")

    with pytest.raises(ValueError):
        get_database_settings()

###############################################################################
def test_schema_is_canonical_for_sqlite() -> None:
    sqlite_sql = "\n".join(
        str(CreateTable(table).compile(dialect=sqlite.dialect()))
        for table in Base.metadata.sorted_tables
    )

    expected = {
        "projects",
        "project_graph_settings",
        "project_nodes",
        "project_edges",
        "project_revisions",
        "generated_code_snapshots",
    }
    assert {table.name for table in Base.metadata.sorted_tables} == expected
    assert "FOREIGN KEY(project_id, source_node_id)" in sqlite_sql
    assert "uq_project_revisions_revision" in sqlite_sql

###############################################################################
def test_sqlite_database_initialization_creates_schema_and_head(tmp_path: Path) -> None:
    path = tmp_path / "database.db"
    backend = initialize_database(_settings(path))

    assert path.exists()
    assert set(inspect(backend.engine).get_table_names()) == {
        "alembic_version",
        "projects",
        "project_graph_settings",
        "project_nodes",
        "project_edges",
        "project_revisions",
        "generated_code_snapshots",
    }
    with backend.engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD_REVISION
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    backend.engine.dispose()

###############################################################################
def test_empty_existing_sqlite_file_is_initialized(tmp_path: Path) -> None:
    path = tmp_path / "database.db"
    path.touch()

    backend = initialize_database(_settings(path))
    with backend.engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD_REVISION
    backend.engine.dispose()

###############################################################################
def test_already_current_database_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "database.db"
    first = initialize_database(_settings(path))
    first.engine.dispose()

    second = initialize_database(_settings(path))
    with second.engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD_REVISION
    second.engine.dispose()

###############################################################################
def test_incompatible_sqlite_file_is_rejected_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    from sqlalchemy import create_engine

    seed_engine = create_engine(f"sqlite:///{path}")
    with seed_engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy (id INTEGER PRIMARY KEY)"))
    seed_engine.dispose()

    with pytest.raises(RuntimeError, match="no current Alembic revision"):
        initialize_database(_settings(path))

    engine = create_engine(f"sqlite:///{path}")
    try:
        assert inspect(engine).get_table_names() == ["legacy"]
    finally:
        engine.dispose()

###############################################################################
def test_unversioned_projects_table_is_rejected_before_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "unversioned.db"
    from sqlalchemy import create_engine

    seed_engine = create_engine(f"sqlite:///{path}")
    with seed_engine.begin() as connection:
        connection.execute(text("CREATE TABLE projects (id VARCHAR(36) PRIMARY KEY)"))
    seed_engine.dispose()

    with pytest.raises(RuntimeError, match="no current Alembic revision"):
        initialize_database(_settings(path))

    engine = create_engine(f"sqlite:///{path}")
    try:
        assert inspect(engine).get_table_names() == ["projects"]
    finally:
        engine.dispose()

###############################################################################
def test_empty_alembic_version_table_is_initialized_as_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "partial.db"
    from sqlalchemy import create_engine

    seed_engine = create_engine(f"sqlite:///{path}")
    with seed_engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
    seed_engine.dispose()

    backend = initialize_database(_settings(path))
    with backend.engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD_REVISION
    backend.engine.dispose()

###############################################################################
def test_project_repository_operations_share_one_transaction(tmp_path: Path) -> None:
    context = _context(tmp_path / "repository.db")
    now = datetime.now(UTC)
    try:
        with context.new_unit_of_work() as uow:
            created = uow.projects.create(
                name="Repository project",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            graph = ModelGraph()
            uow.graphs.replace(created.id, graph)
            uow.revisions.append(
                created.id,
                revision=1,
                graph=graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.code_snapshots.upsert(
                created.id,
                revision=1,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                source_code="model = None",
                timestamp=now,
            )
            uow.commit()

        with context.new_unit_of_work() as uow:
            metadata = uow.projects.get_metadata(created.id)
            assert metadata is not None
            assert metadata.current_revision == 1
            assert uow.graphs.load(created.id, 2) == ModelGraph()
            assert uow.code_snapshots.get_for_revision(created.id, 1) == "model = None"
    finally:
        context.dispose()

###############################################################################
def test_transaction_rolls_back_partial_project(tmp_path: Path) -> None:
    context = _context(tmp_path / "rollback.db")
    now = datetime.now(UTC)
    project_id: UUID | None = None
    try:
        with pytest.raises(RuntimeError, match="force rollback"), context.new_unit_of_work() as uow:
            metadata = uow.projects.create(
                name="Rolled back",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            project_id = metadata.id
            raise RuntimeError("force rollback")

        with context.new_unit_of_work() as uow:
            assert project_id is not None
            assert uow.projects.get_metadata(project_id) is None
    finally:
        context.dispose()

###############################################################################
def test_project_and_dependents_cascade(tmp_path: Path) -> None:
    context = _context(tmp_path / "cascade.db")
    now = datetime.now(UTC)
    try:
        with context.new_unit_of_work() as uow:
            project = uow.projects.create(
                name="Cascade",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            graph = ModelGraph()
            uow.graphs.replace(project.id, graph)
            uow.revisions.append(
                project.id,
                revision=1,
                graph=graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.code_snapshots.upsert(
                project.id,
                revision=1,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                source_code="code",
                timestamp=now,
            )
            uow.commit()

        with context.new_unit_of_work() as uow:
            assert uow.projects.delete(project.id)
            uow.commit()
        with Session(context.engine) as session:
            assert session.scalar(select(ProjectRecord)) is None
            assert session.scalar(select(ProjectGraphSettingsRecord)) is None
            assert session.scalar(select(ProjectRevisionRecord)) is None
            assert session.scalar(select(GeneratedCodeSnapshotRecord)) is None
    finally:
        context.dispose()

###############################################################################
def test_foreign_keys_are_enabled_for_sqlite(tmp_path: Path) -> None:
    backend = initialize_database(_settings(tmp_path / "foreign-keys.db"))
    with backend.engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    backend.engine.dispose()

###############################################################################
def test_node_delete_cascades_incident_edges(tmp_path: Path) -> None:
    context = _context(tmp_path / "node-cascade.db")
    now = datetime.now(UTC)
    graph = ModelGraph(
        nodes=[
            NodeInstance(
                id="source",
                definition_id="input.tabular",
                position=Position(x=0, y=0),
            ),
            NodeInstance(
                id="target",
                definition_id="output.classification",
                position=Position(x=1, y=0),
            ),
        ],
        edges=[
            EdgeInstance(
                id="edge",
                source_node_id="source",
                source_port_id="output",
                target_node_id="target",
                target_port_id="input",
            )
        ],
    )
    try:
        with context.new_unit_of_work() as uow:
            project = uow.projects.create(
                name="Node cascade",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            uow.graphs.replace(project.id, graph)
            uow.revisions.append(
                project.id,
                revision=1,
                graph=graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.commit()

        with Session(context.engine) as session:
            node = session.get(ProjectNodeRecord, (str(project.id), "source"))
            assert node is not None
            session.delete(node)
            session.commit()

        with Session(context.engine) as session:
            assert session.scalar(select(ProjectEdgeRecord)) is None
    finally:
        context.dispose()

###############################################################################
def test_cross_project_edge_reference_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path / "cross-project-edge.db")
    now = datetime.now(UTC)
    try:
        with context.new_unit_of_work() as uow:
            first = uow.projects.create(
                name="First",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            second = uow.projects.create(
                name="Second",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            first_graph = ModelGraph(
                nodes=[
                    NodeInstance(
                        id="source",
                        definition_id="input.tabular",
                        position=Position(x=0, y=0),
                    )
                ]
            )
            second_graph = ModelGraph(
                nodes=[
                    NodeInstance(
                        id="target",
                        definition_id="output.classification",
                        position=Position(x=0, y=0),
                    )
                ]
            )
            uow.graphs.replace(first.id, first_graph)
            uow.graphs.replace(second.id, second_graph)
            uow.revisions.append(
                first.id,
                revision=1,
                graph=first_graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.revisions.append(
                second.id,
                revision=1,
                graph=second_graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.commit()

        with pytest.raises(Exception, match="FOREIGN KEY"), Session(context.engine) as session:
            session.add(
                ProjectEdgeRecord(
                    project_id=str(first.id),
                    edge_id="cross",
                    ordinal=0,
                    source_node_id="source",
                    source_port_id="output",
                    target_node_id="target",
                    target_port_id="input",
                )
            )
            session.commit()
    finally:
        context.dispose()

###############################################################################
def test_unique_and_check_constraints_reject_invalid_rows(tmp_path: Path) -> None:
    context = _context(tmp_path / "constraints.db")
    now = datetime.now(UTC)
    graph = ModelGraph(
        nodes=[
            NodeInstance(
                id="source",
                definition_id="input.tabular",
                position=Position(x=0, y=0),
            ),
            NodeInstance(
                id="target",
                definition_id="output.classification",
                position=Position(x=1, y=0),
            ),
        ],
        edges=[
            EdgeInstance(
                id="edge",
                source_node_id="source",
                source_port_id="output",
                target_node_id="target",
                target_port_id="input",
            )
        ],
    )
    try:
        with context.new_unit_of_work() as uow:
            project = uow.projects.create(
                name="Constraint project",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=2,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            uow.graphs.replace(project.id, graph)
            uow.revisions.append(
                project.id,
                revision=1,
                graph=graph,
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                timestamp=now,
            )
            uow.commit()

        with pytest.raises(IntegrityError), Session(context.engine) as session:
            session.add(
                ProjectNodeRecord(
                    project_id=str(project.id),
                    node_id="source",
                    ordinal=0,
                    definition_id="input.tabular",
                    definition_version=1,
                    position_x=0,
                    position_y=0,
                    parameters_json={},
                )
            )
            session.commit()

        with pytest.raises(IntegrityError), Session(context.engine) as session:
            session.add(
                ProjectEdgeRecord(
                    project_id=str(project.id),
                    edge_id="edge",
                    ordinal=0,
                    source_node_id="source",
                    source_port_id="output",
                    target_node_id="target",
                    target_port_id="input",
                )
            )
            session.commit()

        with pytest.raises(IntegrityError), Session(context.engine) as session:
            session.add(
                ProjectRevisionRecord(
                    project_id=str(project.id),
                    revision=1,
                    graph_schema_version=2,
                    execution_backend="pytorch",
                    keras_enabled=True,
                    graph_snapshot_json={},
                    created_at=now,
                )
            )
            session.commit()

        with pytest.raises(IntegrityError), Session(context.engine) as session:
            session.add(
                ProjectRecord(
                    name=" ",
                    description="",
                    execution_backend="unknown",
                    keras_enabled=True,
                    validation_status=ValidationStatus.UNKNOWN,
                )
            )
            session.commit()
    finally:
        context.dispose()
