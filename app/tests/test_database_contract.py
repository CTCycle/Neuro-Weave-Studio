from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from configurations.settings import DatabaseSettings
from domain.enums import ExecutionBackend, ValidationStatus
from domain.graph import ModelGraph, NodeInstance, Position
from repositories.backend import build_session_factory
from repositories.initializer import HEAD_REVISION, initialize_database
from repositories.models import (
    GeneratedCodeSnapshotRecord,
    ProjectEdgeRecord,
    ProjectGraphSettingsRecord,
    ProjectRecord,
    ProjectRevisionRecord,
)
from repositories.session import RepositoryContext
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


###############################################################################
def settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(sqlite_path=path)


###############################################################################
def context(path: Path) -> RepositoryContext:
    backend = initialize_database(settings(path))
    return RepositoryContext(
        backend=backend,
        engine=backend.engine,
        session_factory=build_session_factory(backend.engine),
    )


###############################################################################
def test_sqlite_initialization_is_idempotent_and_enforces_foreign_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "database.db"
    first = initialize_database(settings(path))
    with first.engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == HEAD_REVISION
        )
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    first.engine.dispose()

    second = initialize_database(settings(path))
    with second.engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == HEAD_REVISION
        )
    second.engine.dispose()


###############################################################################
def test_incompatible_unversioned_database_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy (id INTEGER PRIMARY KEY)"))
    engine.dispose()

    with pytest.raises(RuntimeError, match="no current Alembic revision"):
        initialize_database(settings(path))

    engine = create_engine(f"sqlite:///{path}")
    try:
        assert inspect(engine).get_table_names() == ["legacy"]
    finally:
        engine.dispose()


###############################################################################
def test_unit_of_work_rolls_back_partial_project(tmp_path: Path) -> None:
    repository = context(tmp_path / "rollback.db")
    project_id: UUID | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="force rollback"),
            repository.new_unit_of_work() as uow,
        ):
            metadata = uow.projects.create(
                name="Rolled back",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=4,
                timestamp=datetime.now(UTC),
                validation_status=ValidationStatus.VALID,
            )
            project_id = metadata.id
            raise RuntimeError("force rollback")

        with repository.new_unit_of_work() as uow:
            assert project_id is not None
            assert uow.projects.get_metadata(project_id) is None
    finally:
        repository.dispose()


###############################################################################
def test_project_delete_cascades_revision_graph_and_generated_source(
    tmp_path: Path,
) -> None:
    repository = context(tmp_path / "cascade.db")
    now = datetime.now(UTC)
    try:
        with repository.new_unit_of_work() as uow:
            project = uow.projects.create(
                name="Cascade",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=4,
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

        with repository.new_unit_of_work() as uow:
            assert uow.projects.delete(project.id)
            uow.commit()

        with Session(repository.engine) as session:
            assert session.scalar(select(ProjectRecord)) is None
            assert session.scalar(select(ProjectGraphSettingsRecord)) is None
            assert session.scalar(select(ProjectRevisionRecord)) is None
            assert session.scalar(select(GeneratedCodeSnapshotRecord)) is None
    finally:
        repository.dispose()


###############################################################################
def test_cross_project_edge_reference_is_rejected(tmp_path: Path) -> None:
    repository = context(tmp_path / "cross-project.db")
    now = datetime.now(UTC)
    try:
        with repository.new_unit_of_work() as uow:
            first = uow.projects.create(
                name="First",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=4,
                timestamp=now,
                validation_status=ValidationStatus.VALID,
            )
            second = uow.projects.create(
                name="Second",
                description="",
                execution_backend=ExecutionBackend.PYTORCH,
                keras_enabled=True,
                graph_schema_version=4,
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

        with pytest.raises(IntegrityError), Session(repository.engine) as session:
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
        repository.dispose()
