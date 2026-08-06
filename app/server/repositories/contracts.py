from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from domain.enums import CodeTarget, ValidationStatus
from domain.graph import ModelGraph


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    graph_schema_version: int
    code_target: CodeTarget
    current_revision: int
    validation_status: ValidationStatus


class RepositoryRevisionConflictError(RuntimeError):
    def __init__(self, expected_revision: int, current_revision: int) -> None:
        super().__init__(f"Revision conflict: expected {expected_revision}, current {current_revision}")
        self.expected_revision = expected_revision
        self.current_revision = current_revision


class ProjectRepositoryProtocol(Protocol):
    def list_metadata(self) -> list[ProjectMetadata]: ...
    def get_metadata(self, project_id: UUID) -> ProjectMetadata | None: ...
    def create(self, *, name: str, description: str, code_target: CodeTarget,
               graph_schema_version: int, timestamp: datetime,
               validation_status: ValidationStatus) -> ProjectMetadata: ...
    def update(self, project_id: UUID, *, name: str, description: str,
               code_target: CodeTarget, graph_schema_version: int,
               timestamp: datetime, validation_status: ValidationStatus,
               expected_revision: int) -> ProjectMetadata | None: ...
    def delete(self, project_id: UUID) -> bool: ...


class GraphRepositoryProtocol(Protocol):
    def load(self, project_id: UUID, graph_schema_version: int) -> ModelGraph | None: ...
    def replace(self, project_id: UUID, graph: ModelGraph) -> None: ...


class RevisionRepositoryProtocol(Protocol):
    def append(self, project_id: UUID, *, revision: int, graph: ModelGraph,
               code_target: CodeTarget, timestamp: datetime) -> None: ...


class CodeSnapshotRepositoryProtocol(Protocol):
    def list_for_revision(self, project_id: UUID, revision: int) -> dict[str, str]: ...
    def upsert(self, project_id: UUID, *, revision: int, code_target: CodeTarget,
               source_code: str, timestamp: datetime) -> None: ...


class UnitOfWorkProtocol(Protocol):
    @property
    def projects(self) -> ProjectRepositoryProtocol: ...
    @property
    def graphs(self) -> GraphRepositoryProtocol: ...
    @property
    def revisions(self) -> RevisionRepositoryProtocol: ...
    @property
    def code_snapshots(self) -> CodeSnapshotRepositoryProtocol: ...
    def __enter__(self) -> UnitOfWorkProtocol: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    def commit(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWorkProtocol: ...
