from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from domain.enums import ExecutionBackend, ValidationStatus
from domain.graph import ModelGraph

###############################################################################
@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    execution_backend: ExecutionBackend
    keras_enabled: bool
    current_revision: int
    validation_status: ValidationStatus

###############################################################################
@dataclass(frozen=True, slots=True)
class GraphSummary:
    node_count: int
    edge_count: int

###############################################################################
class RepositoryRevisionConflictError(RuntimeError):

    # -------------------------------------------------------------------------
    def __init__(self, expected_revision: int, current_revision: int) -> None:
        super().__init__(
            f"Revision conflict: expected {expected_revision}, current {current_revision}"
        )
        self.expected_revision = expected_revision
        self.current_revision = current_revision

###############################################################################
class RepositoryIntegrityError(RuntimeError):
    """Persistence integrity failure translated from the ORM boundary."""

###############################################################################
class ProjectRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def list_metadata(self) -> list[ProjectMetadata]: ...

    # -------------------------------------------------------------------------
    def get_metadata(self, project_id: UUID) -> ProjectMetadata | None: ...

    # -------------------------------------------------------------------------
    def create(
        self,
        *,
        name: str,
        description: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        timestamp: datetime,
        validation_status: ValidationStatus,
    ) -> ProjectMetadata: ...

    # -------------------------------------------------------------------------
    def update(
        self,
        project_id: UUID,
        *,
        name: str,
        description: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        timestamp: datetime,
        validation_status: ValidationStatus,
        expected_revision: int,
    ) -> ProjectMetadata | None: ...

    # -------------------------------------------------------------------------
    def delete(self, project_id: UUID) -> bool: ...

###############################################################################
class GraphRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def load(self, project_id: UUID) -> ModelGraph | None: ...

    # -------------------------------------------------------------------------
    def load_many(self, project_ids: Sequence[UUID]) -> dict[UUID, ModelGraph]: ...

    # -------------------------------------------------------------------------
    def replace(self, project_id: UUID, graph: ModelGraph) -> None: ...

    # -------------------------------------------------------------------------
    def summaries(self, project_ids: Sequence[UUID]) -> dict[UUID, GraphSummary]: ...

###############################################################################
class RevisionRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def append(
        self,
        project_id: UUID,
        *,
        revision: int,
        graph: ModelGraph,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        timestamp: datetime,
    ) -> None: ...

###############################################################################
class CodeSnapshotRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def get_for_revision(self, project_id: UUID, revision: int) -> str | None: ...

    # -------------------------------------------------------------------------
    def has_for_revisions(self, revisions: Mapping[UUID, int]) -> dict[UUID, bool]: ...

    # -------------------------------------------------------------------------
    def upsert(
        self,
        project_id: UUID,
        *,
        revision: int,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
        source_code: str,
        timestamp: datetime,
    ) -> None: ...

    # -------------------------------------------------------------------------
    def delete_for_projects(self, project_ids: Sequence[UUID]) -> int: ...

###############################################################################
@dataclass(frozen=True, slots=True)
class MetricUsage:
    project_id: UUID
    project_name: str

###############################################################################
class MetricRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def list_all(self) -> list[Any]: ...

    # -------------------------------------------------------------------------
    def get(self, metric_id: str) -> Any | None: ...

    # -------------------------------------------------------------------------
    def add(self, record: Any) -> None: ...

    # -------------------------------------------------------------------------
    def delete(self, record: Any) -> None: ...

    # -------------------------------------------------------------------------
    def referencing_projects(self, metric_id: str) -> list[MetricUsage]: ...

###############################################################################
class CustomNodeRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def list_all(self) -> list[Any]: ...

    # -------------------------------------------------------------------------
    def get(self, definition_id: str) -> Any | None: ...

    # -------------------------------------------------------------------------
    def add(self, record: Any) -> None: ...

    # -------------------------------------------------------------------------
    def delete(self, record: Any) -> None: ...

###############################################################################
class DatasetRepositoryProtocol(Protocol):

    # -------------------------------------------------------------------------
    def list_all(self) -> list[Any]: ...

    # -------------------------------------------------------------------------
    def get(self, dataset_id: UUID) -> Any | None: ...

    # -------------------------------------------------------------------------
    def add(self, record: Any) -> None: ...

    # -------------------------------------------------------------------------
    def delete(self, record: Any) -> None: ...

###############################################################################
class UnitOfWorkProtocol(Protocol):

    # -------------------------------------------------------------------------
    @property
    def projects(self) -> ProjectRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def graphs(self) -> GraphRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def revisions(self) -> RevisionRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def code_snapshots(self) -> CodeSnapshotRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def metrics(self) -> MetricRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def custom_nodes(self) -> CustomNodeRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    @property
    def datasets(self) -> DatasetRepositoryProtocol: ...

    # -------------------------------------------------------------------------
    def __enter__(self) -> UnitOfWorkProtocol: ...

    # -------------------------------------------------------------------------
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    # -------------------------------------------------------------------------
    def commit(self) -> None: ...

###############################################################################
class UnitOfWorkFactory(Protocol):

    # -------------------------------------------------------------------------
    def __call__(self) -> UnitOfWorkProtocol: ...
