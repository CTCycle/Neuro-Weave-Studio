from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from common.timestamps import utc_now
from domain.enums import ExecutionBackend, ValidationStatus
from repositories.base import Base
from repositories.types import UTCDateTime

_EXECUTION_BACKEND_SQL_VALUES = ", ".join(repr(item.value) for item in ExecutionBackend)
_VALIDATION_STATUS_SQL_VALUES = ", ".join(repr(item.value) for item in ValidationStatus)


###############################################################################
class ProjectRecord(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    graph_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    execution_backend: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ExecutionBackend.PYTORCH.value
    )
    keras_enabled: Mapped[bool] = mapped_column(
        Boolean(create_constraint=True, name="ck_projects_keras_enabled"),
        nullable=False,
        default=True,
    )
    current_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    validation_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ValidationStatus.UNKNOWN.value
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    graph_settings: Mapped[ProjectGraphSettingsRecord] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    nodes: Mapped[list[ProjectNodeRecord]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProjectNodeRecord.ordinal",
    )
    edges: Mapped[list[ProjectEdgeRecord]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProjectEdgeRecord.ordinal",
    )
    revisions: Mapped[list[ProjectRevisionRecord]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProjectRevisionRecord.revision",
    )

    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_projects_name_nonempty"),
        CheckConstraint("graph_schema_version >= 1", name="ck_projects_graph_version"),
        CheckConstraint(
            f"execution_backend IN ({_EXECUTION_BACKEND_SQL_VALUES})",
            name="ck_projects_execution_backend",
        ),
        CheckConstraint("current_revision >= 1", name="ck_projects_revision"),
        CheckConstraint(
            f"validation_status IN ({_VALIDATION_STATUS_SQL_VALUES})",
            name="ck_projects_validation_status",
        ),
        Index("ix_projects_updated_at", "updated_at"),
    )


###############################################################################
class ProjectGraphSettingsRecord(Base):
    __tablename__ = "project_graph_settings"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    output_bindings_json: Mapped[list[object]] = mapped_column(
        JSON, nullable=False, default=list
    )
    model_configuration_json: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    include_comments: Mapped[bool] = mapped_column(
        Boolean(create_constraint=True, name="ck_project_graph_settings_include_comments"),
        nullable=False,
        default=True,
    )
    include_shape_assertions: Mapped[bool] = mapped_column(
        Boolean(
            create_constraint=True,
            name="ck_project_graph_settings_include_shape_assertions",
        ),
        nullable=False,
        default=True,
    )
    preferred_binary_format: Mapped[str] = mapped_column(
        String(32), nullable=False, default="keras"
    )
    future_training_configuration_json: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    project: Mapped[ProjectRecord] = relationship(back_populates="graph_settings")


###############################################################################
class ProjectNodeRecord(Base):
    __tablename__ = "project_nodes"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(String(120), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    definition_id: Mapped[str] = mapped_column(String(160), nullable=False)
    definition_version: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(240), nullable=True)
    position_x: Mapped[float] = mapped_column(Float, nullable=False)
    position_y: Mapped[float] = mapped_column(Float, nullable=False)
    parameters_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    locked: Mapped[bool] = mapped_column(
        Boolean(create_constraint=True, name="ck_project_nodes_locked"),
        nullable=False,
        default=False,
    )

    project: Mapped[ProjectRecord] = relationship(back_populates="nodes")

    __table_args__ = (
        PrimaryKeyConstraint("project_id", "node_id", name="pk_project_nodes"),
        UniqueConstraint("project_id", "ordinal", name="uq_project_nodes_ordinal"),
        CheckConstraint("length(trim(node_id)) > 0", name="ck_project_nodes_id_nonempty"),
        CheckConstraint("ordinal >= 0", name="ck_project_nodes_ordinal"),
        CheckConstraint("definition_version >= 1", name="ck_project_nodes_definition_version"),
        CheckConstraint(
            "position_x = position_x AND position_x BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_x_finite",
        ),
        CheckConstraint(
            "position_y = position_y AND position_y BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_y_finite",
        ),
        Index("ix_project_nodes_definition", "definition_id", "definition_version"),
    )


###############################################################################
class ProjectEdgeRecord(Base):
    __tablename__ = "project_edges"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    edge_id: Mapped[str] = mapped_column(String(160), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_node_id: Mapped[str] = mapped_column(String(120), nullable=False)
    source_port_id: Mapped[str] = mapped_column(String(120), nullable=False)
    target_node_id: Mapped[str] = mapped_column(String(120), nullable=False)
    target_port_id: Mapped[str] = mapped_column(String(120), nullable=False)

    project: Mapped[ProjectRecord] = relationship(back_populates="edges")

    __table_args__ = (
        PrimaryKeyConstraint("project_id", "edge_id", name="pk_project_edges"),
        UniqueConstraint("project_id", "ordinal", name="uq_project_edges_ordinal"),
        CheckConstraint("length(trim(edge_id)) > 0", name="ck_project_edges_id_nonempty"),
        ForeignKeyConstraint(
            ["project_id", "source_node_id"],
            ["project_nodes.project_id", "project_nodes.node_id"],
            name="fk_project_edges_source_node",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["project_id", "target_node_id"],
            ["project_nodes.project_id", "project_nodes.node_id"],
            name="fk_project_edges_target_node",
            ondelete="CASCADE",
        ),
        CheckConstraint("source_node_id <> target_node_id", name="ck_project_edges_no_self_edge"),
        CheckConstraint("ordinal >= 0", name="ck_project_edges_ordinal"),
        Index("ix_project_edges_source", "project_id", "source_node_id"),
        Index("ix_project_edges_target", "project_id", "target_node_id"),
    )


###############################################################################
class ProjectRevisionRecord(Base):
    __tablename__ = "project_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    graph_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    keras_enabled: Mapped[bool] = mapped_column(
        Boolean(create_constraint=True, name="ck_project_revisions_keras_enabled"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    graph_snapshot_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)

    project: Mapped[ProjectRecord] = relationship(back_populates="revisions")

    __table_args__ = (
        UniqueConstraint("project_id", "revision", name="uq_project_revisions_revision"),
        CheckConstraint("revision >= 1", name="ck_project_revisions_revision"),
        CheckConstraint("graph_schema_version >= 1", name="ck_project_revisions_graph_version"),
        CheckConstraint(
            f"execution_backend IN ({_EXECUTION_BACKEND_SQL_VALUES})",
            name="ck_project_revisions_execution_backend",
        ),
        Index("ix_project_revisions_project_created", "project_id", "created_at"),
    )


###############################################################################
class GeneratedCodeSnapshotRecord(Base):
    __tablename__ = "generated_code_snapshots"

    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    keras_enabled: Mapped[bool] = mapped_column(
        Boolean(create_constraint=True, name="ck_generated_code_snapshots_keras_enabled"),
        nullable=False,
    )
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    __table_args__ = (
        PrimaryKeyConstraint("project_id", "revision", name="pk_generated_code_snapshots"),
        ForeignKeyConstraint(
            ["project_id", "revision"],
            ["project_revisions.project_id", "project_revisions.revision"],
            name="fk_generated_code_snapshots_revision",
            ondelete="CASCADE",
        ),
        CheckConstraint("revision >= 1", name="ck_generated_code_snapshots_revision"),
        CheckConstraint(
            f"execution_backend IN ({_EXECUTION_BACKEND_SQL_VALUES})",
            name="ck_generated_code_snapshots_execution_backend",
        ),
        Index("ix_generated_code_snapshots_project", "project_id", "revision"),
    )
