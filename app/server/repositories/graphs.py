from __future__ import annotations

from typing import Literal, cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from domain.graph import (
    EdgeInstance,
    ExportSettings,
    ModelConfiguration,
    ModelGraph,
    NodeInstance,
    Position,
)
from repositories.models import (
    ProjectEdgeRecord,
    ProjectGraphSettingsRecord,
    ProjectNodeRecord,
)

###############################################################################
class GraphRepository:

    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def load(self, project_id: UUID, graph_schema_version: int) -> ModelGraph | None:
        project_key = str(project_id)
        settings = self._session.get(ProjectGraphSettingsRecord, project_key)
        if settings is None:
            return None

        nodes = self._session.scalars(
            select(ProjectNodeRecord)
            .where(ProjectNodeRecord.project_id == project_key)
            .order_by(ProjectNodeRecord.ordinal)
        ).all()
        edges = self._session.scalars(
            select(ProjectEdgeRecord)
            .where(ProjectEdgeRecord.project_id == project_key)
            .order_by(ProjectEdgeRecord.ordinal)
        ).all()
        return ModelGraph(
            schema_version=cast(Literal[2], graph_schema_version),
            nodes=[
                NodeInstance(
                    id=node.node_id,
                    definition_id=node.definition_id,
                    definition_version=node.definition_version,
                    label=node.label,
                    position=Position(x=node.position_x, y=node.position_y),
                    parameters=node.parameters_json,
                )
                for node in nodes
            ],
            edges=[
                EdgeInstance(
                    id=edge.edge_id,
                    source_node_id=edge.source_node_id,
                    source_port_id=edge.source_port_id,
                    target_node_id=edge.target_node_id,
                    target_port_id=edge.target_port_id,
                )
                for edge in edges
            ],
            model_configuration=ModelConfiguration(
                optimizer=settings.optimizer,
                optimizer_parameters=settings.optimizer_parameters_json,
                losses=settings.losses_json,
                metrics=settings.metrics_json,
                default_initializer=settings.default_initializer,
                default_regularizer=settings.default_regularizer,
                default_constraint=settings.default_constraint,
                mixed_precision=settings.mixed_precision,
                random_seed=settings.random_seed,
            ),
            export_settings=ExportSettings(
                include_comments=settings.include_comments,
                include_shape_assertions=settings.include_shape_assertions,
                preferred_binary_format=settings.preferred_binary_format,
            ),
            future_training_configuration=settings.future_training_configuration_json,
        )

    # -------------------------------------------------------------------------
    def replace(self, project_id: UUID, graph: ModelGraph) -> None:
        project_key = str(project_id)
        self._session.execute(
            delete(ProjectEdgeRecord).where(ProjectEdgeRecord.project_id == project_key)
        )
        self._session.execute(
            delete(ProjectNodeRecord).where(ProjectNodeRecord.project_id == project_key)
        )
        self._session.execute(
            delete(ProjectGraphSettingsRecord).where(
                ProjectGraphSettingsRecord.project_id == project_key
            )
        )
        self._session.add(
            ProjectGraphSettingsRecord(
                project_id=project_key,
                optimizer=graph.model_configuration.optimizer,
                optimizer_parameters_json=graph.model_configuration.optimizer_parameters,
                losses_json=graph.model_configuration.losses,
                metrics_json=graph.model_configuration.metrics,
                default_initializer=graph.model_configuration.default_initializer,
                default_regularizer=graph.model_configuration.default_regularizer,
                default_constraint=graph.model_configuration.default_constraint,
                mixed_precision=graph.model_configuration.mixed_precision,
                random_seed=graph.model_configuration.random_seed,
                include_comments=graph.export_settings.include_comments,
                include_shape_assertions=graph.export_settings.include_shape_assertions,
                preferred_binary_format=graph.export_settings.preferred_binary_format,
                future_training_configuration_json=graph.future_training_configuration,
            )
        )
        self._session.add_all(
            [
                ProjectNodeRecord(
                    project_id=project_key,
                    node_id=node.id,
                    ordinal=ordinal,
                    definition_id=node.definition_id,
                    definition_version=node.definition_version,
                    label=node.label,
                    position_x=node.position.x,
                    position_y=node.position.y,
                    parameters_json=node.parameters,
                )
                for ordinal, node in enumerate(graph.nodes)
            ]
        )
        self._session.flush()
        self._session.add_all(
            [
                ProjectEdgeRecord(
                    project_id=project_key,
                    edge_id=edge.id,
                    ordinal=ordinal,
                    source_node_id=edge.source_node_id,
                    source_port_id=edge.source_port_id,
                    target_node_id=edge.target_node_id,
                    target_port_id=edge.target_port_id,
                )
                for ordinal, edge in enumerate(graph.edges)
            ]
        )
        self._session.flush()
