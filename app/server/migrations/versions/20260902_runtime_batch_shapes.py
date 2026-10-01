# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Store input feature shapes without a persisted batch dimension."""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260902_runtime_batch_shapes"
down_revision = "20260901_model_builder_graph_boundaries"
branch_labels = None
depends_on = None

_BATCHED_INPUT_DEFINITIONS = frozenset(
    {
        "input.tensor",
        "input.image",
        "input.sequence",
        "input.tabular",
        "input.token",
        "input.mask",
        "input.constant",
    }
)

###############################################################################
def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}

###############################################################################
def _empty_quantization_config(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    class_name = value.get("class_name")
    config = value.get("config")
    return (class_name is None or str(class_name).strip() == "") and (
        config is None or config == {} or config == []
    )

###############################################################################
def _migrate_parameters(
    definition_id: str,
    parameters: dict[str, Any],
    *,
    downgrade: bool,
) -> dict[str, Any]:
    migrated = dict(parameters)
    if definition_id in _BATCHED_INPUT_DEFINITIONS:
        shape = migrated.get("shape")
        if isinstance(shape, list):
            if downgrade:
                migrated["shape"] = shape if shape and shape[0] is None else [None, *shape]
            elif shape and shape[0] is None:
                migrated["shape"] = shape[1:]
    if not downgrade and _empty_quantization_config(migrated.get("quantization_config")):
        migrated.pop("quantization_config", None)
    return migrated

###############################################################################
def _migrate_project_nodes(connection: Any, *, downgrade: bool) -> None:
    rows = connection.execute(
        sa.text(
            "SELECT project_id, node_id, definition_id, parameters_json "
            "FROM project_nodes"
        )
    ).mappings()
    for row in rows:
        parameters = _json_object(row["parameters_json"])
        migrated = _migrate_parameters(
            str(row["definition_id"]),
            parameters,
            downgrade=downgrade,
        )
        if migrated == parameters:
            continue
        connection.execute(
            sa.text(
                "UPDATE project_nodes SET parameters_json = :parameters "
                "WHERE project_id = :project_id AND node_id = :node_id"
            ),
            {
                "project_id": row["project_id"],
                "node_id": row["node_id"],
                "parameters": json.dumps(migrated),
            },
        )

###############################################################################
def _migrate_revision_snapshots(connection: Any, *, downgrade: bool) -> None:
    rows = connection.execute(
        sa.text("SELECT id, graph_snapshot_json FROM project_revisions")
    ).mappings()
    for row in rows:
        snapshot = _json_object(row["graph_snapshot_json"])
        migrated_nodes: list[object] = []
        for raw_node in snapshot.get("nodes", []):
            if not isinstance(raw_node, dict):
                migrated_nodes.append(raw_node)
                continue
            node = dict(raw_node)
            node["parameters"] = _migrate_parameters(
                str(node.get("definition_id", "")),
                _json_object(node.get("parameters", {})),
                downgrade=downgrade,
            )
            migrated_nodes.append(node)
        snapshot["nodes"] = migrated_nodes
        schema_version = 4 if downgrade else 5
        snapshot["schema_version"] = schema_version
        connection.execute(
            sa.text(
                "UPDATE project_revisions SET graph_schema_version = :schema_version, "
                "graph_snapshot_json = :snapshot WHERE id = :revision_id"
            ),
            {
                "revision_id": row["id"],
                "schema_version": schema_version,
                "snapshot": json.dumps(snapshot),
            },
        )

###############################################################################
def upgrade() -> None:
    connection = op.get_bind()
    _migrate_project_nodes(connection, downgrade=False)
    _migrate_revision_snapshots(connection, downgrade=False)
    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 5"))

###############################################################################
def downgrade() -> None:
    connection = op.get_bind()
    _migrate_project_nodes(connection, downgrade=True)
    _migrate_revision_snapshots(connection, downgrade=True)
    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 4"))
