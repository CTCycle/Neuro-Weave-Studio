"""Persist Model Builder output bindings and node locking state."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260901_model_builder_graph_boundaries"
down_revision = "20260827_canonical_model_configuration"
branch_labels = None
depends_on = None


###############################################################################
def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


###############################################################################
def _legacy_output_bindings(connection: Any) -> dict[str, list[dict[str, str]]]:
    bindings: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows = connection.execute(
        sa.text(
            "SELECT project_id, node_id FROM project_nodes "
            "WHERE definition_id LIKE 'output.%' ORDER BY project_id, ordinal"
        )
    ).mappings()
    for row in rows:
        bindings[str(row["project_id"])].append(
            {"node_id": str(row["node_id"]), "port_id": "output"}
        )
    return dict(bindings)


###############################################################################
def _migrate_revision_snapshots(connection: Any, *, downgrade: bool) -> None:
    rows = connection.execute(
        sa.text("SELECT id, graph_snapshot_json FROM project_revisions")
    ).mappings()
    for row in rows:
        snapshot = _json_object(row["graph_snapshot_json"])
        nodes = snapshot.get("nodes")
        if isinstance(nodes, list):
            migrated_nodes = []
            for raw_node in nodes:
                if not isinstance(raw_node, dict):
                    migrated_nodes.append(raw_node)
                    continue
                node = dict(raw_node)
                if downgrade:
                    node.pop("locked", None)
                else:
                    node.setdefault("locked", False)
                migrated_nodes.append(node)
            snapshot["nodes"] = migrated_nodes
        if downgrade:
            snapshot.pop("output_bindings", None)
            schema_version = 3
        else:
            if "output_bindings" not in snapshot:
                snapshot["output_bindings"] = []
                if isinstance(nodes, list):
                    snapshot["output_bindings"] = [
                        {"node_id": raw_node.get("id"), "port_id": "output"}
                        for raw_node in nodes
                        if isinstance(raw_node, dict)
                        and isinstance(raw_node.get("id"), str)
                        and str(raw_node.get("definition_id", "")).startswith("output.")
                    ]
            schema_version = 4
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
    with op.batch_alter_table("project_graph_settings") as batch:
        batch.add_column(sa.Column("output_bindings_json", sa.JSON(), nullable=True))
    with op.batch_alter_table("project_nodes") as batch:
        batch.add_column(
            sa.Column(
                "locked",
                sa.Boolean(create_constraint=True, name="ck_project_nodes_locked"),
                nullable=True,
            )
        )

    connection = op.get_bind()
    connection.execute(sa.text("UPDATE project_nodes SET locked = 0 WHERE locked IS NULL"))
    bindings_by_project = _legacy_output_bindings(connection)
    project_ids = connection.execute(
        sa.text("SELECT project_id FROM project_graph_settings")
    ).scalars().all()
    for project_id in project_ids:
        connection.execute(
            sa.text(
                "UPDATE project_graph_settings SET output_bindings_json = :bindings "
                "WHERE project_id = :project_id"
            ),
            {
                "project_id": project_id,
                "bindings": json.dumps(bindings_by_project.get(str(project_id), [])),
            },
        )

    _migrate_revision_snapshots(connection, downgrade=False)
    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 4"))

    with op.batch_alter_table("project_graph_settings") as batch:
        batch.alter_column("output_bindings_json", existing_type=sa.JSON(), nullable=False)
    with op.batch_alter_table("project_nodes") as batch:
        batch.alter_column("locked", existing_type=sa.Boolean(), nullable=False)


###############################################################################
def downgrade() -> None:
    connection = op.get_bind()
    _migrate_revision_snapshots(connection, downgrade=True)
    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 3"))

    with op.batch_alter_table("project_graph_settings") as batch:
        batch.drop_column("output_bindings_json")
    with op.batch_alter_table("project_nodes") as batch:
        batch.drop_constraint("ck_project_nodes_locked", type_="check")
        batch.drop_column("locked")
