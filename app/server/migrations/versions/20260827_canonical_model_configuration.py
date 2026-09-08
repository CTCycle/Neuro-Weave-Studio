# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260827_canonical_model_configuration"
down_revision = "20260821_execution_profile"
branch_labels = None
depends_on = None


###############################################################################
def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


###############################################################################
def _native_defaults() -> dict[str, object]:
    return {
        "tensorflow": {"jit_compile": False},
        "pytorch": {
            "compile_enabled": False,
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
        },
        "jax": {"jit_enabled": False},
    }


###############################################################################
def _canonical_configuration(row: Any) -> dict[str, object]:
    return {
        "keras": {
            "optimizer": row["optimizer"],
            "optimizer_parameters": _json_object(row["optimizer_parameters_json"]),
            "losses": _json_object(row["losses_json"]),
            "metrics": _json_object(row["metrics_json"]),
            "default_initializer": row["default_initializer"],
            "default_regularizer": row["default_regularizer"],
            "default_constraint": row["default_constraint"],
            "mixed_precision": bool(row["mixed_precision"]),
            "random_seed": row["random_seed"],
        },
        "native": _native_defaults(),
    }


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("project_graph_settings") as batch:
        batch.add_column(sa.Column("model_configuration_json", sa.JSON(), nullable=True))

    connection = op.get_bind()
    settings = (
        connection.execute(
            sa.text(
                "SELECT project_id, optimizer, optimizer_parameters_json, losses_json, "
                "metrics_json, default_initializer, default_regularizer, default_constraint, "
                "mixed_precision, random_seed FROM project_graph_settings"
            )
        )
        .mappings()
        .all()
    )
    for row in settings:
        connection.execute(
            sa.text(
                "UPDATE project_graph_settings SET model_configuration_json = :configuration "
                "WHERE project_id = :project_id"
            ),
            {
                "project_id": row["project_id"],
                "configuration": json.dumps(_canonical_configuration(row)),
            },
        )

    revisions = (
        connection.execute(sa.text("SELECT id, graph_snapshot_json FROM project_revisions"))
        .mappings()
        .all()
    )
    for row in revisions:
        snapshot = _json_object(row["graph_snapshot_json"])
        configuration = _json_object(snapshot.get("model_configuration"))
        if "keras" not in configuration:
            configuration = {"keras": configuration, "native": _native_defaults()}
        snapshot["schema_version"] = 3
        snapshot["model_configuration"] = configuration
        connection.execute(
            sa.text(
                "UPDATE project_revisions SET graph_schema_version = 3, "
                "graph_snapshot_json = :snapshot WHERE id = :revision_id"
            ),
            {"revision_id": row["id"], "snapshot": json.dumps(snapshot)},
        )

    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 3"))
    with op.batch_alter_table("project_graph_settings") as batch:
        batch.alter_column("model_configuration_json", existing_type=sa.JSON(), nullable=False)
        for column in (
            "optimizer",
            "optimizer_parameters_json",
            "losses_json",
            "metrics_json",
            "default_initializer",
            "default_regularizer",
            "default_constraint",
            "mixed_precision",
            "random_seed",
        ):
            batch.drop_column(column)


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("project_graph_settings") as batch:
        batch.add_column(sa.Column("optimizer", sa.String(length=120), nullable=True))
        batch.add_column(sa.Column("optimizer_parameters_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("losses_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("metrics_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("default_initializer", sa.String(length=120), nullable=True))
        batch.add_column(sa.Column("default_regularizer", sa.String(length=120), nullable=True))
        batch.add_column(sa.Column("default_constraint", sa.String(length=120), nullable=True))
        batch.add_column(sa.Column("mixed_precision", sa.Boolean(), nullable=True))
        batch.add_column(sa.Column("random_seed", sa.Integer(), nullable=True))

    connection = op.get_bind()
    rows = (
        connection.execute(
            sa.text("SELECT project_id, model_configuration_json FROM project_graph_settings")
        )
        .mappings()
        .all()
    )
    for row in rows:
        configuration = _json_object(row["model_configuration_json"])
        keras = _json_object(configuration.get("keras"))
        connection.execute(
            sa.text(
                "UPDATE project_graph_settings SET optimizer = :optimizer, "
                "optimizer_parameters_json = :optimizer_parameters, losses_json = :losses, "
                "metrics_json = :metrics, default_initializer = :initializer, "
                "default_regularizer = :regularizer, default_constraint = :constraint, "
                "mixed_precision = :mixed_precision, random_seed = :random_seed "
                "WHERE project_id = :project_id"
            ),
            {
                "project_id": row["project_id"],
                "optimizer": keras.get("optimizer", "adam"),
                "optimizer_parameters": json.dumps(keras.get("optimizer_parameters", {})),
                "losses": json.dumps(keras.get("losses", {})),
                "metrics": json.dumps(keras.get("metrics", {})),
                "initializer": keras.get("default_initializer", "glorot_uniform"),
                "regularizer": keras.get("default_regularizer", "none"),
                "constraint": keras.get("default_constraint", "none"),
                "mixed_precision": bool(keras.get("mixed_precision", False)),
                "random_seed": keras.get("random_seed", 42),
            },
        )

    revisions = (
        connection.execute(sa.text("SELECT id, graph_snapshot_json FROM project_revisions"))
        .mappings()
        .all()
    )
    for row in revisions:
        snapshot = _json_object(row["graph_snapshot_json"])
        configuration = _json_object(snapshot.get("model_configuration"))
        snapshot["schema_version"] = 2
        snapshot["model_configuration"] = (
            _json_object(configuration.get("keras")) if "keras" in configuration else configuration
        )
        connection.execute(
            sa.text(
                "UPDATE project_revisions SET graph_snapshot_json = :snapshot "
                "WHERE id = :revision_id"
            ),
            {"revision_id": row["id"], "snapshot": json.dumps(snapshot)},
        )
    connection.execute(sa.text("UPDATE projects SET graph_schema_version = 2"))
    connection.execute(sa.text("UPDATE project_revisions SET graph_schema_version = 2"))
    with op.batch_alter_table("project_graph_settings") as batch:
        for column in (
            "optimizer",
            "optimizer_parameters_json",
            "losses_json",
            "metrics_json",
            "default_initializer",
            "default_regularizer",
            "default_constraint",
            "mixed_precision",
        ):
            batch.alter_column(column, nullable=False)
        batch.drop_column("model_configuration_json")
