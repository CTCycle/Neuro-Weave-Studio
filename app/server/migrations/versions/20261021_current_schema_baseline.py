# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Create the current database schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261021_current_schema_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "application_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_application_settings_singleton"),
        sa.CheckConstraint("schema_version = 1", name="ck_application_settings_schema"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "custom_metric_definitions",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=160), nullable=False),
        sa.Column("normalized_label", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("formula_latex", sa.Text(), nullable=False),
        sa.Column("expression_json", sa.JSON(), nullable=False),
        sa.Column("capabilities_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('metric', 'loss')", name="ck_custom_metric_definitions_kind"),
        sa.CheckConstraint(
            "length(trim(formula_latex)) > 0", name="ck_custom_metric_definitions_formula"
        ),
        sa.CheckConstraint(
            "length(trim(label)) > 0", name="ck_custom_metric_definitions_label"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kind", "normalized_label", name="uq_custom_metric_definitions_kind_label"
        ),
    )
    op.create_index(
        "ix_custom_metric_definitions_kind", "custom_metric_definitions", ["kind"], unique=False
    )
    op.create_table(
        "custom_node_definitions",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=160), nullable=False),
        sa.Column("normalized_name", sa.String(length=160), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('mathematical', 'graph_template', 'repository')",
            name="ck_custom_node_definitions_kind",
        ),
        sa.CheckConstraint(
            "length(trim(category)) > 0", name="ck_custom_node_definitions_category"
        ),
        sa.CheckConstraint(
            "length(trim(display_name)) > 0", name="ck_custom_node_definitions_display_name"
        ),
        sa.CheckConstraint("version >= 1", name="ck_custom_node_definitions_version"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_custom_node_definitions_name"),
    )
    op.create_index(
        "ix_custom_node_definitions_category", "custom_node_definitions", ["category"], unique=False
    )
    op.create_index(
        "ix_custom_node_definitions_kind", "custom_node_definitions", ["kind"], unique=False
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("dataset_type", sa.String(length=32), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("selected_sample_count", sa.Integer(), nullable=False),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("definition_json", sa.JSON(), nullable=False),
        sa.Column("inspection_json", sa.JSON(), nullable=False),
        sa.Column("validation_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "dataset_type IN ('tabular', 'images', 'text', 'time_series', 'array', 'composed')",
            name="ck_datasets_type",
        ),
        sa.CheckConstraint(
            "validation_status IN ('valid', 'invalid')",
            name="ck_datasets_validation_status",
        ),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_datasets_name_nonempty"),
        sa.CheckConstraint("sample_count >= 0", name="ck_datasets_sample_count"),
        sa.CheckConstraint(
            "selected_sample_count >= 0 AND selected_sample_count <= sample_count",
            name="ck_datasets_selected_sample_count",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_type", "datasets", ["dataset_type"], unique=False)
    op.create_index("ix_datasets_updated_at", "datasets", ["updated_at"], unique=False)
    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column(
            "keras_enabled",
            sa.Boolean(create_constraint=True, name="ck_projects_keras_enabled"),
            nullable=False,
        ),
        sa.Column("current_revision", sa.Integer(), nullable=False),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
            name="ck_projects_execution_backend",
        ),
        sa.CheckConstraint(
            "validation_status IN ('unknown', 'valid', 'invalid')",
            name="ck_projects_validation_status",
        ),
        sa.CheckConstraint("current_revision >= 1", name="ck_projects_revision"),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_projects_name_nonempty"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_projects_updated_at", "projects", ["updated_at"], unique=False)
    op.create_table(
        "project_graph_settings",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("output_bindings_json", sa.JSON(), nullable=False),
        sa.Column("model_configuration_json", sa.JSON(), nullable=False),
        sa.Column(
            "include_comments",
            sa.Boolean(create_constraint=True, name="ck_project_graph_settings_include_comments"),
            nullable=False,
        ),
        sa.Column(
            "include_shape_assertions",
            sa.Boolean(
                create_constraint=True,
                name="ck_project_graph_settings_include_shape_assertions",
            ),
            nullable=False,
        ),
        sa.Column("preferred_binary_format", sa.String(length=32), nullable=False),
        sa.Column("future_training_configuration_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_table(
        "project_nodes",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("node_id", sa.String(length=120), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("definition_id", sa.String(length=160), nullable=False),
        sa.Column("definition_version", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=240), nullable=True),
        sa.Column("position_x", sa.Float(), nullable=False),
        sa.Column("position_y", sa.Float(), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=False),
        sa.Column(
            "locked",
            sa.Boolean(create_constraint=True, name="ck_project_nodes_locked"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "definition_version >= 1", name="ck_project_nodes_definition_version"
        ),
        sa.CheckConstraint("length(trim(node_id)) > 0", name="ck_project_nodes_id_nonempty"),
        sa.CheckConstraint("ordinal >= 0", name="ck_project_nodes_ordinal"),
        sa.CheckConstraint(
            "position_x = position_x AND position_x BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_x_finite",
        ),
        sa.CheckConstraint(
            "position_y = position_y AND position_y BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_y_finite",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "node_id", name="pk_project_nodes"),
        sa.UniqueConstraint("project_id", "ordinal", name="uq_project_nodes_ordinal"),
    )
    op.create_index(
        "ix_project_nodes_definition",
        "project_nodes",
        ["definition_id", "definition_version"],
        unique=False,
    )
    op.create_table(
        "project_revisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_project_revisions_revision"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "revision", name="uq_project_revisions_revision"),
    )
    op.create_index(
        "ix_project_revisions_project_created",
        "project_revisions",
        ["project_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "generated_code_snapshots",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_code", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_generated_code_snapshots_revision"),
        sa.ForeignKeyConstraint(
            ["project_id", "revision"],
            ["project_revisions.project_id", "project_revisions.revision"],
            name="fk_generated_code_snapshots_revision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "project_id", "revision", name="pk_generated_code_snapshots"
        ),
    )
    op.create_index(
        "ix_generated_code_snapshots_project",
        "generated_code_snapshots",
        ["project_id", "revision"],
        unique=False,
    )
    op.create_table(
        "project_edges",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("edge_id", sa.String(length=160), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source_node_id", sa.String(length=120), nullable=False),
        sa.Column("source_port_id", sa.String(length=120), nullable=False),
        sa.Column("target_node_id", sa.String(length=120), nullable=False),
        sa.Column("target_port_id", sa.String(length=120), nullable=False),
        sa.CheckConstraint("length(trim(edge_id)) > 0", name="ck_project_edges_id_nonempty"),
        sa.CheckConstraint("ordinal >= 0", name="ck_project_edges_ordinal"),
        sa.CheckConstraint("source_node_id <> target_node_id", name="ck_project_edges_no_self_edge"),
        sa.ForeignKeyConstraint(
            ["project_id", "source_node_id"],
            ["project_nodes.project_id", "project_nodes.node_id"],
            name="fk_project_edges_source_node",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id", "target_node_id"],
            ["project_nodes.project_id", "project_nodes.node_id"],
            name="fk_project_edges_target_node",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "edge_id", name="pk_project_edges"),
        sa.UniqueConstraint("project_id", "ordinal", name="uq_project_edges_ordinal"),
    )
    op.create_index(
        "ix_project_edges_source",
        "project_edges",
        ["project_id", "source_node_id"],
        unique=False,
    )
    op.create_index(
        "ix_project_edges_target",
        "project_edges",
        ["project_id", "target_node_id"],
        unique=False,
    )
