"""Create the current execution-profile NeuroWeave schema."""

import sqlalchemy as sa
from alembic import op

revision = "20260821_execution_profile"
down_revision = None
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("graph_schema_version", sa.Integer(), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("keras_enabled", sa.Boolean(create_constraint=True), nullable=False),
        sa.Column("current_revision", sa.Integer(), nullable=False),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_projects_name_nonempty"),
        sa.CheckConstraint("graph_schema_version >= 1", name="ck_projects_graph_version"),
        sa.CheckConstraint(
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
            name="ck_projects_execution_backend",
        ),
        sa.CheckConstraint("current_revision >= 1", name="ck_projects_revision"),
        sa.CheckConstraint(
            "validation_status IN ('unknown', 'valid', 'invalid')",
            name="ck_projects_validation_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_projects_updated_at", "projects", ["updated_at"])
    op.create_table(
        "project_graph_settings",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("optimizer", sa.String(length=120), nullable=False),
        sa.Column("optimizer_parameters_json", sa.JSON(), nullable=False),
        sa.Column("losses_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("default_initializer", sa.String(length=120), nullable=False),
        sa.Column("default_regularizer", sa.String(length=120), nullable=False),
        sa.Column("default_constraint", sa.String(length=120), nullable=False),
        sa.Column("mixed_precision", sa.Boolean(create_constraint=True), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=True),
        sa.Column("include_comments", sa.Boolean(create_constraint=True), nullable=False),
        sa.Column("include_shape_assertions", sa.Boolean(create_constraint=True), nullable=False),
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
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "node_id", name="pk_project_nodes"),
        sa.UniqueConstraint("project_id", "ordinal", name="uq_project_nodes_ordinal"),
        sa.CheckConstraint("length(trim(node_id)) > 0", name="ck_project_nodes_id_nonempty"),
        sa.CheckConstraint("ordinal >= 0", name="ck_project_nodes_ordinal"),
        sa.CheckConstraint("definition_version >= 1", name="ck_project_nodes_definition_version"),
        sa.CheckConstraint(
            "position_x = position_x AND position_x BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_x_finite",
        ),
        sa.CheckConstraint(
            "position_y = position_y AND position_y BETWEEN -1e308 AND 1e308",
            name="ck_project_nodes_position_y_finite",
        ),
    )
    op.create_index("ix_project_nodes_definition", "project_nodes", ["definition_id", "definition_version"])
    op.create_table(
        "project_revisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("graph_schema_version", sa.Integer(), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("keras_enabled", sa.Boolean(create_constraint=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("graph_snapshot_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "revision", name="uq_project_revisions_revision"),
        sa.CheckConstraint("revision >= 1", name="ck_project_revisions_revision"),
        sa.CheckConstraint("graph_schema_version >= 1", name="ck_project_revisions_graph_version"),
        sa.CheckConstraint(
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
            name="ck_project_revisions_execution_backend",
        ),
    )
    op.create_index("ix_project_revisions_project_created", "project_revisions", ["project_id", "created_at"])
    op.create_table(
        "generated_code_snapshots",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("keras_enabled", sa.Boolean(create_constraint=True), nullable=False),
        sa.Column("source_code", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id", "revision"],
            ["project_revisions.project_id", "project_revisions.revision"],
            name="fk_generated_code_snapshots_revision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "revision", name="pk_generated_code_snapshots"),
        sa.CheckConstraint("revision >= 1", name="ck_generated_code_snapshots_revision"),
        sa.CheckConstraint(
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
            name="ck_generated_code_snapshots_execution_backend",
        ),
    )
    op.create_index("ix_generated_code_snapshots_project", "generated_code_snapshots", ["project_id", "revision"])
    op.create_table(
        "project_edges",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("edge_id", sa.String(length=160), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source_node_id", sa.String(length=120), nullable=False),
        sa.Column("source_port_id", sa.String(length=120), nullable=False),
        sa.Column("target_node_id", sa.String(length=120), nullable=False),
        sa.Column("target_port_id", sa.String(length=120), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
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
        sa.PrimaryKeyConstraint("project_id", "edge_id", name="pk_project_edges"),
        sa.UniqueConstraint("project_id", "ordinal", name="uq_project_edges_ordinal"),
        sa.CheckConstraint("length(trim(edge_id)) > 0", name="ck_project_edges_id_nonempty"),
        sa.CheckConstraint("source_node_id <> target_node_id", name="ck_project_edges_no_self_edge"),
        sa.CheckConstraint("ordinal >= 0", name="ck_project_edges_ordinal"),
    )
    op.create_index("ix_project_edges_source", "project_edges", ["project_id", "source_node_id"])
    op.create_index("ix_project_edges_target", "project_edges", ["project_id", "target_node_id"])


###############################################################################
def downgrade() -> None:
    op.drop_index("ix_project_edges_target", table_name="project_edges")
    op.drop_index("ix_project_edges_source", table_name="project_edges")
    op.drop_table("project_edges")
    op.drop_index("ix_generated_code_snapshots_project", table_name="generated_code_snapshots")
    op.drop_table("generated_code_snapshots")
    op.drop_index("ix_project_revisions_project_created", table_name="project_revisions")
    op.drop_table("project_revisions")
    op.drop_index("ix_project_nodes_definition", table_name="project_nodes")
    op.drop_table("project_nodes")
    op.drop_table("project_graph_settings")
    op.drop_index("ix_projects_updated_at", table_name="projects")
    op.drop_table("projects")
