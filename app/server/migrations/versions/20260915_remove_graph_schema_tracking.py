"""Remove redundant graph schema tracking columns from project metadata."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260915_remove_graph_schema_tracking"
down_revision = "20260910_dataset_registrations"
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("projects", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_projects_graph_version", type_="check")
        batch_op.drop_column("graph_schema_version")

    with op.batch_alter_table("project_revisions", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_project_revisions_graph_version", type_="check")
        batch_op.drop_column("graph_schema_version")


###############################################################################
def downgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "graph_schema_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("5"),
        )
    )
    with op.batch_alter_table("projects", recreate="always") as batch_op:
        batch_op.alter_column("graph_schema_version", server_default=None)
        batch_op.create_check_constraint(
            "ck_projects_graph_version", "graph_schema_version >= 1"
        )

    op.add_column(
        "project_revisions",
        sa.Column(
            "graph_schema_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("5"),
        )
    )
    with op.batch_alter_table("project_revisions", recreate="always") as batch_op:
        batch_op.alter_column("graph_schema_version", server_default=None)
        batch_op.create_check_constraint(
            "ck_project_revisions_graph_version", "graph_schema_version >= 1"
        )
