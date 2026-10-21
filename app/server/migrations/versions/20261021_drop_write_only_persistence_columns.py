# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Drop write-only persistence columns left behind by earlier lifecycles."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261021_drop_write_only_persistence_columns"
down_revision = "20261020_database_application_settings"
branch_labels = None
depends_on = None

###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("project_revisions", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_project_revisions_execution_backend", type_="check")
        batch_op.drop_column("graph_snapshot_json")
        batch_op.drop_column("execution_backend")
        batch_op.drop_column("keras_enabled")

    with op.batch_alter_table("generated_code_snapshots", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_generated_code_snapshots_execution_backend", type_="check")
        batch_op.drop_column("execution_backend")
        batch_op.drop_column("keras_enabled")

    with op.batch_alter_table("application_settings", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_application_settings_revision", type_="check")
        batch_op.drop_column("revision")

###############################################################################
def downgrade() -> None:
    op.add_column(
        "project_revisions",
        sa.Column(
            "graph_snapshot_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "project_revisions",
        sa.Column(
            "execution_backend",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pytorch'"),
        ),
    )
    op.add_column(
        "project_revisions",
        sa.Column("keras_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    with op.batch_alter_table("project_revisions", recreate="always") as batch_op:
        batch_op.alter_column("graph_snapshot_json", server_default=None)
        batch_op.alter_column("execution_backend", server_default=None)
        batch_op.alter_column("keras_enabled", server_default=None)
        batch_op.create_check_constraint(
            "ck_project_revisions_execution_backend",
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
        )
        batch_op.create_check_constraint(
            "ck_project_revisions_keras_enabled", "keras_enabled IN (0, 1)"
        )

    op.add_column(
        "generated_code_snapshots",
        sa.Column(
            "execution_backend",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pytorch'"),
        ),
    )
    op.add_column(
        "generated_code_snapshots",
        sa.Column("keras_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    with op.batch_alter_table("generated_code_snapshots", recreate="always") as batch_op:
        batch_op.alter_column("execution_backend", server_default=None)
        batch_op.alter_column("keras_enabled", server_default=None)
        batch_op.create_check_constraint(
            "ck_generated_code_snapshots_execution_backend",
            "execution_backend IN ('tensorflow', 'pytorch', 'jax')",
        )
        batch_op.create_check_constraint(
            "ck_generated_code_snapshots_keras_enabled", "keras_enabled IN (0, 1)"
        )

    op.add_column(
        "application_settings",
        sa.Column("revision", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )
    with op.batch_alter_table("application_settings", recreate="always") as batch_op:
        batch_op.alter_column("revision", server_default=None)
        batch_op.create_check_constraint(
            "ck_application_settings_revision", "revision >= 1"
        )
