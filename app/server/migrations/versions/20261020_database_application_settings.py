# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Persist application runtime settings in the active database."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261020_database_application_settings"
down_revision = "20260919_graph_template_custom_nodes"
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    op.create_table(
        "application_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_application_settings_singleton"),
        sa.CheckConstraint("schema_version = 1", name="ck_application_settings_schema"),
        sa.CheckConstraint("revision >= 1", name="ck_application_settings_revision"),
        sa.PrimaryKeyConstraint("id"),
    )


###############################################################################
def downgrade() -> None:
    op.drop_table("application_settings")
