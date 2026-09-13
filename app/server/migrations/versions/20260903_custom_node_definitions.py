"""Persist installation-wide custom node definitions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260903_custom_node_definitions"
down_revision = "20260903_custom_metric_definitions"
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    op.create_table(
        "custom_node_definitions",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=160), nullable=False),
        sa.Column("normalized_name", sa.String(length=160), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('mathematical', 'composite', 'repository')",
            name="ck_custom_node_definitions_kind",
        ),
        sa.CheckConstraint(
            "version >= 1",
            name="ck_custom_node_definitions_version",
        ),
        sa.CheckConstraint(
            "length(trim(display_name)) > 0",
            name="ck_custom_node_definitions_display_name",
        ),
        sa.CheckConstraint(
            "length(trim(category)) > 0",
            name="ck_custom_node_definitions_category",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_custom_node_definitions_name"),
    )
    op.create_index(
        "ix_custom_node_definitions_kind",
        "custom_node_definitions",
        ["kind"],
    )
    op.create_index(
        "ix_custom_node_definitions_category",
        "custom_node_definitions",
        ["category"],
    )


###############################################################################
def downgrade() -> None:
    op.drop_index("ix_custom_node_definitions_category", table_name="custom_node_definitions")
    op.drop_index("ix_custom_node_definitions_kind", table_name="custom_node_definitions")
    op.drop_table("custom_node_definitions")
