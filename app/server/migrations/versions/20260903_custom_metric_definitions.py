# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""Persist user-defined metric and loss definitions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260903_custom_metric_definitions"
down_revision = "20260902_runtime_batch_shapes"
branch_labels = None
depends_on = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "custom_metric_definitions",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=160), nullable=False),
        sa.Column("normalized_label", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("formula_latex", sa.Text(), nullable=False),
        sa.Column("expression_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('metric', 'loss')", name="ck_custom_metric_definitions_kind"),
        sa.CheckConstraint("length(trim(label)) > 0", name="ck_custom_metric_definitions_label"),
        sa.CheckConstraint("length(trim(formula_latex)) > 0", name="ck_custom_metric_definitions_formula"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kind", "normalized_label", name="uq_custom_metric_definitions_kind_label"),
    )
    op.create_index(
        "ix_custom_metric_definitions_kind",
        "custom_metric_definitions",
        ["kind"],
    )

###############################################################################
def downgrade() -> None:
    op.drop_index("ix_custom_metric_definitions_kind", table_name="custom_metric_definitions")
    op.drop_table("custom_metric_definitions")
