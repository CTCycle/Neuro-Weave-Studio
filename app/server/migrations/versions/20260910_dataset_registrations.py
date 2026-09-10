"""Persist lightweight external dataset registrations."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260910_dataset_registrations"
down_revision = "20260903_custom_node_definitions"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_datasets_name_nonempty"),
        sa.CheckConstraint(
            "dataset_type IN ('tabular', 'images', 'text', 'time_series', 'array', 'composed')",
            name="ck_datasets_type",
        ),
        sa.CheckConstraint("sample_count >= 0", name="ck_datasets_sample_count"),
        sa.CheckConstraint(
            "selected_sample_count >= 0 AND selected_sample_count <= sample_count",
            name="ck_datasets_selected_sample_count",
        ),
        sa.CheckConstraint(
            "validation_status IN ('valid', 'invalid')",
            name="ck_datasets_validation_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_updated_at", "datasets", ["updated_at"])
    op.create_index("ix_datasets_type", "datasets", ["dataset_type"])


def downgrade() -> None:
    op.drop_index("ix_datasets_type", table_name="datasets")
    op.drop_index("ix_datasets_updated_at", table_name="datasets")
    op.drop_table("datasets")
