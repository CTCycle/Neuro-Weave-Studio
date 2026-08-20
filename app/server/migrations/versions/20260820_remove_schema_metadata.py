"""Make Alembic the sole schema-version authority."""

from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision = "20260820_head"
down_revision = "20260820_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("schema_metadata")


def downgrade() -> None:
    op.create_table(
        "schema_metadata",
        sa.Column("id", sa.SmallInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("initialized_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_schema_metadata_singleton"),
        sa.CheckConstraint("schema_version >= 1", name="ck_schema_metadata_version"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        sa.text(
            "INSERT INTO schema_metadata (id, schema_version, initialized_at) "
            "VALUES (1, 2, :initialized_at)"
        ).bindparams(initialized_at=datetime.now(UTC))
    )
