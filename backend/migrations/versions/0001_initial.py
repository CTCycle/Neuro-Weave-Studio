"""Initial project and revision tables."""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("target_framework", sa.String(length=32), nullable=False),
        sa.Column("current_revision", sa.Integer(), nullable=False),
        sa.Column("validation_status", sa.String(length=32), nullable=False),
        sa.Column("graph_json", sa.Text(), nullable=False),
        sa.Column("generated_code_json", sa.Text(), nullable=False),
    )
    op.create_table(
        "project_revisions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("graph_json", sa.Text(), nullable=False),
        sa.Column("target_framework", sa.String(length=32), nullable=False),
        sa.UniqueConstraint("project_id", "revision", name="uq_project_revision"),
    )


def downgrade() -> None:
    op.drop_table("project_revisions")
    op.drop_table("projects")
