"""Replace the sequential custom-node kind with graph templates."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260919_graph_template_custom_nodes"
down_revision = "20260919_execution_node_compilation_consolidation"
branch_labels = None
depends_on = None


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("custom_node_definitions", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_custom_node_definitions_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_custom_node_definitions_kind",
            "kind IN ('mathematical', 'graph_template', 'repository')",
        )

    op.execute(
        sa.text(
            "UPDATE custom_node_definitions SET kind = 'graph_template' "
            "WHERE kind = 'composite'"
        )
    )


###############################################################################
def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE custom_node_definitions SET kind = 'composite' "
            "WHERE kind = 'graph_template'"
        )
    )
    with op.batch_alter_table("custom_node_definitions", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_custom_node_definitions_kind", type_="check")
        batch_op.create_check_constraint(
            "ck_custom_node_definitions_kind",
            "kind IN ('mathematical', 'composite', 'repository')",
        )
