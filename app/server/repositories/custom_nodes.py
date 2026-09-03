from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, CheckConstraint, Index, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from common.timestamps import utc_now
from repositories.base import Base
from repositories.types import UTCDateTime


###############################################################################
class CustomNodeDefinitionRecord(Base):
    """Installation-wide user-authored node definitions."""

    __tablename__ = "custom_node_definitions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    spec_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint(
            "kind IN ('mathematical', 'composite', 'repository')",
            name="ck_custom_node_definitions_kind",
        ),
        CheckConstraint(
            "length(trim(display_name)) > 0",
            name="ck_custom_node_definitions_display_name",
        ),
        CheckConstraint(
            "length(trim(category)) > 0",
            name="ck_custom_node_definitions_category",
        ),
        UniqueConstraint("normalized_name", name="uq_custom_node_definitions_name"),
        Index("ix_custom_node_definitions_kind", "kind"),
        Index("ix_custom_node_definitions_category", "category"),
    )


###############################################################################
class CustomNodeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_all(self) -> list[CustomNodeDefinitionRecord]:
        return list(
            self._session.scalars(
                select(CustomNodeDefinitionRecord).order_by(
                    CustomNodeDefinitionRecord.category,
                    CustomNodeDefinitionRecord.display_name,
                    CustomNodeDefinitionRecord.id,
                )
            ).all()
        )

    def get(self, definition_id: str) -> CustomNodeDefinitionRecord | None:
        return self._session.get(CustomNodeDefinitionRecord, definition_id)

    def add(self, record: CustomNodeDefinitionRecord) -> None:
        self._session.add(record)
        self._session.flush()

    def delete(self, record: CustomNodeDefinitionRecord) -> None:
        self._session.delete(record)
        self._session.flush()
