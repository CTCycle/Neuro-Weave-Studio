from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, CheckConstraint, Index, Integer, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from common.timestamps import utc_now
from repositories.base import Base
from repositories.types import UTCDateTime


###############################################################################
class DatasetRecord(Base):
    """A lightweight external dataset registration; source bytes are never stored here."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    format: Mapped[str] = mapped_column(String(80), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_status: Mapped[str] = mapped_column(String(16), nullable=False)
    definition_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    inspection_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    validation_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_datasets_name_nonempty"),
        CheckConstraint(
            "dataset_type IN ('tabular', 'images', 'text', 'time_series', 'array', 'composed')",
            name="ck_datasets_type",
        ),
        CheckConstraint("sample_count >= 0", name="ck_datasets_sample_count"),
        CheckConstraint(
            "selected_sample_count >= 0 AND selected_sample_count <= sample_count",
            name="ck_datasets_selected_sample_count",
        ),
        CheckConstraint(
            "validation_status IN ('valid', 'invalid')",
            name="ck_datasets_validation_status",
        ),
        Index("ix_datasets_updated_at", "updated_at"),
        Index("ix_datasets_type", "dataset_type"),
    )

###############################################################################
class DatasetRepository:

    # -------------------------------------------------------------------------
    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def list_all(self) -> list[DatasetRecord]:
        return list(
            self._session.scalars(
                select(DatasetRecord).order_by(DatasetRecord.updated_at.desc(), DatasetRecord.id)
            ).all()
        )

    # -------------------------------------------------------------------------
    def get(self, dataset_id: UUID) -> DatasetRecord | None:
        return self._session.get(DatasetRecord, str(dataset_id))

    # -------------------------------------------------------------------------
    def add(self, record: DatasetRecord) -> None:
        self._session.add(record)
        self._session.flush()

    # -------------------------------------------------------------------------
    def delete(self, record: DatasetRecord) -> None:
        self._session.delete(record)
        self._session.flush()
