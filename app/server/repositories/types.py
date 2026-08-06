from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

###############################################################################
class UTCDateTime(TypeDecorator[datetime]):
    """Store UTC consistently while supporting SQLite and PostgreSQL."""

    impl = DateTime(timezone=True)
    cache_ok = True

    # -------------------------------------------------------------------------
    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    # -------------------------------------------------------------------------
    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        normalized = self._as_utc(value)
        if dialect.name == "sqlite":
            return normalized.replace(tzinfo=None)
        return normalized

    # -------------------------------------------------------------------------
    def process_result_value(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, value: datetime | None, _dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        return self._as_utc(value)
