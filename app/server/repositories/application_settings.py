# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from sqlalchemy.orm import Session

from common.timestamps import utc_now
from configurations.settings import Settings
from repositories.models import ApplicationSettingsRecord


###############################################################################
class ApplicationSettingsRepository:
    """Read and write the database-local application settings singleton."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    def get(self) -> Settings | None:
        record = self._session.get(ApplicationSettingsRecord, 1)
        if record is None:
            return None
        if record.schema_version != 1:
            raise ValueError(
                f"Unsupported application settings schema version {record.schema_version}."
            )
        return Settings.model_validate(record.settings_json)

    # -------------------------------------------------------------------------
    def save(self, settings: Settings) -> None:
        record = self._session.get(ApplicationSettingsRecord, 1)
        if record is None:
            record = ApplicationSettingsRecord(
                id=1,
                schema_version=1,
                revision=1,
                settings_json=settings.model_dump(mode="json"),
                updated_at=utc_now(),
            )
            self._session.add(record)
            return
        record.settings_json = settings.model_dump(mode="json")
        record.revision += 1
        record.updated_at = utc_now()
