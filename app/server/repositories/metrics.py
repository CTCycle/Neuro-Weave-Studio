from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from repositories.contracts import MetricUsage
from repositories.models import (
    CustomMetricDefinitionRecord,
    ProjectGraphSettingsRecord,
    ProjectRecord,
)


class MetricRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_all(self) -> list[CustomMetricDefinitionRecord]:
        return list(
            self._session.scalars(
                select(CustomMetricDefinitionRecord).order_by(
                    CustomMetricDefinitionRecord.kind,
                    CustomMetricDefinitionRecord.created_at,
                    CustomMetricDefinitionRecord.id,
                )
            ).all()
        )

    def get(self, metric_id: str) -> CustomMetricDefinitionRecord | None:
        return self._session.get(CustomMetricDefinitionRecord, metric_id)

    def add(self, record: CustomMetricDefinitionRecord) -> None:
        self._session.add(record)
        self._session.flush()

    def delete(self, record: CustomMetricDefinitionRecord) -> None:
        self._session.delete(record)
        self._session.flush()

    def referencing_projects(self, metric_id: str) -> list[MetricUsage]:
        rows = self._session.execute(
            select(ProjectRecord.id, ProjectRecord.name, ProjectGraphSettingsRecord.model_configuration_json)
            .join(ProjectGraphSettingsRecord, ProjectGraphSettingsRecord.project_id == ProjectRecord.id)
        ).all()
        result: list[MetricUsage] = []
        for project_id, project_name, configuration in rows:
            if _configuration_references(configuration, metric_id):
                result.append(MetricUsage(project_id=UUID(str(project_id)), project_name=str(project_name)))
        return result


def _configuration_references(configuration: object, metric_id: str) -> bool:
    if not isinstance(configuration, dict):
        return False
    for profile in (configuration.get("keras"), configuration.get("native")):
        if not isinstance(profile, dict):
            continue
        profiles = profile.values() if profile is configuration.get("native") else (profile,)
        for item in profiles:
            if not isinstance(item, dict):
                continue
            for key in ("losses", "metrics"):
                values = item.get(key)
                if _contains_metric_id(values, metric_id):
                    return True
    return False


def _contains_metric_id(value: object, metric_id: str) -> bool:
    if isinstance(value, str):
        return value == metric_id
    if isinstance(value, list):
        return any(_contains_metric_id(item, metric_id) for item in value)
    if isinstance(value, dict):
        return any(_contains_metric_id(item, metric_id) for item in value.values())
    return False
