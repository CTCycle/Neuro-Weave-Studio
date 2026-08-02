from __future__ import annotations

from uuid import UUID

from neuro_weave.domain.graph import (
    ProjectCreate,
    ProjectDocument,
    ProjectDuplicateRequest,
    ProjectExportBundle,
    ProjectUpdate,
    utc_now,
)
from neuro_weave.persistence.repositories import ProjectRepository
from neuro_weave.validation.validator import GraphValidator


class ProjectService:
    def __init__(self, repository: ProjectRepository, validator: GraphValidator) -> None:
        self._repository = repository
        self._validator = validator

    def create(self, payload: ProjectCreate) -> ProjectDocument:
        result = self._validator.validate(payload.graph, payload.target_framework)
        return self._repository.create(
            name=payload.name,
            description=payload.description,
            target_framework=payload.target_framework,
            graph=payload.graph,
            timestamp=utc_now(),
            validation_status="valid" if result.valid else "invalid",
        )

    def update(self, project_id: UUID, payload: ProjectUpdate) -> ProjectDocument | None:
        result = self._validator.validate(payload.graph, payload.target_framework)
        return self._repository.update(
            project_id,
            name=payload.name,
            description=payload.description,
            target_framework=payload.target_framework,
            graph=payload.graph,
            timestamp=utc_now(),
            validation_status="valid" if result.valid else "invalid",
            expected_revision=payload.expected_revision,
        )

    def duplicate(
        self, project_id: UUID, payload: ProjectDuplicateRequest
    ) -> ProjectDocument | None:
        source = self._repository.get(project_id)
        if source is None:
            return None
        return self.create(
            ProjectCreate(
                name=payload.name or f"{source.name} copy",
                description=source.description,
                target_framework=source.target_framework,
                graph=source.graph,
            )
        )

    def export_bundle(self, project_id: UUID) -> ProjectExportBundle | None:
        project = self._repository.get(project_id)
        if project is None:
            return None
        return ProjectExportBundle(
            project=ProjectCreate(
                name=project.name,
                description=project.description,
                target_framework=project.target_framework,
                graph=project.graph,
            )
        )
