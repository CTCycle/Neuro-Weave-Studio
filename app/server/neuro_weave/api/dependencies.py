from __future__ import annotations

from fastapi import Depends

from neuro_weave.catalog.loader import NodeCatalog, get_node_catalog
from neuro_weave.persistence.repositories import ProjectRepository
from neuro_weave.persistence.session import get_repository_context
from neuro_weave.services.compilation_service import CompilationService
from neuro_weave.services.project_service import ProjectService
from neuro_weave.validation.validator import GraphValidator


def get_repository() -> ProjectRepository:
    return ProjectRepository(get_repository_context())


def get_validator(catalog: NodeCatalog = Depends(get_node_catalog)) -> GraphValidator:
    return GraphValidator(catalog)


def get_project_service(
    repository: ProjectRepository = Depends(get_repository),
    validator: GraphValidator = Depends(get_validator),
) -> ProjectService:
    return ProjectService(repository, validator)


def get_compilation_service(
    catalog: NodeCatalog = Depends(get_node_catalog),
    validator: GraphValidator = Depends(get_validator),
) -> CompilationService:
    return CompilationService(catalog, validator)
