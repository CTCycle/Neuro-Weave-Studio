from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from common.catalogs.loader import NodeCatalog, get_node_catalog
from repositories.projects import ProjectRepository
from repositories.session import get_session_factory
from services.compilation_service import CompilationService
from services.project_service import ProjectService
from validation.validator import GraphValidator


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_repository(session: Session = Depends(get_session)) -> ProjectRepository:
    return ProjectRepository(session)


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
