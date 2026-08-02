from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_project_service, get_repository
from domain.graph import (
    ProjectCreate,
    ProjectDocument,
    ProjectDuplicateRequest,
    ProjectExportBundle,
    ProjectListResponse,
    ProjectUpdate,
)
from repositories.projects import ProjectRepository
from services.project_service import ProjectService

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse)
def list_projects(repository: ProjectRepository = Depends(get_repository)) -> ProjectListResponse:
    return ProjectListResponse(projects=repository.list())


@router.post("", response_model=ProjectDocument, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectDocument:
    return service.create(payload)


@router.get("/{project_id}", response_model=ProjectDocument)
def get_project(
    project_id: UUID,
    repository: ProjectRepository = Depends(get_repository),
) -> ProjectDocument:
    project = repository.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=ProjectDocument)
def update_project(
    project_id: UUID,
    payload: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectDocument:
    try:
        project = service.update(project_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: UUID,
    repository: ProjectRepository = Depends(get_repository),
) -> Response:
    if not repository.delete(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{project_id}/duplicate", response_model=ProjectDocument, status_code=201)
def duplicate_project(
    project_id: UUID,
    payload: ProjectDuplicateRequest,
    service: ProjectService = Depends(get_project_service),
) -> ProjectDocument:
    project = service.duplicate(project_id, payload)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/{project_id}/export", response_model=ProjectExportBundle)
def export_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
) -> ProjectExportBundle:
    bundle = service.export_bundle(project_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return bundle


@router.post("/import", response_model=ProjectDocument, status_code=201)
def import_project(
    payload: ProjectExportBundle,
    service: ProjectService = Depends(get_project_service),
) -> ProjectDocument:
    if payload.format != "neuro-weave-project" or payload.format_version != 1:
        raise HTTPException(status_code=422, detail="Unsupported project bundle version")
    return service.create(payload.project)
