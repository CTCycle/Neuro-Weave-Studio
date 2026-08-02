from __future__ import annotations

import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_compilation_service, get_repository, get_validator
from configurations.settings import get_settings
from domain.enums import TargetFramework
from exporters.keras_exporter import KerasBinaryExporter
from repositories.projects import ProjectRepository
from services.compilation_service import CompilationService
from validation.validator import GraphValidator

router = APIRouter(prefix="/api/exports", tags=["exports"])


def _safe_filename_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (sanitized or "model")[:120]


@router.get("/{project_id}/source/{framework}")
def export_source(
    project_id: UUID,
    framework: TargetFramework,
    repository: ProjectRepository = Depends(get_repository),
    service: CompilationService = Depends(get_compilation_service),
) -> dict[str, str]:
    project = repository.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    result = service.compile(project.graph, framework)
    if not result.valid or result.code is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Graph cannot be compiled",
                "diagnostics": [
                    item.model_dump(mode="json") for item in result.diagnostics
                ],
            },
        )
    repository.save_code_snapshot(project_id, framework, result.code)
    return {"framework": framework.value, "code": result.code}


@router.post("/{project_id}/keras-model")
def export_keras_model(
    project_id: UUID,
    repository: ProjectRepository = Depends(get_repository),
    validator: GraphValidator = Depends(get_validator),
) -> FileResponse:
    project = repository.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    validation = validator.validate(project.graph, TargetFramework.KERAS)
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Graph is not Keras compatible",
                "diagnostics": [
                    item.model_dump(mode="json") for item in validation.diagnostics
                ],
            },
        )
    destination = (
        get_settings().export_root
        / str(project_id)
        / f"{_safe_filename_stem(project.name)}.keras"
    )
    try:
        KerasBinaryExporter().export(project.graph, destination)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return FileResponse(
        path=destination,
        filename=destination.name,
        media_type="application/octet-stream",
    )
