from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.dependencies import get_project_service
from domain.enums import ExecutionBackend
from domain.imports import KerasArchitectureImportResponse
from services.keras_import_service import KerasArchitectureImportError, parse_keras_architecture
from services.project_service import ProjectService

router = APIRouter(prefix="/api/imports", tags=["imports"])

###############################################################################
@router.post("/keras-architecture", response_model=KerasArchitectureImportResponse, status_code=201)
async def import_keras_architecture(
    request: Request,
    execution_backend: ExecutionBackend = Query(default=ExecutionBackend.PYTORCH),
    name: str | None = Query(default=None, max_length=160),
    service: ProjectService = Depends(get_project_service),
) -> KerasArchitectureImportResponse:
    try:
        parsed = parse_keras_architecture(
            await request.body(),
            execution_backend=execution_backend,
            name_override=name,
        )
        project = service.create(parsed.payload)
        return KerasArchitectureImportResponse(project=project, warnings=parsed.warnings)
    except KerasArchitectureImportError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message}) from exc
