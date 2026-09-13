from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.dependencies import get_keras_import_service
from domain.enums import ExecutionBackend
from domain.imports import KerasArchitectureImportRequest, KerasArchitectureImportResponse
from services.errors import GraphPersistenceError
from services.keras_import_service import KerasArchitectureImportError, KerasImportService

router = APIRouter(prefix="/api/imports", tags=["imports"])

###############################################################################
@router.post("/keras-architecture", response_model=KerasArchitectureImportResponse, status_code=201)
def import_keras_architecture(
    payload: KerasArchitectureImportRequest = Body(
        ...,
        media_type="application/vnd.keras.zip",
    ),
    execution_backend: ExecutionBackend = Query(default=ExecutionBackend.PYTORCH),
    name: str | None = Query(default=None, max_length=160),
    service: KerasImportService = Depends(get_keras_import_service),
) -> KerasArchitectureImportResponse:
    try:
        return service.import_architecture(
            payload.root,
            execution_backend=execution_backend,
            name_override=name,
        )
    except KerasArchitectureImportError as exc:
        raise HTTPException(
            status_code=422, detail={"code": exc.code, "message": exc.message}
        ) from exc
    except GraphPersistenceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
