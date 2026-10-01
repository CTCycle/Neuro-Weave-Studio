# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_export_service
from domain.exports import ExportCapabilitiesResponse, SourceExportResponse
from services.errors import (
    CompilationRejectedError,
    ProjectNotFoundError,
    RevisionConflictError,
)
from services.export_service import ExportService

router = APIRouter(prefix="/api/exports", tags=["exports"])

###############################################################################
@router.get("/{project_id}/capabilities", response_model=ExportCapabilitiesResponse)
def export_capabilities(
    project_id: UUID,
    service: ExportService = Depends(get_export_service),
) -> ExportCapabilitiesResponse:
    try:
        return service.capabilities(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

###############################################################################
@router.get("/{project_id}/source", response_model=SourceExportResponse)
def export_source(
    project_id: UUID,
    service: ExportService = Depends(get_export_service),
) -> SourceExportResponse:
    try:
        return service.source(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CompilationRejectedError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "diagnostics": [item.model_dump(mode="json") for item in exc.diagnostics],
            },
        ) from exc
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


###############################################################################
