from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_export_service
from domain.enums import CodeTarget
from domain.exports import SourceExportResponse
from services.errors import (
    CompilationRejectedError,
    ProjectNotFoundError,
    RevisionConflictError,
)
from services.export_service import ExportService

router = APIRouter(prefix="/api/exports", tags=["exports"])

###############################################################################
@router.get("/{project_id}/source/{code_target}", response_model=SourceExportResponse)
def export_source(
    project_id: UUID,
    code_target: CodeTarget,
    service: ExportService = Depends(get_export_service),
) -> SourceExportResponse:
    try:
        return service.source(project_id, code_target)
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
