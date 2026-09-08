from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_runtime_service
from domain.metadata import MetadataResponse
from services.runtime_service import RuntimeService

router = APIRouter(prefix="/api/metadata", tags=["metadata"])


###############################################################################
@router.get("", response_model=MetadataResponse)
def metadata(service: RuntimeService = Depends(get_runtime_service)) -> MetadataResponse:
    return service.metadata()
