from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_catalog_service
from domain.manifests import NodeCatalogResponse
from services.catalog_service import CatalogService

router = APIRouter(prefix="/api/nodes", tags=["node catalog"])


###############################################################################
@router.get("/catalog", response_model=NodeCatalogResponse)
def catalog(service: CatalogService = Depends(get_catalog_service)) -> NodeCatalogResponse:
    return service.list()
