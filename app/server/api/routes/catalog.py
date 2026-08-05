from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_catalog_service
from domain.manifests import NodeCatalogResponse, NodeManifest
from services.catalog_service import CatalogService

router = APIRouter(prefix="/api/nodes", tags=["node catalog"])


@router.get("/catalog", response_model=NodeCatalogResponse)
def catalog(service: CatalogService = Depends(get_catalog_service)) -> NodeCatalogResponse:
    return service.list()


@router.get("/{definition_id}/{version}", response_model=NodeManifest)
def manifest(
    definition_id: str,
    version: int,
    service: CatalogService = Depends(get_catalog_service),
) -> NodeManifest:
    try:
        return service.get(definition_id, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
