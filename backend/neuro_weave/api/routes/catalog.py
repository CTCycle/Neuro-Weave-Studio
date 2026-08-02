from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from neuro_weave.api.dependencies import get_node_catalog
from neuro_weave.catalog.loader import NodeCatalog
from neuro_weave.domain.manifests import NodeCatalogResponse, NodeManifest

router = APIRouter(prefix="/api/nodes", tags=["node catalog"])


@router.get("/catalog", response_model=NodeCatalogResponse)
def catalog(node_catalog: NodeCatalog = Depends(get_node_catalog)) -> NodeCatalogResponse:
    return NodeCatalogResponse(nodes=node_catalog.manifests)


@router.get("/{definition_id}/{version}", response_model=NodeManifest)
def manifest(
    definition_id: str,
    version: int,
    node_catalog: NodeCatalog = Depends(get_node_catalog),
) -> NodeManifest:
    try:
        return node_catalog.get(definition_id, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
