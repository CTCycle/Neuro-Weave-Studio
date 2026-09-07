from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_catalog_service, get_custom_node_service
from domain.custom_nodes import (
    CustomNodeCatalogResponse,
    CustomNodeDefinitionCreate,
    CustomNodeDefinitionResponse,
    CustomNodeDefinitionUpdate,
)
from domain.manifests import NodeCatalogResponse
from services.catalog_service import CatalogService
from services.custom_node_service import CustomNodeDefinitionConflictError, CustomNodeService
from services.huggingface_repository_service import HuggingFaceUnavailableError

router = APIRouter(prefix="/api/nodes", tags=["node catalog"])

###############################################################################
@router.get("/catalog", response_model=NodeCatalogResponse)
def catalog(service: CatalogService = Depends(get_catalog_service)) -> NodeCatalogResponse:
    return service.list()

###############################################################################
@router.get("/custom", response_model=CustomNodeCatalogResponse)
def custom_nodes(
    service: CustomNodeService = Depends(get_custom_node_service),
) -> CustomNodeCatalogResponse:
    return service.list()


###############################################################################
def _service_unavailable(exc: HuggingFaceUnavailableError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

###############################################################################
@router.post(
    "/custom",
    response_model=CustomNodeDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_custom_node(
    payload: CustomNodeDefinitionCreate,
    service: CustomNodeService = Depends(get_custom_node_service),
) -> CustomNodeDefinitionResponse:
    try:
        return service.create(payload)
    except HuggingFaceUnavailableError as exc:
        raise _service_unavailable(exc) from exc
    except CustomNodeDefinitionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

###############################################################################
@router.put("/custom/{definition_id}", response_model=CustomNodeDefinitionResponse)
def update_custom_node(
    definition_id: str,
    payload: CustomNodeDefinitionUpdate,
    service: CustomNodeService = Depends(get_custom_node_service),
) -> CustomNodeDefinitionResponse:
    try:
        return service.update(definition_id, payload)
    except HuggingFaceUnavailableError as exc:
        raise _service_unavailable(exc) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Custom node definition not found") from exc
    except CustomNodeDefinitionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

###############################################################################
@router.delete("/custom/{definition_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_node(
    definition_id: str,
    service: CustomNodeService = Depends(get_custom_node_service),
) -> Response:
    try:
        service.delete(definition_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Custom node definition not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
