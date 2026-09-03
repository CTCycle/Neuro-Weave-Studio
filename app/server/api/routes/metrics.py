from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_metric_service
from domain.metrics import (
    MetricCatalogResponse,
    MetricDefinitionCreate,
    MetricDefinitionResponse,
    MetricDefinitionUpdate,
)
from services.metric_service import MetricDefinitionConflictError, MetricService

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=MetricCatalogResponse)
def list_metrics(service: MetricService = Depends(get_metric_service)) -> MetricCatalogResponse:
    return service.list()


@router.post("", response_model=MetricDefinitionResponse, status_code=status.HTTP_201_CREATED)
def create_metric(
    payload: MetricDefinitionCreate,
    service: MetricService = Depends(get_metric_service),
) -> MetricDefinitionResponse:
    try:
        return service.create(payload)
    except MetricDefinitionConflictError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/{metric_id}", response_model=MetricDefinitionResponse)
def update_metric(
    metric_id: str,
    payload: MetricDefinitionUpdate,
    service: MetricService = Depends(get_metric_service),
) -> MetricDefinitionResponse:
    try:
        return service.update(metric_id, payload)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Metric definition not found") from exc
    except MetricDefinitionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "usages": [item.model_dump(mode="json") for item in exc.usages]},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/{metric_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_metric(metric_id: str, service: MetricService = Depends(get_metric_service)) -> Response:
    try:
        service.delete(metric_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Metric definition not found") from exc
    except MetricDefinitionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "usages": [item.model_dump(mode="json") for item in exc.usages]},
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
