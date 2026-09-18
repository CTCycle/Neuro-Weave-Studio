from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_dataset_service
from domain.datasets import (
    DatasetDefinitionMutation,
    DatasetDocument,
    DatasetInspection,
    DatasetInspectionRequest,
    DatasetListResponse,
    DatasetValidationResult,
)
from services.dataset_inspection_service import DatasetInspectionError
from services.dataset_service import DatasetDefinitionInvalidError, DatasetService

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


###############################################################################
@router.post("/inspect", response_model=DatasetInspection)
def inspect_dataset(
    payload: DatasetInspectionRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetInspection:
    try:
        return service.inspect(payload)
    except DatasetInspectionError as exc:
        raise HTTPException(status_code=422, detail={"code": "inspection_failed", "message": str(exc)}) from exc


###############################################################################
@router.post("/validate", response_model=DatasetValidationResult)
def validate_dataset(
    payload: DatasetDefinitionMutation,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetValidationResult:
    return service.validate(payload)


###############################################################################
@router.post("/{dataset_id}/validate", response_model=DatasetDocument)
def revalidate_dataset(
    dataset_id: UUID,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetDocument:
    dataset = service.revalidate(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


###############################################################################
@router.get("", response_model=DatasetListResponse)
def list_datasets(service: DatasetService = Depends(get_dataset_service)) -> DatasetListResponse:
    return service.list()


###############################################################################
@router.post("", response_model=DatasetDocument, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetDefinitionMutation,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetDocument:
    try:
        return service.create(payload)
    except DatasetDefinitionInvalidError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "dataset_definition_invalid", "validation": exc.result.model_dump(mode="json")},
        ) from exc


###############################################################################
@router.get("/{dataset_id}", response_model=DatasetDocument)
def get_dataset(
    dataset_id: UUID,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetDocument:
    dataset = service.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


###############################################################################
@router.put("/{dataset_id}", response_model=DatasetDocument)
def update_dataset(
    dataset_id: UUID,
    payload: DatasetDefinitionMutation,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetDocument:
    if service.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        dataset = service.update(dataset_id, payload)
    except DatasetDefinitionInvalidError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "dataset_definition_invalid", "validation": exc.result.model_dump(mode="json")},
        ) from exc
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


###############################################################################
@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: UUID,
    service: DatasetService = Depends(get_dataset_service),
) -> Response:
    if not service.delete(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
