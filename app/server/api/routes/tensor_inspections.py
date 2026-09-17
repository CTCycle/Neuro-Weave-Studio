from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.dependencies import get_tensor_inspection_service
from domain.tensor_inspection import (
    TensorInspectionNode,
    TensorInspectionSession,
    TensorInspectionViewRequest,
    TensorInspectionViewResponse,
)
from services.tensor_inspection_service import TensorInspectionError, TensorInspectionService

router = APIRouter(prefix="/api/tensor-inspections", tags=["tensor-inspection"])


def _raise(exc: TensorInspectionError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc


@router.get("/{session_id}", response_model=TensorInspectionSession)
def get_session(
    session_id: str,
    service: TensorInspectionService = Depends(get_tensor_inspection_service),
) -> TensorInspectionSession:
    try:
        return service.get(session_id)
    except TensorInspectionError as exc:
        _raise(exc)


@router.get("/{session_id}/nodes/{node_id}", response_model=TensorInspectionNode)
def get_node(
    session_id: str,
    node_id: str,
    service: TensorInspectionService = Depends(get_tensor_inspection_service),
) -> TensorInspectionNode:
    try:
        return service.node(session_id, node_id)
    except TensorInspectionError as exc:
        _raise(exc)


@router.post("/{session_id}/view", response_model=TensorInspectionViewResponse)
def get_view(
    session_id: str,
    payload: TensorInspectionViewRequest,
    service: TensorInspectionService = Depends(get_tensor_inspection_service),
) -> TensorInspectionViewResponse:
    try:
        return service.view(session_id, payload)
    except TensorInspectionError as exc:
        _raise(exc)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    service: TensorInspectionService = Depends(get_tensor_inspection_service),
) -> Response:
    try:
        service.delete(session_id)
    except TensorInspectionError as exc:
        _raise(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
