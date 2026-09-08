from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from domain.compilation_catalog import CompilationTask, TargetContract

MetricKind = Literal["metric", "loss"]
MetricOrigin = Literal["builtin", "custom"]


class MetricDefinitionResponse(BaseModel):
    id: str
    kind: MetricKind
    label: str
    description: str
    formula: str | None = None
    origin: MetricOrigin
    locked: bool
    can_edit: bool
    can_delete: bool
    supported_tasks: list[CompilationTask] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    target_contract: TargetContract | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MetricCatalogResponse(BaseModel):
    definitions: list[MetricDefinitionResponse]


class MetricDefinitionCreate(BaseModel):
    kind: MetricKind
    label: str
    description: str
    formula: str

    @field_validator("label", "description", "formula")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty")
        return cleaned


class MetricDefinitionUpdate(BaseModel):
    label: str
    description: str
    formula: str

    @field_validator("label", "description", "formula")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty")
        return cleaned


class MetricUsageResponse(BaseModel):
    project_id: UUID
    project_name: str
