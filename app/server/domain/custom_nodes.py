from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

CustomNodeKind = Literal["mathematical", "composite", "repository"]
CustomNodeOrigin = Literal["custom"]
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


###############################################################################
class MathematicalNodeSpec(BaseModel):
    type: Literal["mathematical"] = "mathematical"
    formula: str = Field(min_length=1, max_length=8000)
    input_symbols: list[str] = Field(default_factory=lambda: ["x"], min_length=1, max_length=16)

    @field_validator("formula")
    @classmethod
    def clean_formula(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Formula cannot be empty")
        return cleaned

    @field_validator("input_symbols")
    @classmethod
    def validate_input_symbols(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not _IDENTIFIER_PATTERN.fullmatch(item) for item in cleaned):
            raise ValueError("Input symbols must be valid identifiers")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Input symbols must be unique")
        return cleaned


###############################################################################
class CompositeNodeComponent(BaseModel):
    definition_id: str = Field(min_length=1, max_length=160)
    definition_version: int = Field(default=1, ge=1)
    label: str | None = Field(default=None, max_length=160)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("definition_id")
    @classmethod
    def clean_definition_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


###############################################################################
class CompositeNodeSpec(BaseModel):
    type: Literal["composite"] = "composite"
    connection_mode: Literal["sequential"] = "sequential"
    components: list[CompositeNodeComponent] = Field(min_length=1, max_length=64)


###############################################################################
class RepositoryNodeSpec(BaseModel):
    type: Literal["repository"] = "repository"
    provider: Literal["huggingface"] = "huggingface"
    repository_id: str = Field(min_length=3, max_length=240)
    revision: str = Field(default="main", min_length=1, max_length=160)
    artifact_type: Literal["model", "layer"] = "model"
    entrypoint: str | None = Field(default=None, max_length=240)

    @field_validator("repository_id")
    @classmethod
    def validate_repository_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not _REPOSITORY_ID_PATTERN.fullmatch(cleaned):
            raise ValueError("Hugging Face repository ids must use the namespace/repository form")
        return cleaned

    @field_validator("revision")
    @classmethod
    def clean_revision(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Revision cannot be empty")
        return cleaned

    @field_validator("entrypoint")
    @classmethod
    def clean_entrypoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


CustomNodeSpec = Annotated[
    MathematicalNodeSpec | CompositeNodeSpec | RepositoryNodeSpec,
    Field(discriminator="type"),
]


###############################################################################
class CustomNodeDefinitionCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=160)
    category: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=4000)
    spec: CustomNodeSpec

    @field_validator("display_name", "category", "description")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty")
        return cleaned


###############################################################################
class CustomNodeDefinitionUpdate(CustomNodeDefinitionCreate):
    pass


###############################################################################
class CustomNodeDefinitionResponse(BaseModel):
    id: str
    version: int = 1
    display_name: str
    category: str
    description: str
    kind: CustomNodeKind
    spec: CustomNodeSpec
    origin: CustomNodeOrigin = "custom"
    locked: bool = False
    can_edit: bool = True
    can_delete: bool = True
    created_at: datetime
    updated_at: datetime


###############################################################################
class CustomNodeCatalogResponse(BaseModel):
    definitions: list[CustomNodeDefinitionResponse]
