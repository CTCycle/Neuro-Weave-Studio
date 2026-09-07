from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CustomNodeKind = Literal["mathematical", "composite", "repository"]
CustomNodeOrigin = Literal["custom"]
_REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

###############################################################################
class MathematicalNodeSpec(BaseModel):
    type: Literal["mathematical"] = "mathematical"
    formula: str = Field(min_length=1, max_length=1200)
    expression: str | None = Field(default=None, max_length=500)
    input_symbols: list[Literal["x"]] = Field(default_factory=lambda: ["x"], min_length=1, max_length=1)

    # -------------------------------------------------------------------------
    @field_validator("formula")
    @classmethod
    def clean_formula(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Formula cannot be empty")
        return cleaned

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def single_tensor_input(self) -> MathematicalNodeSpec:
        if self.input_symbols != ["x"]:
            raise ValueError("Mathematical custom nodes currently use the single tensor input 'x'")
        return self

###############################################################################
class CompositeNodeComponent(BaseModel):
    definition_id: str = Field(min_length=1, max_length=160)
    definition_version: int = Field(default=1, ge=1)
    label: str | None = Field(default=None, max_length=160)
    parameters: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @field_validator("definition_id")
    @classmethod
    def clean_definition_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Component definition id cannot be empty")
        return cleaned

    # -------------------------------------------------------------------------
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
    resolved_commit: str | None = Field(default=None, min_length=40, max_length=40)
    library_name: str | None = Field(default=None, max_length=120)
    pipeline_tag: str | None = Field(default=None, max_length=160)

    # -------------------------------------------------------------------------
    @field_validator("repository_id")
    @classmethod
    def validate_repository_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not _REPOSITORY_ID_PATTERN.fullmatch(cleaned):
            raise ValueError("Hugging Face repository ids must use the namespace/repository form")
        return cleaned

    # -------------------------------------------------------------------------
    @field_validator("revision")
    @classmethod
    def clean_revision(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Revision cannot be empty")
        return cleaned

    # -------------------------------------------------------------------------
    @field_validator("entrypoint", "library_name", "pipeline_tag")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    # -------------------------------------------------------------------------
    @field_validator("resolved_commit")
    @classmethod
    def validate_resolved_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().casefold()
        if not _COMMIT_PATTERN.fullmatch(cleaned):
            raise ValueError("Resolved Hugging Face commit must be a full 40-character SHA")
        return cleaned


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

    # -------------------------------------------------------------------------
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
    version: int = Field(ge=1)
    display_name: str
    category: str
    description: str
    kind: CustomNodeKind
    spec: CustomNodeSpec
    origin: CustomNodeOrigin = "custom"
    locked: bool = False
    can_edit: bool = True
    can_delete: bool = True
    executable_template: bool = False
    created_at: datetime
    updated_at: datetime

###############################################################################
class CustomNodeCatalogResponse(BaseModel):
    definitions: list[CustomNodeDefinitionResponse]
