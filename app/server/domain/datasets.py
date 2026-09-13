from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


###############################################################################
class DatasetType(StrEnum):
    TABULAR = "tabular"
    IMAGES = "images"
    TEXT = "text"
    TIME_SERIES = "time_series"
    ARRAY = "array"
    COMPOSED = "composed"


###############################################################################
class DatasetValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


###############################################################################
class DatasetAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


###############################################################################
class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


###############################################################################
class TableSource(DatasetModel):
    type: Literal["table"] = "table"
    id: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    format: Literal["csv", "tsv", "parquet"]
    has_header: bool = True
    delimiter: Literal[",", "\t", ";", "|"] | None = None


###############################################################################
class ImageFolderSource(DatasetModel):
    type: Literal["image_folder"] = "image_folder"
    id: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    recursive: bool = True
    label_mode: Literal["none", "parent_folder"] = "none"


###############################################################################
class TextSource(DatasetModel):
    type: Literal["text"] = "text"
    id: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    path_kind: Literal["file", "folder"]
    recursive: bool = True


###############################################################################
class ArraySource(DatasetModel):
    type: Literal["array"] = "array"
    id: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    format: Literal["npy", "npz"]


DatasetSource = Annotated[
    TableSource | ImageFolderSource | TextSource | ArraySource,
    Field(discriminator="type"),
]


###############################################################################
class ColumnBinding(DatasetModel):
    type: Literal["columns"] = "columns"
    source_id: str = Field(min_length=1, max_length=80)
    columns: list[str] = Field(min_length=1)


###############################################################################
class ImageBinding(DatasetModel):
    type: Literal["image"] = "image"
    source_id: str = Field(min_length=1, max_length=80)


###############################################################################
class TextBinding(DatasetModel):
    type: Literal["text"] = "text"
    source_id: str = Field(min_length=1, max_length=80)


###############################################################################
class ArrayBinding(DatasetModel):
    type: Literal["array"] = "array"
    source_id: str = Field(min_length=1, max_length=80)
    keys: list[str] = Field(min_length=1)


###############################################################################
class FolderLabelBinding(DatasetModel):
    type: Literal["folder_label"] = "folder_label"
    source_id: str = Field(min_length=1, max_length=80)


DatasetBinding = Annotated[
    ColumnBinding | ImageBinding | TextBinding | ArrayBinding | FolderLabelBinding,
    Field(discriminator="type"),
]


###############################################################################
class DatasetRoleDefinition(DatasetModel):
    bindings: list[DatasetBinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


###############################################################################
class FileTableRelationship(DatasetModel):
    type: Literal["file_table"] = "file_table"
    file_source_id: str = Field(min_length=1, max_length=80)
    file_key: Literal["filename", "stem", "relative_path"] = "filename"
    table_source_id: str = Field(min_length=1, max_length=80)
    table_key_column: str = Field(min_length=1, max_length=512)


DatasetRelationship = FileTableRelationship


###############################################################################
class TimeSeriesStructure(DatasetModel):
    timestamp_column: str | None = Field(default=None, max_length=512)
    group_column: str | None = Field(default=None, max_length=512)


###############################################################################
class AllSamplesSubset(DatasetModel):
    type: Literal["all"] = "all"


###############################################################################
class FirstNSamplesSubset(DatasetModel):
    type: Literal["first_n"] = "first_n"
    count: int = Field(ge=1)


###############################################################################
class RangeSamplesSubset(DatasetModel):
    type: Literal["range"] = "range"
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def ordered_range(self) -> RangeSamplesSubset:
        if self.end < self.start:
            raise ValueError("Sample range end must be greater than or equal to start")
        return self


DatasetSubsetDefinition = Annotated[
    AllSamplesSubset | FirstNSamplesSubset | RangeSamplesSubset,
    Field(discriminator="type"),
]


###############################################################################
class DatasetDefinitionMutation(DatasetModel):
    name: str = Field(min_length=1, max_length=160)
    dataset_type: DatasetType
    sources: list[DatasetSource] = Field(min_length=1, max_length=3)
    input_definition: DatasetRoleDefinition
    output_definition: DatasetRoleDefinition = Field(default_factory=DatasetRoleDefinition)
    metadata_definition: DatasetRoleDefinition = Field(default_factory=DatasetRoleDefinition)
    relationships: list[DatasetRelationship] = Field(default_factory=list, max_length=1)  # pyright: ignore[reportUnknownVariableType]
    sequence_definition: TimeSeriesStructure | None = None
    subset_definition: DatasetSubsetDefinition = Field(default_factory=AllSamplesSubset)

    # -------------------------------------------------------------------------
    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Dataset name cannot be empty")
        return cleaned


###############################################################################
class DatasetInspectionRequest(DatasetModel):
    dataset_type: DatasetType
    sources: list[DatasetSource] = Field(min_length=1, max_length=3)


###############################################################################
class ColumnMetadata(DatasetModel):
    name: str
    dtype: str
    nullable: bool = False


###############################################################################
class ArrayMetadata(DatasetModel):
    key: str
    shape: list[int]
    dtype: str


###############################################################################
class TableInspection(DatasetModel):
    type: Literal["table"] = "table"
    source_id: str
    path: str
    format: Literal["csv", "tsv", "parquet"]
    sample_count: int = Field(ge=0)
    columns: list[ColumnMetadata]
    inference_sample_size: int = Field(ge=0)


###############################################################################
class ImageFolderInspection(DatasetModel):
    type: Literal["image_folder"] = "image_folder"
    source_id: str
    path: str
    sample_count: int = Field(ge=0)
    extensions: list[str]
    class_names: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)


###############################################################################
class TextInspection(DatasetModel):
    type: Literal["text"] = "text"
    source_id: str
    path: str
    sample_count: int = Field(ge=0)
    extensions: list[str]


###############################################################################
class ArrayInspection(DatasetModel):
    type: Literal["array"] = "array"
    source_id: str
    path: str
    format: Literal["npy", "npz"]
    sample_count: int = Field(ge=0)
    arrays: list[ArrayMetadata]


DatasetSourceInspection = Annotated[
    TableInspection | ImageFolderInspection | TextInspection | ArrayInspection,
    Field(discriminator="type"),
]


###############################################################################
class DatasetInspection(DatasetModel):
    dataset_type: DatasetType
    sample_count: int = Field(ge=0)
    sources: list[DatasetSourceInspection]
    warnings: list[str] = Field(default_factory=list)


###############################################################################
class DatasetValidationFinding(DatasetModel):
    code: str
    message: str
    source_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class DatasetValidationMetadata(DatasetModel):
    status: DatasetValidationStatus
    checked_at: datetime
    errors: list[DatasetValidationFinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    warnings: list[DatasetValidationFinding] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


###############################################################################
class DatasetValidationResult(DatasetModel):
    valid: bool
    sample_count: int = Field(ge=0)
    selected_sample_count: int = Field(ge=0)
    inspection: DatasetInspection | None = None
    validation: DatasetValidationMetadata


###############################################################################
class DatasetAvailabilityState(DatasetModel):
    status: DatasetAvailability
    unavailable_source_ids: list[str] = Field(default_factory=list)


###############################################################################
class DatasetSummary(DatasetModel):
    id: UUID
    name: str
    dataset_type: DatasetType
    format: str
    primary_path: str
    sample_count: int = Field(ge=0)
    selected_sample_count: int = Field(ge=0)
    input_summary: str
    output_summary: str
    validation_status: DatasetValidationStatus
    availability: DatasetAvailabilityState
    created_at: datetime
    updated_at: datetime


###############################################################################
class DatasetListResponse(DatasetModel):
    datasets: list[DatasetSummary]


###############################################################################
class DatasetDocument(DatasetDefinitionMutation):
    id: UUID
    format: str
    sample_count: int = Field(ge=0)
    selected_sample_count: int = Field(ge=0)
    inspection: DatasetInspection
    validation: DatasetValidationMetadata
    availability: DatasetAvailabilityState
    created_at: datetime
    updated_at: datetime
