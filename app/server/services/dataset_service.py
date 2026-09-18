from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from common.timestamps import utc_now
from domain.datasets import (
    AllSamplesSubset,
    ArrayBinding,
    ArrayInspection,
    ArraySource,
    ColumnBinding,
    DatasetAvailability,
    DatasetAvailabilityState,
    DatasetBinding,
    DatasetDefinitionMutation,
    DatasetDocument,
    DatasetInspection,
    DatasetInspectionRequest,
    DatasetListResponse,
    DatasetRoleDefinition,
    DatasetSplitDefinition,
    DatasetSummary,
    DatasetType,
    DatasetValidationFinding,
    DatasetValidationMetadata,
    DatasetValidationResult,
    DatasetValidationStatus,
    FirstNSamplesSubset,
    FolderLabelBinding,
    ImageBinding,
    ImageFolderInspection,
    ImageFolderSource,
    TableInspection,
    TableSource,
    TextBinding,
    TextInspection,
    TextSource,
)
from repositories.contracts import UnitOfWorkFactory
from repositories.datasets import DatasetRecord
from services.dataset_inspection_service import DatasetInspectionError, DatasetInspectionService


###############################################################################
class DatasetDefinitionInvalidError(ValueError):

    # -------------------------------------------------------------------------
    def __init__(self, result: DatasetValidationResult) -> None:
        self.result = result
        super().__init__("Dataset definition is invalid")


###############################################################################
class DatasetService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        inspector: DatasetInspectionService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._inspector = inspector or DatasetInspectionService()

    # -------------------------------------------------------------------------
    def inspect(self, request: DatasetInspectionRequest) -> DatasetInspection:
        return self._inspector.inspect(request)

    # -------------------------------------------------------------------------
    def validate(self, payload: DatasetDefinitionMutation) -> DatasetValidationResult:
        now = utc_now()
        errors: list[DatasetValidationFinding] = []
        warnings: list[DatasetValidationFinding] = []
        try:
            inspection = self._inspector.inspect(
                DatasetInspectionRequest(dataset_type=payload.dataset_type, sources=payload.sources)
            )
        except DatasetInspectionError as exc:
            errors.append(DatasetValidationFinding(code="source_inspection_failed", message=str(exc)))
            return self._validation_result(now, None, 0, 0, errors, warnings)

        errors.extend(self._validate_layout(payload))
        errors.extend(self._validate_bindings(payload, inspection))
        sample_count = self._resolve_sample_count(payload, inspection, errors)
        selected_count = self._validate_subset(payload, sample_count, errors)
        errors.extend(self._validate_split(payload.split_definition))
        if sample_count == 0:
            errors.append(
                DatasetValidationFinding(
                    code="empty_dataset",
                    message="Dataset contains no samples",
                )
            )
        warnings.extend(
            DatasetValidationFinding(code="inspection_warning", message=message)
            for message in inspection.warnings
        )
        return self._validation_result(
            now,
            inspection,
            sample_count,
            selected_count,
            errors,
            warnings,
        )

    # -------------------------------------------------------------------------
    def list(self) -> DatasetListResponse:
        with self._uow_factory() as uow:
            records = uow.datasets.list_all()
            datasets = [self._summary(record) for record in records]
        return DatasetListResponse(datasets=datasets)

    # -------------------------------------------------------------------------
    def get(self, dataset_id: UUID) -> DatasetDocument | None:
        with self._uow_factory() as uow:
            record = uow.datasets.get(dataset_id)
            return None if record is None else self._document(record)

    # -------------------------------------------------------------------------
    def create(self, payload: DatasetDefinitionMutation) -> DatasetDocument:
        result = self.validate(payload)
        if not result.valid or result.inspection is None:
            raise DatasetDefinitionInvalidError(result)
        payload = self._normalize_payload(payload, result.inspection)
        now = utc_now()
        record = DatasetRecord(
            id=str(uuid4()),
            name=payload.name,
            dataset_type=payload.dataset_type.value,
            format=self._format_label(payload),
            sample_count=result.sample_count,
            selected_sample_count=result.selected_sample_count,
            validation_status=DatasetValidationStatus.VALID.value,
            definition_json=payload.model_dump(mode="json"),
            inspection_json=result.inspection.model_dump(mode="json"),
            validation_json=result.validation.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
        )
        with self._uow_factory() as uow:
            uow.datasets.add(record)
            uow.commit()
            return self._document(record)

    # -------------------------------------------------------------------------
    def update(
        self,
        dataset_id: UUID,
        payload: DatasetDefinitionMutation,
    ) -> DatasetDocument | None:
        with self._uow_factory() as uow:
            record = uow.datasets.get(dataset_id)
            if record is None:
                return None
            registered_type = record.dataset_type
        result = self.validate(payload)
        if registered_type != payload.dataset_type.value:
            result = self._validation_result(
                result.validation.checked_at,
                result.inspection,
                result.sample_count,
                result.selected_sample_count,
                [
                    *result.validation.errors,
                    DatasetValidationFinding(
                        code="dataset_type_immutable",
                        message="Dataset type cannot change after registration; register a new dataset instead",
                        details={"registered": registered_type, "requested": payload.dataset_type.value},
                    ),
                ],
                result.validation.warnings,
            )
        if not result.valid or result.inspection is None:
            raise DatasetDefinitionInvalidError(result)
        payload = self._normalize_payload(payload, result.inspection)
        with self._uow_factory() as uow:
            record = uow.datasets.get(dataset_id)
            if record is None:
                return None
            record.name = payload.name
            record.dataset_type = payload.dataset_type.value
            record.format = self._format_label(payload)
            record.sample_count = result.sample_count
            record.selected_sample_count = result.selected_sample_count
            record.validation_status = DatasetValidationStatus.VALID.value
            record.definition_json = payload.model_dump(mode="json")
            record.inspection_json = result.inspection.model_dump(mode="json")
            record.validation_json = result.validation.model_dump(mode="json")
            record.updated_at = utc_now()
            uow.commit()
            return self._document(record)

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: UUID) -> bool:
        with self._uow_factory() as uow:
            record = uow.datasets.get(dataset_id)
            if record is None:
                return False
            uow.datasets.delete(record)
            uow.commit()
            return True

    # -------------------------------------------------------------------------
    def revalidate(self, dataset_id: UUID) -> DatasetDocument | None:
        """Reinspect a stored definition and persist the live validation state."""
        with self._uow_factory() as uow:
            record = uow.datasets.get(dataset_id)
            if record is None:
                return None

            payload = DatasetDefinitionMutation.model_validate(record.definition_json)
            result = self.validate(payload)
            if result.inspection is not None:
                payload = self._normalize_payload(payload, result.inspection)
                record.definition_json = payload.model_dump(mode="json")
                record.sample_count = result.sample_count
                record.selected_sample_count = result.selected_sample_count
                record.inspection_json = result.inspection.model_dump(mode="json")

            record.validation_status = result.validation.status.value
            record.validation_json = result.validation.model_dump(mode="json")
            record.updated_at = utc_now()
            uow.commit()
            return self._document(record)

    # -------------------------------------------------------------------------
    def _validate_layout(self, payload: DatasetDefinitionMutation) -> list[DatasetValidationFinding]:
        errors: list[DatasetValidationFinding] = []
        tables = [source for source in payload.sources if isinstance(source, TableSource)]
        images = [source for source in payload.sources if isinstance(source, ImageFolderSource)]
        texts = [source for source in payload.sources if isinstance(source, TextSource)]
        arrays = [source for source in payload.sources if isinstance(source, ArraySource)]
        expected = {
            DatasetType.TABULAR: len(tables) == 1 and len(payload.sources) == 1,
            DatasetType.TIME_SERIES: len(tables) == 1 and len(payload.sources) == 1,
            DatasetType.ARRAY: bool(arrays) and len(arrays) == len(payload.sources),
            DatasetType.TEXT: len(payload.sources) == 1 and len(tables) + len(texts) == 1,
            DatasetType.IMAGES: len(images) == 1 and len(payload.sources) in {1, 2} and len(arrays) + len(texts) == 0 and len(tables) == len(payload.sources) - 1,
            DatasetType.COMPOSED: len(payload.sources) == 2 and len(tables) == 1 and len(images) + len(texts) == 1,
        }[payload.dataset_type]
        if not expected:
            errors.append(
                DatasetValidationFinding(
                    code="invalid_source_layout",
                    message=f"Source layout is not supported for dataset type '{payload.dataset_type.value}'",
                )
            )
        if payload.dataset_type is not DatasetType.TIME_SERIES and payload.sequence_definition is not None:
            errors.append(
                DatasetValidationFinding(
                    code="unexpected_sequence_definition",
                    message="Sequence columns are only valid for time-series datasets",
                )
            )
        needs_relationship = payload.dataset_type is DatasetType.COMPOSED or (
            payload.dataset_type is DatasetType.IMAGES and len(payload.sources) == 2
        )
        if needs_relationship != (len(payload.relationships) == 1):
            errors.append(
                DatasetValidationFinding(
                    code="invalid_relationship_count",
                    message="This source layout requires exactly one file-to-table relationship" if needs_relationship else "This source layout does not use a relationship",
                )
            )
        return errors

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_split(
        split: DatasetSplitDefinition | None,
    ) -> list[DatasetValidationFinding]:
        if split is None:
            return []

        errors: list[DatasetValidationFinding] = []
        fractions = {
            "train": split.train_fraction,
            "validation": split.validation_fraction,
            "test": split.test_fraction,
        }
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in fractions.values()):
            errors.append(
                DatasetValidationFinding(
                    code="invalid_split_fraction",
                    message="Train, validation, and test fractions must be between 0 and 1",
                )
            )
        elif not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            errors.append(
                DatasetValidationFinding(
                    code="split_fractions_not_one",
                    message="Train, validation, and test fractions must sum to 1",
                    details={"sum": sum(fractions.values())},
                )
            )
        if split.strategy == "random" and (split.seed is None or split.seed < 0):
            errors.append(
                DatasetValidationFinding(
                    code="invalid_split_seed",
                    message="Random splits require a non-negative deterministic seed",
                )
            )
        return errors

    # -------------------------------------------------------------------------
    def _validate_bindings(
        self,
        payload: DatasetDefinitionMutation,
        inspection: DatasetInspection,
    ) -> list[DatasetValidationFinding]:
        errors: list[DatasetValidationFinding] = []
        sources = {source.id: source for source in payload.sources}
        inspected = {item.source_id: item for item in inspection.sources}
        if not payload.input_definition.bindings:
            errors.append(DatasetValidationFinding(code="missing_inputs", message="Define at least one model input"))

        assigned: dict[tuple[str, str], str] = {}
        for role, definition in (
            ("input", payload.input_definition),
            ("output", payload.output_definition),
            ("metadata", payload.metadata_definition),
        ):
            for binding in definition.bindings:
                source = sources.get(binding.source_id)
                source_inspection = inspected.get(binding.source_id)
                if source is None or source_inspection is None:
                    errors.append(
                        DatasetValidationFinding(
                            code="unknown_binding_source",
                            message=f"{role.title()} binding references an unknown source",
                            source_id=binding.source_id,
                        )
                    )
                    continue
                binding_keys = self._binding_keys(binding, source_inspection, role, errors)
                for key in binding_keys:
                    identity = (binding.source_id, key)
                    previous = assigned.get(identity)
                    if previous is not None:
                        errors.append(
                            DatasetValidationFinding(
                                code="conflicting_assignment" if previous != role else "duplicate_assignment",
                                message=f"'{key}' is assigned as both {previous} and {role}" if previous != role else f"'{key}' is assigned more than once as {role}",
                                source_id=binding.source_id,
                            )
                        )
                    else:
                        assigned[identity] = role

        structural: list[tuple[str, str, str]] = []
        if payload.sequence_definition is not None and payload.sources:
            table_id = payload.sources[0].id
            if payload.sequence_definition.timestamp_column:
                structural.append((table_id, payload.sequence_definition.timestamp_column, "timestamp"))
            if payload.sequence_definition.group_column:
                structural.append((table_id, payload.sequence_definition.group_column, "group"))
        for relationship in payload.relationships:
            structural.append((relationship.table_source_id, relationship.table_key_column, "relationship key"))
        structural_columns = [(source_id, column) for source_id, column, _role in structural]
        if len(structural_columns) != len(set(structural_columns)):
            errors.append(
                DatasetValidationFinding(
                    code="conflicting_structural_columns",
                    message="Timestamp, group, and relationship columns must be distinct",
                )
            )
        for source_id, column, role in structural:
            table = inspected.get(source_id)
            if not isinstance(table, TableInspection) or column not in {item.name for item in table.columns}:
                errors.append(
                    DatasetValidationFinding(
                        code="missing_structural_column",
                        message=f"Configured {role} column does not exist: {column}",
                        source_id=source_id,
                    )
                )
            previous = assigned.get((source_id, column))
            if previous is not None:
                errors.append(
                    DatasetValidationFinding(
                        code="conflicting_structural_assignment",
                        message=f"'{column}' cannot be both {role} and {previous}",
                        source_id=source_id,
                    )
                )
        return errors

    # -------------------------------------------------------------------------
    @staticmethod
    def _binding_keys(
        binding: DatasetBinding,
        source_inspection: object,
        role: str,
        errors: list[DatasetValidationFinding],
    ) -> list[str]:
        if isinstance(binding, ColumnBinding):
            if not isinstance(source_inspection, TableInspection):
                errors.append(DatasetValidationFinding(code="invalid_binding_type", message=f"Column {role}s require a table source", source_id=binding.source_id))
                return []
            available = {item.name for item in source_inspection.columns}
            missing = [column for column in binding.columns if column not in available]
            if missing:
                errors.append(DatasetValidationFinding(code="missing_columns", message=f"Columns do not exist: {', '.join(missing)}", source_id=binding.source_id))
            return binding.columns
        if isinstance(binding, ArrayBinding):
            if not isinstance(source_inspection, ArrayInspection):
                errors.append(DatasetValidationFinding(code="invalid_binding_type", message=f"Array {role}s require an array source", source_id=binding.source_id))
                return []
            available = {item.key for item in source_inspection.arrays}
            missing = [key for key in binding.keys if key not in available]
            if missing:
                errors.append(DatasetValidationFinding(code="missing_arrays", message=f"Arrays do not exist: {', '.join(missing)}", source_id=binding.source_id))
            return binding.keys
        if isinstance(binding, ImageBinding):
            if role != "input":
                errors.append(DatasetValidationFinding(code="invalid_image_role", message="Image samples can only be model inputs", source_id=binding.source_id))
            elif not isinstance(source_inspection, ImageFolderInspection):
                errors.append(DatasetValidationFinding(code="invalid_binding_type", message="Image inputs require an image-folder source", source_id=binding.source_id))
        if isinstance(binding, TextBinding):
            if role != "input":
                errors.append(DatasetValidationFinding(code="invalid_text_role", message="Text samples can only be model inputs", source_id=binding.source_id))
            elif not isinstance(source_inspection, TextInspection):
                errors.append(DatasetValidationFinding(code="invalid_binding_type", message="Text-file inputs require a text source", source_id=binding.source_id))
        if isinstance(binding, FolderLabelBinding) and (
            role != "output"
            or not isinstance(source_inspection, ImageFolderInspection)
            or not source_inspection.class_names
        ):
            errors.append(DatasetValidationFinding(code="invalid_folder_labels", message="Folder labels require a labeled image-folder output", source_id=binding.source_id))
        return [binding.type]

    # -------------------------------------------------------------------------
    def _resolve_sample_count(
        self,
        payload: DatasetDefinitionMutation,
        inspection: DatasetInspection,
        errors: list[DatasetValidationFinding],
    ) -> int:
        if payload.relationships:
            relationship = payload.relationships[0]
            sources = {source.id: source for source in payload.sources}
            file_source = sources.get(relationship.file_source_id)
            table_source = sources.get(relationship.table_source_id)
            if not isinstance(file_source, (ImageFolderSource, TextSource)) or not isinstance(table_source, TableSource):
                errors.append(DatasetValidationFinding(code="invalid_relationship_sources", message="Relationship must connect an image/text file source to a table source"))
                return 0
            try:
                file_keys = self._inspector.file_keys(file_source, relationship.file_key)
                table_keys, duplicates = self._inspector.table_identifiers(table_source, relationship.table_key_column, relationship.file_key)
            except DatasetInspectionError as exc:
                errors.append(DatasetValidationFinding(code="relationship_inspection_failed", message=str(exc)))
                return 0
            if duplicates:
                errors.append(DatasetValidationFinding(code="duplicate_table_identifiers", message="Table identifier values must be unique", source_id=table_source.id, details={"examples": duplicates}))
            missing_table = sorted(file_keys - table_keys)[:10]
            missing_files = sorted(table_keys - file_keys)[:10]
            if missing_table:
                errors.append(DatasetValidationFinding(code="files_without_rows", message=f"{len(file_keys - table_keys)} files have no matching table row", details={"examples": missing_table}))
            if missing_files:
                errors.append(DatasetValidationFinding(code="rows_without_files", message=f"{len(table_keys - file_keys)} table rows have no matching file", details={"examples": missing_files}))
            return len(file_keys & table_keys)

        selected_arrays: list[int] = []
        inspections = {item.source_id: item for item in inspection.sources}
        for definition in (
            payload.input_definition,
            payload.output_definition,
            payload.metadata_definition,
        ):
            for binding in definition.bindings:
                if not isinstance(binding, ArrayBinding):
                    continue
                source = inspections.get(binding.source_id)
                if isinstance(source, ArrayInspection):
                    sizes = {item.shape[0] for item in source.arrays if item.key in binding.keys and item.shape}
                    selected_arrays.extend(sizes)
        if selected_arrays and len(set(selected_arrays)) > 1:
            errors.append(DatasetValidationFinding(code="array_sample_mismatch", message="Selected arrays do not share the same leading sample dimension"))
            return min(selected_arrays)
        return selected_arrays[0] if selected_arrays else inspection.sample_count

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_subset(
        payload: DatasetDefinitionMutation,
        sample_count: int,
        errors: list[DatasetValidationFinding],
    ) -> int:
        subset = payload.subset_definition
        if isinstance(subset, AllSamplesSubset):
            return sample_count
        if isinstance(subset, FirstNSamplesSubset):
            if subset.count > sample_count:
                errors.append(DatasetValidationFinding(code="subset_out_of_bounds", message=f"First N cannot exceed {sample_count} available samples"))
            return min(subset.count, sample_count)
        if subset.start >= sample_count or subset.end >= sample_count:
            errors.append(DatasetValidationFinding(code="subset_out_of_bounds", message=f"Sample range must stay between 0 and {max(sample_count - 1, 0)}"))
        return max(0, min(subset.end, sample_count - 1) - subset.start + 1)

    # -------------------------------------------------------------------------
    @staticmethod
    def _validation_result(
        checked_at: datetime,
        inspection: DatasetInspection | None,
        sample_count: int,
        selected_count: int,
        errors: list[DatasetValidationFinding],
        warnings: list[DatasetValidationFinding],
    ) -> DatasetValidationResult:
        valid = not errors
        metadata = DatasetValidationMetadata(
            status=DatasetValidationStatus.VALID if valid else DatasetValidationStatus.INVALID,
            checked_at=checked_at,
            errors=errors,
            warnings=warnings,
        )
        return DatasetValidationResult(
            valid=valid,
            sample_count=sample_count,
            selected_sample_count=selected_count,
            inspection=inspection,
            validation=metadata,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _availability(payload: DatasetDefinitionMutation) -> DatasetAvailabilityState:
        unavailable: list[str] = []
        for source in payload.sources:
            path = Path(source.path).expanduser()
            if not path.exists() or not os.access(path, os.R_OK):
                unavailable.append(source.id)
        return DatasetAvailabilityState(
            status=DatasetAvailability.UNAVAILABLE if unavailable else DatasetAvailability.AVAILABLE,
            unavailable_source_ids=unavailable,
        )

    # -------------------------------------------------------------------------
    def _summary(self, record: DatasetRecord) -> DatasetSummary:
        payload = DatasetDefinitionMutation.model_validate(record.definition_json)
        return DatasetSummary(
            id=UUID(record.id),
            name=record.name,
            dataset_type=payload.dataset_type,
            format=record.format,
            primary_path=payload.sources[0].path,
            sample_count=record.sample_count,
            selected_sample_count=record.selected_sample_count,
            input_summary=self._role_summary(payload.input_definition),
            output_summary=self._role_summary(payload.output_definition, empty="None (input only)"),
            validation_status=DatasetValidationStatus(record.validation_status),
            availability=self._availability(payload),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_payload(
        payload: DatasetDefinitionMutation,
        inspection: DatasetInspection,
    ) -> DatasetDefinitionMutation:
        """Persist resolved source paths while retaining the typed definition."""
        resolved_paths = {item.source_id: item.path for item in inspection.sources}
        data = payload.model_dump(mode="json")
        for source in data["sources"]:
            source_id = source["id"]
            resolved = resolved_paths.get(source_id)
            if resolved is not None:
                source["path"] = resolved
        return DatasetDefinitionMutation.model_validate(data)

    # -------------------------------------------------------------------------
    def _document(self, record: DatasetRecord) -> DatasetDocument:
        payload = DatasetDefinitionMutation.model_validate(record.definition_json)
        return DatasetDocument.model_validate(
            {
                **payload.model_dump(mode="json"),
                "id": record.id,
                "format": record.format,
                "sample_count": record.sample_count,
                "selected_sample_count": record.selected_sample_count,
                "inspection": record.inspection_json,
                "validation": record.validation_json,
                "availability": self._availability(payload),
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _role_summary(definition: DatasetRoleDefinition, *, empty: str = "None") -> str:
        parts: list[str] = []
        for binding in definition.bindings:
            if isinstance(binding, ColumnBinding):
                parts.append(f"{len(binding.columns)} column{'s' if len(binding.columns) != 1 else ''}")
            elif isinstance(binding, ArrayBinding):
                parts.append(f"{len(binding.keys)} array{'s' if len(binding.keys) != 1 else ''}")
            elif isinstance(binding, ImageBinding):
                parts.append("image")
            elif isinstance(binding, TextBinding):
                parts.append("text")
            else:
                parts.append("folder label")
        return " + ".join(parts) or empty

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_label(payload: DatasetDefinitionMutation) -> str:
        labels: list[str] = []
        for source in payload.sources:
            if isinstance(source, TableSource):
                label = source.format.upper()
            elif isinstance(source, ImageFolderSource):
                label = "Images"
            elif isinstance(source, TextSource):
                label = "Text files"
            else:
                label = source.format.upper()
            if label not in labels:
                labels.append(label)
        return " + ".join(labels)
