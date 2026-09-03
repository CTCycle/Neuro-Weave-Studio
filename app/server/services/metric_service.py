from __future__ import annotations

from uuid import uuid4

from common.timestamps import utc_now
from domain.compilation_catalog import CompilationCatalogProvider, CompilationOption
from domain.metric_formula import MetricFormulaError, parse_metric_formula
from domain.metrics import (
    MetricCatalogResponse,
    MetricDefinitionCreate,
    MetricDefinitionResponse,
    MetricDefinitionUpdate,
    MetricUsageResponse,
)
from repositories.contracts import UnitOfWorkFactory, UnitOfWorkProtocol
from repositories.models import CustomMetricDefinitionRecord


class MetricDefinitionConflictError(ValueError):
    def __init__(self, message: str, *, usages: list[MetricUsageResponse] | None = None) -> None:
        super().__init__(message)
        self.usages = usages or []


class MetricService:
    def __init__(self, uow_factory: UnitOfWorkFactory, catalog_provider: CompilationCatalogProvider) -> None:
        self._uow_factory = uow_factory
        self._catalog_provider = catalog_provider

    def list(self) -> MetricCatalogResponse:
        catalog = self._catalog_provider.get()
        definitions = [
            self._from_option("metric", option) for option in catalog.metrics
        ] + [
            self._from_option("loss", option) for option in catalog.losses
        ]
        return MetricCatalogResponse(definitions=definitions)

    def create(self, payload: MetricDefinitionCreate) -> MetricDefinitionResponse:
        parsed = _parse(payload.formula)
        normalized_label = _normalized_label(payload.label)
        with self._uow_factory() as uow:
            if self._label_taken(uow, payload.kind, normalized_label):
                raise MetricDefinitionConflictError(
                    f"A {payload.kind} named '{payload.label.strip()}' already exists"
                )
            now = utc_now()
            record = CustomMetricDefinitionRecord(
                id=f"custom-{uuid4().hex}",
                kind=payload.kind,
                label=payload.label.strip(),
                normalized_label=normalized_label,
                description=payload.description.strip(),
                formula_latex=payload.formula.strip(),
                expression_json=parsed.to_json(),
                created_at=now,
                updated_at=now,
            )
            uow.metrics.add(record)
            uow.commit()
            return self._from_record(record)

    def update(self, metric_id: str, payload: MetricDefinitionUpdate) -> MetricDefinitionResponse:
        parsed = _parse(payload.formula)
        normalized_label = _normalized_label(payload.label)
        with self._uow_factory() as uow:
            record = uow.metrics.get(metric_id)
            if record is None:
                if self._builtin(metric_id):
                    raise PermissionError("Built-in definitions are locked")
                raise KeyError(metric_id)
            if self._label_taken(uow, record.kind, normalized_label, exclude_id=metric_id):
                raise MetricDefinitionConflictError(
                    f"A {record.kind} named '{payload.label.strip()}' already exists"
                )
            record.label = payload.label.strip()
            record.normalized_label = normalized_label
            record.description = payload.description.strip()
            record.formula_latex = payload.formula.strip()
            record.expression_json = parsed.to_json()
            record.updated_at = utc_now()
            usages = uow.metrics.referencing_projects(metric_id)
            if usages:
                uow.code_snapshots.delete_for_projects([usage.project_id for usage in usages])
            uow.commit()
            return self._from_record(record)

    def delete(self, metric_id: str) -> None:
        with self._uow_factory() as uow:
            record = uow.metrics.get(metric_id)
            if record is None:
                if self._builtin(metric_id):
                    raise PermissionError("Built-in definitions are locked")
                raise KeyError(metric_id)
            usages = uow.metrics.referencing_projects(metric_id)
            if usages:
                references = [
                    MetricUsageResponse(project_id=item.project_id, project_name=item.project_name)
                    for item in usages
                ]
                raise MetricDefinitionConflictError(
                    "This definition is used by saved models and cannot be deleted.",
                    usages=references,
                )
            uow.metrics.delete(record)
            uow.commit()

    @staticmethod
    def _label_taken(uow: UnitOfWorkProtocol, kind: str, normalized: str, exclude_id: str | None = None) -> bool:
        return any(
            item.kind == kind and item.normalized_label == normalized and item.id != exclude_id
            for item in uow.metrics.list_all()
        )

    @staticmethod
    def _from_record(record: CustomMetricDefinitionRecord) -> MetricDefinitionResponse:
        return MetricDefinitionResponse(
            id=record.id,
            kind=record.kind,
            label=record.label,
            description=record.description,
            formula=record.formula_latex,
            origin="custom",
            locked=False,
            can_edit=True,
            can_delete=True,
            supported_tasks=["regression", "binary_classification", "multiclass_classification"],
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _from_option(kind: str, option: CompilationOption) -> MetricDefinitionResponse:
        return MetricDefinitionResponse(
            id=option.id,
            kind=kind,  # type: ignore[arg-type]
            label=option.label,
            description=option.description,
            formula=option.formula,
            origin=option.origin,
            locked=option.locked,
            can_edit=option.can_edit,
            can_delete=option.can_delete,
            supported_tasks=option.supported_tasks,
            target_contract=option.target_contract,
            created_at=option.created_at,
            updated_at=option.updated_at,
        )

    def _builtin(self, metric_id: str) -> bool:
        catalog = self._catalog_provider.get()
        return any(
            option.id == metric_id and option.origin == "builtin"
            for option in (*catalog.metrics, *catalog.losses)
        )


def _normalized_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _parse(formula: str):
    try:
        return parse_metric_formula(formula)
    except MetricFormulaError:
        raise
