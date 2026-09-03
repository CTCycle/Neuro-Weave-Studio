from __future__ import annotations

from typing import cast
from uuid import uuid4

from common.catalogs.loader import NodeCatalog
from common.timestamps import utc_now
from domain.custom_nodes import (
    CompositeNodeSpec,
    CustomNodeCatalogResponse,
    CustomNodeDefinitionCreate,
    CustomNodeDefinitionResponse,
    CustomNodeDefinitionUpdate,
    CustomNodeSpec,
)
from repositories.contracts import UnitOfWorkFactory, UnitOfWorkProtocol
from repositories.custom_nodes import CustomNodeDefinitionRecord


class CustomNodeDefinitionConflictError(ValueError):
    pass


###############################################################################
class CustomNodeService:
    def __init__(self, uow_factory: UnitOfWorkFactory, catalog: NodeCatalog) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog

    def list(self) -> CustomNodeCatalogResponse:
        with self._uow_factory() as uow:
            definitions = [self._from_record(record) for record in uow.custom_nodes.list_all()]
        return CustomNodeCatalogResponse(definitions=definitions)

    def create(self, payload: CustomNodeDefinitionCreate) -> CustomNodeDefinitionResponse:
        normalized_name = _normalized_name(payload.display_name)
        self._assert_not_builtin_name(payload.display_name)
        spec = self._normalize_spec(payload.spec)
        with self._uow_factory() as uow:
            self._assert_name_available(uow, normalized_name)
            now = utc_now()
            record = CustomNodeDefinitionRecord(
                id=f"custom.node.{uuid4().hex}",
                kind=spec.type,
                display_name=payload.display_name,
                normalized_name=normalized_name,
                category=payload.category,
                description=payload.description,
                spec_json=cast(dict[str, object], spec.model_dump(mode="json")),
                created_at=now,
                updated_at=now,
            )
            uow.custom_nodes.add(record)
            uow.commit()
            return self._from_record(record)

    def update(
        self,
        definition_id: str,
        payload: CustomNodeDefinitionUpdate,
    ) -> CustomNodeDefinitionResponse:
        normalized_name = _normalized_name(payload.display_name)
        self._assert_not_builtin_name(payload.display_name)
        spec = self._normalize_spec(payload.spec)
        with self._uow_factory() as uow:
            record = uow.custom_nodes.get(definition_id)
            if record is None:
                if self._builtin(definition_id):
                    raise PermissionError("Built-in node definitions are locked")
                raise KeyError(definition_id)
            self._assert_name_available(uow, normalized_name, exclude_id=definition_id)
            record.kind = spec.type
            record.display_name = payload.display_name
            record.normalized_name = normalized_name
            record.category = payload.category
            record.description = payload.description
            record.spec_json = cast(dict[str, object], spec.model_dump(mode="json"))
            record.updated_at = utc_now()
            uow.commit()
            return self._from_record(record)

    def delete(self, definition_id: str) -> None:
        with self._uow_factory() as uow:
            record = uow.custom_nodes.get(definition_id)
            if record is None:
                if self._builtin(definition_id):
                    raise PermissionError("Built-in node definitions are locked")
                raise KeyError(definition_id)
            uow.custom_nodes.delete(record)
            uow.commit()

    def _normalize_spec(self, spec: CustomNodeSpec) -> CustomNodeSpec:
        if not isinstance(spec, CompositeNodeSpec):
            return spec
        normalized_components = []
        for component in spec.components:
            try:
                manifest = self._catalog.get(component.definition_id, component.definition_version)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown built-in node definition: {component.definition_id}@{component.definition_version}"
                ) from exc
            known_parameters = {parameter.id: parameter for parameter in manifest.parameters}
            unknown_parameters = sorted(set(component.parameters) - set(known_parameters))
            if unknown_parameters:
                raise ValueError(
                    f"Unknown parameters for {manifest.id}: {', '.join(unknown_parameters)}"
                )
            values: dict[str, object] = {}
            for parameter in manifest.parameters:
                default = parameter.editor_default
                if default is None:
                    default = parameter.default
                values[parameter.id] = default
            values.update(component.parameters)
            normalized_components.append(component.model_copy(update={"parameters": values}))
        return spec.model_copy(update={"components": normalized_components})

    def _assert_not_builtin_name(self, display_name: str) -> None:
        normalized = _normalized_name(display_name)
        if any(_normalized_name(item.display_name) == normalized for item in self._catalog.manifests):
            raise CustomNodeDefinitionConflictError(
                f"A built-in node named '{display_name}' already exists"
            )

    @staticmethod
    def _assert_name_available(
        uow: UnitOfWorkProtocol,
        normalized_name: str,
        *,
        exclude_id: str | None = None,
    ) -> None:
        if any(
            record.normalized_name == normalized_name and record.id != exclude_id
            for record in uow.custom_nodes.list_all()
        ):
            raise CustomNodeDefinitionConflictError("A custom node with this name already exists")

    def _builtin(self, definition_id: str) -> bool:
        return any(item.id == definition_id for item in self._catalog.manifests)

    @staticmethod
    def _from_record(record: CustomNodeDefinitionRecord) -> CustomNodeDefinitionResponse:
        return CustomNodeDefinitionResponse(
            id=record.id,
            display_name=record.display_name,
            category=record.category,
            description=record.description,
            kind=cast(str, record.kind),  # type: ignore[arg-type]
            spec=cast(CustomNodeSpec, record.spec_json),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())
