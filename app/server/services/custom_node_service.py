from __future__ import annotations

from typing import cast
from uuid import uuid4

from common.catalogs.loader import NodeCatalog
from common.timestamps import utc_now
from domain.custom_math import CustomMathFormulaError, parse_custom_math_formula
from domain.custom_nodes import (
    CompositeNodeComponent,
    CompositeNodeSpec,
    CustomNodeCatalogResponse,
    CustomNodeDefinitionCreate,
    CustomNodeDefinitionResponse,
    CustomNodeDefinitionUpdate,
    CustomNodeSpec,
    MathematicalNodeSpec,
    RepositoryNodeSpec,
)
from domain.manifests import NodeManifest
from repositories.contracts import UnitOfWorkFactory, UnitOfWorkProtocol
from repositories.custom_nodes import CustomNodeDefinitionRecord
from services.huggingface_repository_service import HuggingFaceRepositoryService
from validation.parameter_values import (
    ParameterValueError,
    condition_matches,
    normalize_manifest_parameters,
)


class CustomNodeDefinitionConflictError(ValueError):
    pass


###############################################################################
class CustomNodeService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        catalog: NodeCatalog,
        repository_service: HuggingFaceRepositoryService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog
        self._repository_service = repository_service or HuggingFaceRepositoryService()

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
                version=1,
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
            record.version += 1
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
        if isinstance(spec, MathematicalNodeSpec):
            try:
                expression = parse_custom_math_formula(spec.formula)
            except CustomMathFormulaError as exc:
                raise ValueError(str(exc)) from exc
            return spec.model_copy(update={"expression": expression, "input_symbols": ["x"]})
        if isinstance(spec, RepositoryNodeSpec):
            metadata = self._repository_service.resolve(spec.repository_id, spec.revision)
            return spec.model_copy(
                update={
                    "resolved_commit": metadata.commit,
                    "library_name": metadata.library_name,
                    "pipeline_tag": metadata.pipeline_tag,
                }
            )
        if not isinstance(spec, CompositeNodeSpec):
            return spec

        normalized_components: list[CompositeNodeComponent] = []
        manifests: list[NodeManifest] = []
        for component in spec.components:
            try:
                manifest = self._catalog.get(component.definition_id, component.definition_version)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown built-in node definition: {component.definition_id}@{component.definition_version}"
                ) from exc
            if manifest.semantic_role != "operation" or manifest.deprecated:
                raise ValueError(
                    f"Composite templates may only use active operation nodes; {manifest.id} is not eligible"
                )
            try:
                parameters = normalize_manifest_parameters(manifest, component.parameters)
            except ParameterValueError as exc:
                raise ValueError(f"Invalid parameters for {manifest.id}: {exc}") from exc
            manifests.append(manifest)
            normalized_components.append(component.model_copy(update={"parameters": parameters}))

        for index in range(len(manifests) - 1):
            source_manifest = manifests[index]
            target_manifest = manifests[index + 1]
            source_parameters = normalized_components[index].parameters
            target_parameters = normalized_components[index + 1].parameters
            source_ports = [
                port
                for port in source_manifest.outputs
                if condition_matches(port.enabled_when, source_parameters)
            ]
            target_ports = [
                port
                for port in target_manifest.inputs
                if condition_matches(port.enabled_when, target_parameters)
            ]
            compatible = any(
                source_port.kind == target_port.kind
                or (
                    target_port.kind.value == "tensor_list"
                    and target_port.element_kind == source_port.kind
                )
                for source_port in source_ports
                for target_port in target_ports
            )
            if not compatible:
                raise ValueError(
                    f"Sequential components {source_manifest.id} and {target_manifest.id} do not expose compatible active ports"
                )
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
        return CustomNodeDefinitionResponse.model_validate(
            {
                "id": record.id,
                "version": record.version,
                "display_name": record.display_name,
                "category": record.category,
                "description": record.description,
                "kind": record.kind,
                "spec": record.spec_json,
                "executable_template": record.kind in {"mathematical", "composite"},
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
        )


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())
