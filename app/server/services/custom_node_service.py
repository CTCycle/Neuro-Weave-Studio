from __future__ import annotations

from typing import cast
from uuid import uuid4

from common.catalogs.loader import NodeCatalog
from common.timestamps import utc_now
from domain.custom_math import CustomMathFormulaError, parse_custom_math_formula
from domain.custom_nodes import (
    CustomNodeCatalogResponse,
    CustomNodeDefinitionCreate,
    CustomNodeDefinitionResponse,
    CustomNodeDefinitionUpdate,
    CustomNodeSpec,
    GraphTemplateNodeSpec,
    MathematicalNodeSpec,
    RepositoryNodeSpec,
)
from domain.manifests import NodeManifest
from domain.ports import materialized_inputs, materialized_outputs
from repositories.contracts import UnitOfWorkFactory, UnitOfWorkProtocol
from repositories.custom_nodes import CustomNodeDefinitionRecord
from services.huggingface_repository_service import HuggingFaceRepositoryService
from validation.parameter_values import (
    ParameterValueError,
    condition_matches,
    normalize_manifest_parameters,
)

###############################################################################
class CustomNodeDefinitionConflictError(ValueError):
    pass

###############################################################################
class CustomNodeService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        catalog: NodeCatalog,
        repository_service: HuggingFaceRepositoryService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog
        self._repository_service = repository_service or HuggingFaceRepositoryService()

    # -------------------------------------------------------------------------
    def list(self) -> CustomNodeCatalogResponse:
        with self._uow_factory() as uow:
            definitions = [self._from_record(record) for record in uow.custom_nodes.list_all()]
        return CustomNodeCatalogResponse(definitions=definitions)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def delete(self, definition_id: str) -> None:
        with self._uow_factory() as uow:
            record = uow.custom_nodes.get(definition_id)
            if record is None:
                if self._builtin(definition_id):
                    raise PermissionError("Built-in node definitions are locked")
                raise KeyError(definition_id)
            uow.custom_nodes.delete(record)
            uow.commit()

    # -------------------------------------------------------------------------
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
        if not isinstance(spec, GraphTemplateNodeSpec):
            raise TypeError(f"Unsupported custom node specification: {spec.type}")

        normalized_components = []
        manifests: dict[str, NodeManifest] = {}
        parameters_by_id: dict[str, dict[str, object]] = {}
        for component in spec.components:
            try:
                manifest = self._catalog.get(component.definition_id, component.definition_version)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown built-in node definition: {component.definition_id}@{component.definition_version}"
                ) from exc
            if manifest.deprecated:
                raise ValueError(
                    f"Graph templates may only use active built-in nodes; {manifest.id} is not eligible"
                )
            try:
                parameters = normalize_manifest_parameters(manifest, component.parameters)
            except ParameterValueError as exc:
                raise ValueError(f"Invalid parameters for {manifest.id}: {exc}") from exc
            manifests[component.id] = manifest
            parameters_by_id[component.id] = parameters
            normalized_components.append(component.model_copy(update={"parameters": parameters}))

        for edge in spec.edges:
            source_manifest = manifests[edge.source_component_id]
            target_manifest = manifests[edge.target_component_id]
            source_parameters = parameters_by_id[edge.source_component_id]
            target_parameters = parameters_by_id[edge.target_component_id]
            source_port = next(
                (
                    port
                    for port in materialized_outputs(source_manifest, source_parameters)
                    if port.id == edge.source_port_id
                    and condition_matches(port.enabled_when, source_parameters)
                ),
                None,
            )
            target_port = next(
                (
                    port
                    for port in materialized_inputs(target_manifest, target_parameters)
                    if port.id == edge.target_port_id
                    and condition_matches(port.enabled_when, target_parameters)
                ),
                None,
            )
            if source_port is None or target_port is None:
                raise ValueError(
                    f"Graph template edge references an unavailable port: "
                    f"{edge.source_component_id}.{edge.source_port_id} -> "
                    f"{edge.target_component_id}.{edge.target_port_id}"
                )
            expected_kind = (
                target_port.element_kind
                if target_port.kind.value == "tensor_list"
                else target_port.kind
            )
            if source_port.kind != expected_kind:
                raise ValueError(
                    f"Graph template edge {edge.source_component_id}.{edge.source_port_id} -> "
                    f"{edge.target_component_id}.{edge.target_port_id} has incompatible port kinds"
                )

        for binding in [*spec.input_bindings, *spec.output_bindings]:
            manifest = manifests[binding.component_id]
            parameters = parameters_by_id[binding.component_id]
            ports = (
                materialized_inputs(manifest, parameters)
                if binding in spec.input_bindings
                else materialized_outputs(manifest, parameters)
            )
            if not any(
                port.id == binding.port_id
                and condition_matches(port.enabled_when, parameters)
                for port in ports
            ):
                raise ValueError(
                    f"Graph template binding references an unavailable port: "
                    f"{binding.component_id}.{binding.port_id}"
                )
        return spec.model_copy(update={"components": normalized_components})

    # -------------------------------------------------------------------------
    def _assert_not_builtin_name(self, display_name: str) -> None:
        normalized = _normalized_name(display_name)
        if any(_normalized_name(item.display_name) == normalized for item in self._catalog.manifests):
            raise CustomNodeDefinitionConflictError(
                f"A built-in node named '{display_name}' already exists"
            )

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _builtin(self, definition_id: str) -> bool:
        return any(item.id == definition_id for item in self._catalog.manifests)

    # -------------------------------------------------------------------------
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
                "executable_template": record.kind in {"mathematical", "graph_template"},
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
        )

###############################################################################
def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())
