from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from domain.manifests import NodeManifest, PortDefinition

###############################################################################
def materialize_ports(
    definitions: Iterable[PortDefinition], parameters: dict[str, Any] | None = None
) -> list[PortDefinition]:
    """Expand declarative parameter-cardinality ports for one node instance.

    Manifests keep a single port template. All graph-facing callers should use
    this helper so dynamic ports have the same identity in validation, shape
    inference, source generation, inspection, and output selection.
    """

    values = parameters or {}
    materialized: list[PortDefinition] = []
    for definition in definitions:
        cardinality = definition.cardinality
        if cardinality.kind == "single":
            materialized.append(definition)
            continue
        assert cardinality.parameter_id is not None
        value = values.get(cardinality.parameter_id)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Port cardinality parameter '{cardinality.parameter_id}' must be an integer"
            )
        if value < cardinality.minimum or value > cardinality.maximum:
            raise ValueError(
                f"Port cardinality parameter '{cardinality.parameter_id}' must be between "
                f"{cardinality.minimum} and {cardinality.maximum}"
            )
        for index in range(value):
            materialized.append(
                definition.model_copy(
                    update={
                        "id": cardinality.id_pattern.format(
                            id=definition.id,
                            display_name=definition.display_name,
                            index=index,
                        ),
                        "display_name": cardinality.display_name_pattern.format(
                            id=definition.id,
                            display_name=definition.display_name,
                            index=index,
                        ),
                        "cardinality": {"kind": "single"},
                    }
                )
            )
    return materialized

###############################################################################
def materialized_inputs(
    manifest: NodeManifest, parameters: dict[str, Any] | None = None
) -> list[PortDefinition]:
    return materialize_ports(manifest.inputs, parameters)

###############################################################################
def materialized_outputs(
    manifest: NodeManifest, parameters: dict[str, Any] | None = None
) -> list[PortDefinition]:
    return materialize_ports(manifest.outputs, parameters)

