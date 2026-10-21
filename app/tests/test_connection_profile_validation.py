# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from services.connection_validation_service import ConnectionValidationService
from services.validation_service import ValidationService


###############################################################################
def make_node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, object] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def make_service() -> ConnectionValidationService:
    catalog = get_node_catalog()
    return ConnectionValidationService(catalog, ValidationService(catalog))

###############################################################################
def test_proposed_edge_respects_execution_profile() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [12, 8]}),
            make_node("rnn", "recurrent.simple_rnn", {"units": 4}),
        ]
    )
    connection = EdgeInstance(
        id="candidate",
        source_node_id="input",
        source_port_id="output",
        target_node_id="rnn",
        target_port_id="input",
    )
    service = make_service()

    keras_result = service.validate_connection(
        graph,
        connection,
        ExecutionBackend.TENSORFLOW,
        True,
    )
    native_tensorflow_result = service.validate_connection(
        graph,
        connection,
        ExecutionBackend.TENSORFLOW,
        False,
    )

    assert keras_result.valid
    assert native_tensorflow_result.valid
