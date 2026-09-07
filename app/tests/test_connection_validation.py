from common.catalogs.loader import get_node_catalog
from domain.diagnostics import ConnectionValidationResult
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
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
def proposed_edge(
    source_node_id: str,
    target_node_id: str,
    *,
    source_port_id: str = "output",
    target_port_id: str = "input",
) -> EdgeInstance:
    return EdgeInstance(
        id="candidate",
        source_node_id=source_node_id,
        source_port_id=source_port_id,
        target_node_id=target_node_id,
        target_port_id=target_port_id,
    )


###############################################################################
def validate_connection(graph: ModelGraph, edge: EdgeInstance) -> ConnectionValidationResult:
    return ValidationService(get_node_catalog()).validate_connection(
        graph,
        edge,
        ExecutionBackend.PYTORCH,
        True,
    )


###############################################################################
def test_connection_validation_allows_incomplete_draft_edges() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [8]}),
            make_node("dense", "layer.dense", {"units": 4}),
        ]
    )

    result = validate_connection(graph, proposed_edge("input", "dense"))

    assert result.valid
    assert not any(item.severity == DiagnosticSeverity.ERROR for item in result.diagnostics)


###############################################################################
def test_connection_validation_reports_expected_and_received_rank() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [8]}),
            make_node("conv", "conv.conv2d"),
        ]
    )

    result = validate_connection(graph, proposed_edge("input", "conv"))

    diagnostic = next(item for item in result.diagnostics if item.code == "port_rank_mismatch")
    assert not result.valid
    assert diagnostic.node_id == "conv"
    assert diagnostic.edge_id == "candidate"
    assert diagnostic.details["expected_rank"] == 4
    assert diagnostic.details["received_rank"] == 2
    assert diagnostic.details["received_shape"] == [None, 8]


###############################################################################
def test_connection_validation_rejects_duplicate_and_cycle_edges() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [8]}),
            make_node("dense", "layer.dense", {"units": 4}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="dense",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="dense",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

    duplicate = validate_connection(graph, proposed_edge("input", "dense"))
    cycle = validate_connection(graph, proposed_edge("output", "dense"))

    assert not duplicate.valid
    assert any(item.code == "duplicate_connection" for item in duplicate.diagnostics)
    assert not cycle.valid
    assert any(item.code == "cycle_detected" for item in cycle.diagnostics)


###############################################################################
def test_connection_validation_rejects_incompatible_merge_shape() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("first", "input.tabular", {"shape": [4]}),
            make_node("second", "input.tabular", {"shape": [6]}),
            make_node("merge", "merge.residual"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="first",
                source_port_id="output",
                target_node_id="merge",
                target_port_id="inputs",
            )
        ],
    )

    result = validate_connection(
        graph,
        proposed_edge("second", "merge", target_port_id="inputs"),
    )

    assert not result.valid
    diagnostic = next(
        item for item in result.diagnostics if item.code == "incompatible_connection_shape"
    )
    assert diagnostic.node_id == "merge"
    assert diagnostic.edge_id == "candidate"
    assert diagnostic.details["received_shape"] == [None, 6]
