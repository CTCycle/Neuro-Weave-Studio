from common.catalogs.loader import get_node_catalog
from domain.enums import CodeTarget, DiagnosticSeverity
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from services.validation_service import ValidationService


###############################################################################
def make_node(node_id: str, definition_id: str, parameters: dict[str, object] | None = None) -> NodeInstance:
    return NodeInstance(id=node_id, definition_id=definition_id, position=Position(x=0, y=0), parameters=parameters or {})

###############################################################################
def test_valid_mlp_propagates_shapes_and_counts_parameters() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 20], "dtype": "float32"}),
            make_node("dense", "layer.dense", {"units": 3, "activation": "softmax", "use_bias": True}),
            make_node("output", "output.classification"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert result.valid
    assert result.total_parameters == 63
    dense_summary = next(item for item in result.shape_summaries if item.node_id == "dense")
    assert dense_summary.outputs["output"].shape == [None, 3]

###############################################################################
def test_cycle_is_reported_as_error() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            make_node("a", "layer.dense", {"units": 4}),
            make_node("b", "layer.dense", {"units": 4}),
            make_node("output", "output.classification"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="a", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="a", source_port_id="output", target_node_id="b", target_port_id="input"),
            EdgeInstance(id="e3", source_node_id="b", source_port_id="output", target_node_id="a", target_port_id="input"),
            EdgeInstance(id="e4", source_node_id="b", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any(item.code == "cycle_detected" and item.severity == DiagnosticSeverity.ERROR for item in result.diagnostics)

###############################################################################
def test_unsafe_math_expression_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tensor", {"shape": [None, 4], "dtype": "float32"}),
            make_node("expression", "math.expression", {"expression": "__import__('os')"}),
            make_node("output", "output.custom"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="expression", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="expression", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any(item.code == "unsafe_expression" for item in result.diagnostics)

###############################################################################
def test_pooling_updates_spatial_shape() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("image", "input.image", {"shape": [None, 32, 32, 3], "dtype": "float32"}),
            make_node("pool", "conv.max_pool2d", {"pool_size": [2, 2]}),
            make_node("output", "output.custom"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="image", source_port_id="output", target_node_id="pool", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="pool", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert result.valid
    pool = next(item for item in result.shape_summaries if item.node_id == "pool")
    assert pool.outputs["output"].shape == [None, 16, 16, 3]

###############################################################################
def test_optional_non_nullable_parameter_may_be_omitted() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            make_node("dense", "layer.dense", {"units": 2}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert result.valid

###############################################################################
def test_explicit_null_for_non_nullable_parameter_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            make_node("dense", "layer.dense", {"units": 2, "use_bias": None}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    diagnostic = next(item for item in result.diagnostics if item.code == "non_nullable_parameter")
    assert diagnostic.parameter_id == "use_bias"
    assert diagnostic.code_target is CodeTarget.KERAS

###############################################################################
def test_unknown_v2_parameter_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            make_node("dense", "layer.dense", {"units": 2, "strieds": 2}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    diagnostic = next(item for item in result.diagnostics if item.code == "unknown_parameter")
    assert diagnostic.parameter_id == "strieds"
