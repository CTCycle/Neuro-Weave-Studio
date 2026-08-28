from common.catalogs.loader import get_node_catalog
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from services.validation_service import ValidationService
from validation.shape_inference import infer_shapes

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
def two_input_graph(
    definition_id: str,
    first: dict[str, object],
    second: dict[str, object],
    parameters: dict[str, object] | None = None,
) -> ModelGraph:
    return ModelGraph(
        nodes=[
            make_node("first", "input.tabular", first),
            make_node("second", "input.tabular", second),
            make_node("merge", definition_id, parameters),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="first",
                source_port_id="output",
                target_node_id="merge",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="second",
                source_port_id="output",
                target_node_id="merge",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="e3",
                source_node_id="merge",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

###############################################################################
def test_valid_mlp_propagates_shapes_and_counts_parameters() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                {"shape": [None, 20], "dtype": "float32"},
            ),
            make_node(
                "dense",
                "layer.dense",
                {"units": 3, "activation": "softmax", "use_bias": True},
            ),
            make_node("output", "output.classification"),
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

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid
    assert result.total_parameters == 63
    dense = next(item for item in result.shape_summaries if item.node_id == "dense")
    assert dense.outputs["output"].shape == [None, 3]

###############################################################################
def test_cycle_is_reported_as_error() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                {"shape": [None, 4], "dtype": "float32"},
            ),
            make_node("a", "layer.dense", {"units": 4}),
            make_node("b", "layer.dense", {"units": 4}),
            make_node("output", "output.classification"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="a",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="a",
                source_port_id="output",
                target_node_id="b",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e3",
                source_node_id="b",
                source_port_id="output",
                target_node_id="a",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e4",
                source_node_id="b",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        item.code == "cycle_detected"
        and item.severity == DiagnosticSeverity.ERROR
        for item in result.diagnostics
    )

###############################################################################
def test_unsafe_math_expression_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tensor",
                {"shape": [None, 4], "dtype": "float32"},
            ),
            make_node(
                "expression",
                "math.expression",
                {"expression": "__import__('os')"},
            ),
            make_node("output", "output.custom"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="expression",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="expression",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(item.code == "unsafe_expression" for item in result.diagnostics)

###############################################################################
def test_declared_parameter_contract_rejects_null_and_unknown_values() -> None:
    def graph(parameters: dict[str, object]) -> ModelGraph:
        return ModelGraph(
            nodes=[
                make_node(
                    "input",
                    "input.tabular",
                    {"shape": [None, 4], "dtype": "float32"},
                ),
                make_node("dense", "layer.dense", parameters),
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

    null_result = ValidationService(get_node_catalog()).validate(
        graph({"units": 2, "use_bias": None}),
        ExecutionBackend.PYTORCH,
        True,
    )
    unknown_result = ValidationService(get_node_catalog()).validate(
        graph({"units": 2, "strieds": 2}),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert any(
        item.code == "non_nullable_parameter" and item.parameter_id == "use_bias"
        for item in null_result.diagnostics
    )
    assert any(
        item.code == "unknown_parameter" and item.parameter_id == "strieds"
        for item in unknown_result.diagnostics
    )

###############################################################################
def test_native_backend_rejects_parameter_values_without_verified_mapping() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                {"shape": [None, 4], "dtype": "float32"},
            ),
            make_node(
                "dense",
                "layer.dense",
                {"units": 2, "kernel_initializer": "he_normal"},
            ),
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

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        False,
    )

    assert not result.valid
    assert any(
        item.code == "unsupported_parameter_value"
        and item.parameter_id == "kernel_initializer"
        for item in result.diagnostics
    )

###############################################################################
def test_merge_contracts_reject_incompatible_tensor_inputs() -> None:
    dtype_mismatch = two_input_graph(
        "merge.add",
        {"shape": [None, 4], "dtype": "float32"},
        {"shape": [None, 4], "dtype": "int32"},
    )
    dimension_mismatch = two_input_graph(
        "merge.concatenate",
        {"shape": [None, 8, 16], "dtype": "float32"},
        {"shape": [None, 9, 17], "dtype": "float32"},
        {"axis": -1},
    )

    dtype_result = ValidationService(get_node_catalog()).validate(
        dtype_mismatch,
        ExecutionBackend.PYTORCH,
        True,
    )
    dimension_result = ValidationService(get_node_catalog()).validate(
        dimension_mismatch,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not dtype_result.valid
    assert any("compatible shapes" in item.message for item in dtype_result.diagnostics)
    assert not dimension_result.valid
    assert any(
        "non-concatenated" in item.message for item in dimension_result.diagnostics
    )

###############################################################################
def test_convolution_rejects_stride_and_dilation_combination() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.image",
                {"shape": [None, 16, 16, 3], "dtype": "float32"},
            ),
            make_node(
                "conv",
                "conv.conv2d",
                {
                    "filters": 4,
                    "kernel_size": [3, 3],
                    "strides": [2, 2],
                    "dilation_rate": [2, 2],
                },
            ),
            make_node("output", "output.classification"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="conv",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="conv",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any("strides > 1" in item.message for item in result.diagnostics)

###############################################################################
def test_unreachable_model_input_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "used",
                "input.tabular",
                {"shape": [None, 4], "dtype": "float32"},
            ),
            make_node(
                "unused",
                "input.tabular",
                {"shape": [None, 4], "dtype": "float32"},
            ),
            make_node("dense", "layer.dense", {"units": 2}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="used",
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

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        item.code == "unreachable_model_input" and item.node_id == "unused"
        for item in result.diagnostics
    )

###############################################################################
def test_transposed_convolution_uses_declared_dimension_parameter() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.sequence",
                {"shape": [None, 8, 3], "dtype": "float32"},
            ),
            make_node(
                "transposed",
                "conv.transposed",
                {
                    "dimensions": 1,
                    "filters": 4,
                    "kernel_size": [3],
                    "strides": [1],
                    "padding": "valid",
                },
            ),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="transposed",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="transposed",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )

    summaries = infer_shapes(graph, get_node_catalog())
    transposed = next(item for item in summaries if item.node_id == "transposed")

    assert transposed.outputs["output"].shape == [None, 10, 4]
