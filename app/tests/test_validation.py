from common.catalogs.loader import NodeCatalog, get_node_catalog
from domain.enums import CodeTarget, DataType, DiagnosticSeverity, PortKind
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.tensor import TensorSpec
from services.validation_service import ValidationService
from validation.shape_inference import infer_shapes


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

###############################################################################
def _two_input_graph(
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
            EdgeInstance(id="e1", source_node_id="first", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="e2", source_node_id="second", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="e3", source_node_id="merge", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )

###############################################################################
def test_merge_rejects_same_shape_different_dtype() -> None:
    graph = _two_input_graph(
        "merge.add",
        {"shape": [None, 4], "dtype": "float32"},
        {"shape": [None, 4], "dtype": "int32"},
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any("compatible shapes" in item.message for item in result.diagnostics)

###############################################################################
def test_concatenate_rejects_non_concatenated_dimension_mismatch() -> None:
    graph = _two_input_graph(
        "merge.concatenate",
        {"shape": [None, 8, 16], "dtype": "float32"},
        {"shape": [None, 9, 17], "dtype": "float32"},
        {"axis": -1},
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any("non-concatenated" in item.message for item in result.diagnostics)

###############################################################################
def test_stack_increases_rank_in_shape_inference() -> None:
    graph = _two_input_graph(
        "merge.stack",
        {"shape": [None, 4], "dtype": "float32"},
        {"shape": [None, 4], "dtype": "float32"},
        {"axis": 1},
    )
    summaries = infer_shapes(graph, get_node_catalog())
    merge = next(item for item in summaries if item.node_id == "merge")
    assert merge.outputs["output"].shape == [None, 2, 4]

###############################################################################
def test_conv_rejects_stride_and_dilation_combination() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.image", {"shape": [None, 16, 16, 3], "dtype": "float32"}),
            make_node(
                "conv",
                "conv.conv2d",
                {"filters": 4, "kernel_size": [3, 3], "strides": [2, 2], "dilation_rate": [2, 2]},
            ),
            make_node("output", "output.classification"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="conv", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="conv", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any("strides > 1" in item.message for item in result.diagnostics)

###############################################################################
def test_tensor_specs_require_matching_dtype() -> None:
    left = TensorSpec(shape=[None, 4], dtype=DataType.FLOAT32)
    right = TensorSpec(shape=[None, 4], dtype=DataType.INT32)
    assert not left.compatible_with(right)

###############################################################################
def test_declared_port_rank_is_enforced_after_shape_inference() -> None:
    catalog = get_node_catalog()
    input_manifest = catalog.get("input.tabular")
    ranked_input = input_manifest.model_copy(
        update={
            "outputs": [
                input_manifest.outputs[0].model_copy(update={"rank": 3}),
            ]
        }
    )
    test_catalog = NodeCatalog(
        [ranked_input if manifest.id == "input.tabular" else manifest for manifest in catalog.manifests]
    )
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
    result = ValidationService(test_catalog).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any(item.code == "port_rank_mismatch" for item in result.diagnostics)

###############################################################################
def test_tensor_list_declares_and_enforces_element_kind() -> None:
    catalog = get_node_catalog()
    merge_manifest = catalog.get("merge.add")
    tensor_list_input = merge_manifest.inputs[0].model_copy(update={"element_kind": PortKind.MASK})
    masked_merge = merge_manifest.model_copy(update={"inputs": [tensor_list_input]})
    test_catalog = NodeCatalog(
        [masked_merge if manifest.id == "merge.add" else manifest for manifest in catalog.manifests]
    )
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            make_node("merge", "merge.add"),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="e2", source_node_id="merge", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    result = ValidationService(test_catalog).validate(graph, CodeTarget.KERAS)
    assert not result.valid
    assert any(item.code == "incompatible_port_kind" for item in result.diagnostics)

###############################################################################
def test_transposed_convolution_uses_declared_dimension_parameter() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [None, 8, 3], "dtype": "float32"}),
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
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="transposed", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="transposed", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    summaries = infer_shapes(graph, get_node_catalog())
    transposed = next(item for item in summaries if item.node_id == "transposed")
    assert transposed.outputs["output"].shape == [None, 10, 4]
