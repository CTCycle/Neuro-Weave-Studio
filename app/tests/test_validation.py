# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

import pytest

from common.catalogs.loader import get_node_catalog
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, OutputBinding, Position
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
def attention_graph(
    definition_id: str,
    *,
    include_context: bool = False,
    query_shape: list[int] | None = None,
    context_shape: list[int] | None = None,
    parameters: dict[str, object] | None = None,
) -> ModelGraph:
    configuration: dict[str, object] = {
        "num_heads": 2,
        "key_dim": 4,
        "value_dim": 4,
        "layers": 1,
        "use_bias": True,
    }
    if parameters:
        configuration.update(parameters)
    nodes = [
        make_node(
            "query",
            "input.sequence",
            {"shape": query_shape or [5, 8], "dtype": "float32"},
        ),
    ]
    edges = [
        EdgeInstance(
            id="query-attention",
            source_node_id="query",
            source_port_id="output",
            target_node_id="attention",
            target_port_id="query",
        ),
    ]
    if include_context:
        nodes.append(
            make_node(
                "context",
                "input.sequence",
                {"shape": context_shape or [3, 8], "dtype": "float32"},
            )
        )
        edges.append(
            EdgeInstance(
                id="context-attention",
                source_node_id="context",
                source_port_id="output",
                target_node_id="attention",
                target_port_id="context",
            )
        )
    nodes.extend(
        [
            make_node("attention", definition_id, configuration),
            make_node("output", "output.custom"),
        ]
    )
    edges.append(
        EdgeInstance(
            id="attention-output",
            source_node_id="attention",
            source_port_id="output",
            target_node_id="output",
            target_port_id="input",
        )
    )
    return ModelGraph(nodes=nodes, edges=edges)

###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "include_context", "parameters", "expected_shape", "expected_count"),
    [
        (
            "attention.self",
            False,
            {},
            [None, 5, 8],
            288,
        ),
        (
            "attention.cross",
            True,
            {},
            [None, 5, 8],
            288,
        ),
        (
            "attention.multi_head",
            False,
            {"output_shape": [2, 4]},
            [None, 5, 2, 4],
            288,
        ),
        (
            "attention.self",
            False,
            {"use_bias": False},
            [None, 5, 8],
            256,
        ),
    ],
)
def test_attention_primitives_infer_shapes_and_bias_aware_parameter_counts(
    definition_id: str,
    include_context: bool,
    parameters: dict[str, object],
    expected_shape: list[int | None],
    expected_count: int,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph(
            definition_id,
            include_context=include_context,
            parameters=parameters,
        ),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid, result.diagnostics
    summary = next(item for item in result.shape_summaries if item.node_id == "attention")
    assert summary.outputs["output"].shape == expected_shape
    assert summary.parameter_count == expected_count
    assert result.total_parameters == expected_count

###############################################################################
def test_attention_primitive_rejects_layers_above_one() -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph("attention.self", parameters={"layers": 2}),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        "The supported multi-head attention node accepts layers=1" in item.message
        for item in result.diagnostics
    )

###############################################################################
def test_attention_primitive_rejects_non_rank_three_query() -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph("attention.self", query_shape=[8]),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        "Attention requires rank 3: batch, sequence, embedding" in item.message
        for item in result.diagnostics
    )

###############################################################################
@pytest.mark.parametrize("output_shape", [[], [0], [-1]])
def test_attention_primitive_rejects_empty_or_non_positive_output_shape(
    output_shape: list[int],
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph(
            "attention.multi_head",
            parameters={"output_shape": output_shape},
        ),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        item.parameter_id == "output_shape"
        and (
            "Attention output_shape" in item.message
            or "Shape dimensions cannot be zero" in item.message
        )
        for item in result.diagnostics
    )
    assert result.shape_summaries == []

###############################################################################
def test_cross_attention_requires_explicit_context_input() -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph("attention.cross"),
        ExecutionBackend.PYTORCH,
        True,
    )

    assert not result.valid
    assert any(
        item.code == "missing_required_connection"
        and item.node_id == "attention"
        and "Context" in item.message
        for item in result.diagnostics
    )

###############################################################################
def test_native_attention_rejects_nonzero_dropout_instead_of_dropping_it() -> None:
    result = ValidationService(get_node_catalog()).validate(
        attention_graph("attention.self", parameters={"dropout": 0.25}),
        ExecutionBackend.PYTORCH,
        False,
    )

    assert not result.valid
    assert any(
        item.code == "unsupported_parameter_option" and item.parameter_id == "dropout"
        for item in result.diagnostics
    )

###############################################################################
def test_valid_mlp_propagates_shapes_and_counts_parameters() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                {"shape": [20], "dtype": "float32"},
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
@pytest.mark.parametrize(
    ("definition_id", "parameters", "expected_parameter_count"),
    [
        ("recurrent.simple_rnn", {"units": 8, "return_sequences": True}, 96),
        ("recurrent.gru", {"units": 8, "return_sequences": True}, 312),
        ("recurrent.gru", {"units": 8, "return_sequences": True, "reset_after": False}, 288),
        ("recurrent.lstm", {"units": 8, "return_sequences": True}, 384),
        (
            "recurrent.bidirectional",
            {"cell": "lstm", "units": 4, "return_sequences": True, "merge_mode": "concat"},
            256,
        ),
        (
            "recurrent.stack",
            {"cell": "lstm", "units": 4, "layers": 2, "return_sequences": True},
            272,
        ),
    ],
)
def test_dedicated_recurrent_nodes_use_their_declared_cell_for_parameter_counts(
    definition_id: str,
    parameters: dict[str, object],
    expected_parameter_count: int,
) -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [4, 3], "dtype": "float32"}),
            make_node(
                "recurrent",
                definition_id,
                parameters,
            ),
            make_node("output", "output.sequence"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="recurrent",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="recurrent",
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
    assert result.total_parameters == expected_parameter_count
    recurrent = next(item for item in result.shape_summaries if item.node_id == "recurrent")
    assert recurrent.parameter_count == expected_parameter_count

###############################################################################
@pytest.mark.parametrize(
    ("merge_mode", "expected_shape"),
    [
        ("sum", [None, 4, 4]),
        ("mul", [None, 4, 4]),
        ("ave", [None, 4, 4]),
        ("concat", [None, 4, 8]),
    ],
)
def test_bidirectional_recurrent_merge_modes_infer_shapes_and_parameters(
    merge_mode: str,
    expected_shape: list[int | None],
) -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [4, 3], "dtype": "float32"}),
            make_node(
                "recurrent",
                "recurrent.bidirectional",
                {"cell": "lstm", "units": 4, "return_sequences": True, "merge_mode": merge_mode},
            ),
            make_node("output", "output.sequence"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="recurrent",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="recurrent",
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

    assert result.valid, result.diagnostics
    assert result.total_parameters == 256
    recurrent = next(item for item in result.shape_summaries if item.node_id == "recurrent")
    assert recurrent.parameter_count == 256
    assert recurrent.outputs["output"].shape == expected_shape

###############################################################################
def test_recurrent_stack_rejects_non_positive_layer_count() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [4, 3], "dtype": "float32"}),
            make_node(
                "recurrent",
                "recurrent.stack",
                {"cell": "lstm", "units": 4, "layers": 0, "return_sequences": True},
            ),
            make_node("output", "output.sequence"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="recurrent",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="recurrent",
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
        item.code == "shape_inference_failed"
        and "layers must be positive" in item.message
        for item in result.diagnostics
    )

###############################################################################
def test_cycle_is_reported_as_error() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                {"shape": [4], "dtype": "float32"},
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
        item.code == "cycle_detected" and item.severity == DiagnosticSeverity.ERROR
        for item in result.diagnostics
    )

###############################################################################
def test_unsafe_math_expression_is_rejected() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tensor",
                {"shape": [4], "dtype": "float32"},
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
def test_matmul_requires_batch_first_matrix_tensors() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("left", "input.tensor", {"shape": [4], "dtype": "float32"}),
            make_node("right", "input.tensor", {"shape": [4], "dtype": "float32"}),
            make_node("matmul", "math.matmul"),
            make_node("output", "output.custom"),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="left",
                source_port_id="output",
                target_node_id="matmul",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="right",
                source_port_id="output",
                target_node_id="matmul",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="e3",
                source_node_id="matmul",
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
        item.code == "shape_inference_failed"
        and "rank-3 or higher batch-first" in item.message
        for item in result.diagnostics
    )

###############################################################################
def test_binary_merge_preserves_shared_tensor_list_connection_order() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("left", "input.tensor", {"shape": [2, 3], "dtype": "float32"}),
            make_node("right", "input.tensor", {"shape": [3, 4], "dtype": "float32"}),
            make_node("matmul", "merge.matmul"),
            make_node("output", "output.custom"),
        ],
        edges=[
            EdgeInstance(
                id="z-left",
                source_node_id="left",
                source_port_id="output",
                target_node_id="matmul",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="a-right",
                source_node_id="right",
                source_port_id="output",
                target_node_id="matmul",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="matmul-output",
                source_node_id="matmul",
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
    summary = next(item for item in result.shape_summaries if item.node_id == "matmul")
    assert summary.outputs["output"].shape == [None, 2, 4]

###############################################################################
def test_declared_parameter_contract_rejects_null_and_unknown_values() -> None:
    def graph(parameters: dict[str, object]) -> ModelGraph:
        return ModelGraph(
            nodes=[
                make_node(
                    "input",
                    "input.tabular",
                    {"shape": [4], "dtype": "float32"},
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
def test_nullable_optional_parameter_null_is_treated_as_unset() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node(
                "dense",
                "layer.dense",
                {"units": 2, "activation": "softmax", "use_bias": True, "dtype": None},
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
    assert not any(item.parameter_id == "dtype" for item in result.diagnostics)

###############################################################################
def test_native_backend_rejects_parameter_values_without_verified_mapping() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.tabular",
                    {"shape": [4], "dtype": "float32"},
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
        item.code == "unsupported_parameter_value" and item.parameter_id == "kernel_initializer"
        for item in result.diagnostics
    )

###############################################################################
def test_merge_contracts_reject_incompatible_tensor_inputs() -> None:
    dtype_mismatch = two_input_graph(
        "merge.add",
                {"shape": [4], "dtype": "float32"},
                {"shape": [4], "dtype": "int32"},
    )
    dimension_mismatch = two_input_graph(
        "merge.concatenate",
                {"shape": [8, 16], "dtype": "float32"},
                {"shape": [9, 17], "dtype": "float32"},
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
    assert any("non-concatenated" in item.message for item in dimension_result.diagnostics)

###############################################################################
def test_residual_and_concatenate_branches_propagate_shapes() -> None:
    residual = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node("left", "layer.dense", {"units": 8}),
            make_node("right", "layer.dense", {"units": 8}),
            make_node("merge", "merge.residual"),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="input-left", source_node_id="input", source_port_id="output", target_node_id="left", target_port_id="input"),
            EdgeInstance(id="input-right", source_node_id="input", source_port_id="output", target_node_id="right", target_port_id="input"),
            EdgeInstance(id="left-merge", source_node_id="left", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="right-merge", source_node_id="right", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="merge-output", source_node_id="merge", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    concatenate = ModelGraph(
        nodes=[
            make_node("left-input", "input.tabular", {"shape": [8, 16], "dtype": "float32"}),
            make_node("right-input", "input.tabular", {"shape": [8, 9], "dtype": "float32"}),
            make_node("merge", "merge.concatenate", {"axis": -1}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="left", source_node_id="left-input", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="right", source_node_id="right-input", source_port_id="output", target_node_id="merge", target_port_id="inputs"),
            EdgeInstance(id="output", source_node_id="merge", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )

    service = ValidationService(get_node_catalog())
    residual_result = service.validate(residual, ExecutionBackend.PYTORCH, True)
    concatenate_result = service.validate(concatenate, ExecutionBackend.PYTORCH, True)

    assert residual_result.valid
    assert next(item for item in residual_result.shape_summaries if item.node_id == "merge").outputs["output"].shape == [None, 8]
    assert concatenate_result.valid
    assert next(item for item in concatenate_result.shape_summaries if item.node_id == "merge").outputs["output"].shape == [None, 8, 25]

###############################################################################
def test_reshape_rejects_impossible_target_and_identifies_cause() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node("reshape", "layer.reshape", {"target_shape": [3]}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="reshape", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="reshape", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    diagnostic = next(item for item in result.diagnostics if item.code == "shape_inference_failed")
    assert not result.valid
    assert diagnostic.node_id == "reshape"
    assert diagnostic.parameter_id == "target_shape"
    assert diagnostic.details["reason"] == "target_shape is incompatible with the input element count"
    assert "reshape" in diagnostic.message.lower()

###############################################################################
def test_repeat_vector_shape_matches_native_and_keras_semantics() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tensor", {"shape": [4], "dtype": "float32"}),
            make_node("repeat", "layer.repeat", {"repeats": 3, "axis": 1}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="repeat", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="repeat", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid
    summary = next(item for item in result.shape_summaries if item.node_id == "repeat")
    assert summary.outputs["output"].shape == [None, 3, 4]

###############################################################################
def test_shape_summaries_recompute_after_an_upstream_parameter_change() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node("dense", "layer.dense", {"units": 3}),
            make_node("output", "output.regression"),
        ],
        edges=[
            EdgeInstance(id="e1", source_node_id="input", source_port_id="output", target_node_id="dense", target_port_id="input"),
            EdgeInstance(id="e2", source_node_id="dense", source_port_id="output", target_node_id="output", target_port_id="input"),
        ],
    )
    service = ValidationService(get_node_catalog())
    initial = service.validate(graph, ExecutionBackend.PYTORCH, True)
    graph.nodes[1].parameters["units"] = 7
    updated = service.validate(graph, ExecutionBackend.PYTORCH, True)

    assert next(item for item in initial.shape_summaries if item.node_id == "dense").outputs["output"].shape == [None, 3]
    assert next(item for item in updated.shape_summaries if item.node_id == "dense").outputs["output"].shape == [None, 7]

###############################################################################
def test_convolution_rejects_stride_and_dilation_combination() -> None:
    graph = ModelGraph(
        nodes=[
            make_node(
                "input",
                "input.image",
                {"shape": [16, 16, 3], "dtype": "float32"},
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
                {"shape": [4], "dtype": "float32"},
            ),
            make_node(
                "unused",
                "input.tabular",
                {"shape": [4], "dtype": "float32"},
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
                {"shape": [8, 3], "dtype": "float32"},
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

###############################################################################
def test_terminal_operation_is_inferred_as_output_without_output_adapter() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node("prediction", "layer.dense", {"units": 1, "activation": "sigmoid"}),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="prediction",
                target_port_id="input",
            )
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid
    assert result.output_mode == "inferred"
    assert [(item.node_id, item.port_id) for item in result.effective_outputs] == [
        ("prediction", "output")
    ]
    assert result.model_type.kind == "binary_classification"

###############################################################################
def test_explicit_output_bindings_support_multiple_inputs_and_outputs() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("left_input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            make_node("right_input", "input.tabular", {"shape": [3], "dtype": "float32"}),
            make_node("left_head", "layer.dense", {"units": 1, "activation": "sigmoid"}),
            make_node("right_head", "layer.dense", {"units": 2, "activation": "softmax"}),
        ],
        edges=[
            EdgeInstance(
                id="left",
                source_node_id="left_input",
                source_port_id="output",
                target_node_id="left_head",
                target_port_id="input",
            ),
            EdgeInstance(
                id="right",
                source_node_id="right_input",
                source_port_id="output",
                target_node_id="right_head",
                target_port_id="input",
            ),
        ],
        output_bindings=[
            OutputBinding(node_id="left_head", port_id="output"),
            OutputBinding(node_id="right_head", port_id="output"),
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid
    assert result.output_mode == "explicit"
    assert {item.node_id for item in result.effective_outputs} == {"left_head", "right_head"}
    assert result.model_type.kind == "multi_output"

###############################################################################
def test_recurrent_return_state_ports_are_conditional_and_shape_propagated() -> None:
    graph = ModelGraph(
        nodes=[
            make_node("input", "input.sequence", {"shape": [5, 3], "dtype": "float32"}),
            make_node("lstm", "recurrent.lstm", {"units": 4, "return_state": True}),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="lstm",
                target_port_id="input",
            )
        ],
    )

    result = ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

    assert result.valid
    summary = next(item for item in result.shape_summaries if item.node_id == "lstm")
    assert summary.outputs["output"].shape == [None, 4]
    assert summary.outputs["hidden_state"].shape == [None, 4]
    assert summary.outputs["cell_state"].shape == [None, 4]
    assert {
        (item.node_id, item.port_id)
        for item in result.effective_outputs
    } == {("lstm", "output"), ("lstm", "hidden_state"), ("lstm", "cell_state")}

    disabled = graph.model_copy(deep=True)
    disabled.nodes[1].parameters["return_state"] = False
    disabled.output_bindings = [OutputBinding(node_id="lstm", port_id="hidden_state")]
    disabled_result = ValidationService(get_node_catalog()).validate(
        disabled,
        ExecutionBackend.PYTORCH,
        True,
    )
    assert not disabled_result.valid
    assert any(item.code == "disabled_output_port" for item in disabled_result.diagnostics)
