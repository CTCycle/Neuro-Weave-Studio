from __future__ import annotations

from common.catalogs.loader import get_node_catalog
from domain.compilation_capabilities import get_compilation_capabilities
from domain.compilation_catalog import get_compilation_catalog
from domain.enums import ExecutionBackend
from domain.graph import (
    EdgeInstance,
    JaxNativeConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    PyTorchNativeConfiguration,
    TensorFlowNativeConfiguration,
)
from domain.metadata import ExecutionBackendStatus
from services.compilation_service import CompilationService
from services.validation_service import ValidationService
from validation.compile_configuration import validate_model_configuration

###############################################################################
def _native_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tabular",
                position=Position(x=0, y=0),
                parameters={"shape": [4], "dtype": "float32"},
            ),
            NodeInstance(
                id="dense",
                definition_id="layer.dense",
                position=Position(x=1, y=0),
                parameters={"units": 2, "activation": "softmax"},
            ),
            NodeInstance(
                id="output",
                definition_id="output.classification",
                position=Position(x=2, y=0),
            ),
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
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                tensorflow=TensorFlowNativeConfiguration(jit_compile=True),
                pytorch=PyTorchNativeConfiguration(
                    compile_enabled=True,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=True,
                ),
                jax=JaxNativeConfiguration(jit_enabled=True),
            )
        ),
    )

###############################################################################
def test_metadata_exposes_one_capability_tree_with_explicit_runtime_reasons() -> None:
    statuses = {
        ExecutionBackend.TENSORFLOW: ExecutionBackendStatus(
            id=ExecutionBackend.TENSORFLOW,
            available=True,
            native_compilers={"tensorflow_jit": True},
        ),
        ExecutionBackend.PYTORCH: ExecutionBackendStatus(
            id=ExecutionBackend.PYTORCH,
            available=True,
            native_compilers={"torch_compile": False},
        ),
        ExecutionBackend.JAX: ExecutionBackendStatus(
            id=ExecutionBackend.JAX,
            available=False,
            native_compilers={},
        ),
    }

    capabilities = get_compilation_capabilities(get_compilation_catalog(), statuses)

    assert set(capabilities.native) == set(ExecutionBackend)
    assert capabilities.native[ExecutionBackend.TENSORFLOW].available is True
    assert capabilities.native[ExecutionBackend.PYTORCH].reason_code == "compiler_unavailable"
    assert capabilities.native[ExecutionBackend.JAX].reason_code == "compiler_unavailable"
    assert capabilities.native[ExecutionBackend.PYTORCH].parameters[-1].type == "boolean"
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.optimizers
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.losses
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.metrics
    assert all(
        "pytorch" in item.implementations
        for item in capabilities.native[ExecutionBackend.PYTORCH].catalog.losses
    )

###############################################################################
def test_native_configuration_round_trips_and_invalid_options_are_diagnostic() -> None:
    graph = _native_graph()
    round_tripped = ModelGraph.model_validate(graph.model_dump(mode="json"))

    assert round_tripped.model_configuration == graph.model_configuration
    assert round_tripped.model_configuration.native.pytorch.mode == "max-autotune"
    invalid = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=PyTorchNativeConfiguration(mode="not-a-torch-mode")
                )
            )
        }
    )

    diagnostics = validate_model_configuration(invalid, ExecutionBackend.PYTORCH, False)

    assert [item.code for item in diagnostics] == ["invalid_native_compilation"]

###############################################################################
def test_native_compilation_source_uses_each_backend_native_compiler() -> None:
    graph = _native_graph()
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    tensorflow = service.compile(graph, ExecutionBackend.TENSORFLOW, False)
    pytorch = service.compile(graph, ExecutionBackend.PYTORCH, False)
    jax = service.compile(graph, ExecutionBackend.JAX, False)

    assert tensorflow.valid is True
    assert pytorch.valid is True
    assert jax.valid is True
    assert tensorflow.code is not None
    assert pytorch.code is not None
    assert jax.code is not None
    assert "@tf.function(jit_compile=True)" in tensorflow.code
    assert "torch.compile(" in pytorch.code
    assert "mode='max-autotune'" in pytorch.code
    assert "def build_optimizer" in pytorch.code
    assert "def configured_losses" in pytorch.code
    assert "def configured_metrics" in pytorch.code
    assert "return jax.jit(model.apply)" in jax.code
