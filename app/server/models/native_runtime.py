# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from domain.compilation import CanonicalModelIR
from domain.compilation_catalog import CompilationCatalog
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from exporters.native_jax_exporter import NativeJaxSourceExporter
from exporters.native_tensorflow_exporter import NativeTensorFlowSourceExporter
from exporters.pytorch_exporter import PyTorchSourceExporter
from validation.output_resolution import input_nodes


###############################################################################
@dataclass
class NativeModelHandle:
    backend: ExecutionBackend
    model: Any
    execution_model: Any
    variables: Any
    model_input: Any
    inputs: list[Any]
    outputs: list[Any]
    ir: CanonicalModelIR
    source: str
    apply: Callable[[Any, Any], Any] | None = None
    compilation: dict[str, Any] | None = None

    # -------------------------------------------------------------------------
    def parameter_count(self) -> int:
        if self.backend is ExecutionBackend.PYTORCH:
            return int(sum(int(parameter.numel()) for parameter in self.model.parameters()))
        if self.backend is ExecutionBackend.TENSORFLOW:
            return int(sum(_shape_product(variable.shape) for variable in self.model.variables))
        return int(sum(_array_size(value) for value in _flatten_jax_variables(self.variables)))


###############################################################################
def build_native_model(
    graph: ModelGraph,
    backend: ExecutionBackend,
    *,
    checkpoint: Callable[[], None] | None = None,
    compilation_catalog: CompilationCatalog | None = None,
) -> NativeModelHandle:
    _checkpoint(checkpoint)
    catalog = get_node_catalog()
    ir = compile_to_ir(graph, backend, False, catalog)
    exporter = {
        ExecutionBackend.TENSORFLOW: NativeTensorFlowSourceExporter(),
        ExecutionBackend.PYTORCH: PyTorchSourceExporter(),
        ExecutionBackend.JAX: NativeJaxSourceExporter(),
    }[backend]
    source = exporter.generate(graph, ir, catalog, compilation_catalog=compilation_catalog)
    module = types.ModuleType(f"neuro_weave_native_{backend.value}")
    exec(compile(source, f"<neuro-weave-{backend.value}-source>", "exec"), module.__dict__)
    factory = (
        module.__dict__.get("build_raw_model")
        or module.__dict__.get("build_model")
        or module.__dict__.get("GeneratedModel")
    )
    if factory is None:
        raise RuntimeError(f"Native {backend.value} source did not define a model factory")
    model = cast(Any, factory)()
    compile_factory = module.__dict__.get("compile_model")
    execution_model = (
        cast(Callable[[Any], Any], compile_factory)(model)
        if backend is ExecutionBackend.PYTORCH and callable(compile_factory)
        else model
    )
    inputs = _dummy_inputs(graph, backend, ir.input_node_ids)
    graph_inputs = input_nodes(graph, catalog)
    model_input: Any = (
        inputs[0]
        if len(inputs) == 1
        else {
            node.id: value
            for node, value in zip(
                graph_inputs,
                inputs,
                strict=True,
            )
        }
    )
    variables: Any = None
    apply: Callable[[Any, Any], Any] | None = None
    compilation: dict[str, Any] | None = None
    if backend is ExecutionBackend.PYTORCH:
        execution_model.eval()
        outputs = _as_list(execution_model(model_input))
        _zero_pytorch_parameters(model)
        outputs = _as_list(execution_model(model_input))
        compilation_factory = module.__dict__.get("build_compilation")
        if callable(compilation_factory):
            compilation = cast(Callable[[Any], dict[str, Any]], compilation_factory)(model)
    elif backend is ExecutionBackend.TENSORFLOW:
        outputs = _as_list(execution_model(model_input))
    else:
        import jax

        variables = model.init(jax.random.PRNGKey(0), model_input)
        variables = _zero_jax_variables(variables)
        apply_factory = module.__dict__.get("build_apply")
        apply_callable = (
            cast(Callable[[Any], Callable[[Any, Any], Any]], apply_factory)(model)
            if callable(apply_factory)
            else cast(Callable[[Any, Any], Any], model.apply)
        )
        apply = apply_callable
        outputs = _as_list(apply_callable(variables, model_input))
    _checkpoint(checkpoint)
    return NativeModelHandle(
        backend=backend,
        model=model,
        execution_model=execution_model,
        variables=variables,
        model_input=model_input,
        inputs=inputs,
        outputs=outputs,
        ir=ir,
        source=source,
        apply=apply,
        compilation=compilation,
    )


###############################################################################
def forward_native(handle: NativeModelHandle) -> list[Any]:
    if handle.backend is ExecutionBackend.PYTORCH:
        return _as_list(handle.execution_model(handle.model_input))
    if handle.backend is ExecutionBackend.TENSORFLOW:
        return _as_list(handle.execution_model(handle.model_input))
    apply = handle.apply or handle.model.apply
    return _as_list(apply(handle.variables, handle.model_input))


###############################################################################
def _dummy_inputs(
    graph: ModelGraph,
    backend: ExecutionBackend,
    input_node_ids: list[str],
) -> list[Any]:
    graph_nodes = {node.id: node for node in graph.nodes}
    nodes = [graph_nodes[node_id] for node_id in input_node_ids]
    result: list[Any] = []
    for node in nodes:
        shape_value = node.parameters.get("shape", [None, 1])
        shape = [
            1 if value is None or not isinstance(value, int) or value <= 0 else value
            for value in shape_value
        ]
        dtype = str(node.parameters.get("dtype", "float32"))
        if backend is ExecutionBackend.PYTORCH:
            import torch

            result.append(torch.zeros(shape, dtype=_torch_dtype(torch, dtype)))
        elif backend is ExecutionBackend.TENSORFLOW:
            import tensorflow as tf

            result.append(tf.zeros(shape, dtype=_tensorflow_dtype(tf, dtype)))
        else:
            import jax.numpy as jnp

            result.append(jnp.zeros(shape, dtype=dtype))
    return result


###############################################################################
def _torch_dtype(torch_module: Any, value: str) -> Any:
    return {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "float64": torch_module.float64,
        "int32": torch_module.int32,
        "int64": torch_module.int64,
        "bool": torch_module.bool,
    }.get(value, torch_module.float32)


###############################################################################
def _tensorflow_dtype(tf_module: Any, value: str) -> Any:
    return getattr(tf_module, value, tf_module.float32)


###############################################################################
def _zero_pytorch_parameters(model: Any) -> None:
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for name, buffer in model.named_buffers():
            if "running_var" in name:
                buffer.fill_(1.0)
            elif buffer.dtype.is_floating_point:
                buffer.zero_()


###############################################################################
def _zero_jax_variables(variables: Any) -> Any:
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict

    flattened = flatten_dict(variables)
    for path, value in flattened.items():
        # Flax BatchNorm stores moving variance in the batch_stats collection.
        # Keep those leaves at one so deterministic zero-input verification does
        # not manufacture a divide-by-zero NaN.
        is_variance = any(
            str(part).lower() in {"var", "variance", "running_var", "moving_variance"}
            for part in path
        )
        flattened[path] = jnp.ones_like(value) if is_variance else jnp.zeros_like(value)
    return unflatten_dict(flattened)


###############################################################################
def _flatten_jax_variables(variables: Any) -> list[Any]:
    from jax.tree_util import tree_leaves

    return list(tree_leaves(variables))


###############################################################################
def _shape_product(shape: Any) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


###############################################################################
def _array_size(value: Any) -> int:
    size = getattr(value, "size", None)
    if size is not None:
        return int(size)
    return _shape_product(getattr(value, "shape", []))


###############################################################################
def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


###############################################################################
def _checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()
