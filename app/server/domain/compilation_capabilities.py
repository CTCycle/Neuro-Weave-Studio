from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from domain.compilation_catalog import (
    CompilationCatalog,
    get_compilation_catalog,
    get_compilation_catalog_for_implementation,
)
from domain.enums import ExecutionBackend
from domain.manifests import ParameterDefinition


###############################################################################
class NativeCompilationCapability(BaseModel):
    backend: ExecutionBackend
    compiler_id: str
    compiler: str
    available: bool = False
    reason_code: str | None = None
    reason: str | None = None
    parameters: list[ParameterDefinition] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    catalog: CompilationCatalog = Field(default_factory=CompilationCatalog)

###############################################################################
class CompilationCapabilities(BaseModel):
    keras: CompilationCatalog
    native: dict[ExecutionBackend, NativeCompilationCapability]

###############################################################################
def get_native_compilation_capabilities(
    statuses: Mapping[ExecutionBackend, object] | None = None,
    catalog: CompilationCatalog | None = None,
) -> dict[ExecutionBackend, NativeCompilationCapability]:
    """Return the explicit native compiler controls for every backend.

    Runtime availability is merged from worker probes when statuses are
    supplied. The compiler definitions remain server-owned so the frontend
    cannot invent settings that a worker does not understand.
    """

    compilation_catalog = catalog or get_compilation_catalog()
    definitions = {
        ExecutionBackend.TENSORFLOW: NativeCompilationCapability(
            backend=ExecutionBackend.TENSORFLOW,
            compiler_id="tensorflow_function",
            compiler="TensorFlow tf.function / XLA",
            parameters=[
                ParameterDefinition(
                    id="jit_compile",
                    display_name="XLA JIT compilation",
                    type="boolean",
                    default=False,
                    description="Compile the native TensorFlow function with XLA.",
                )
            ],
        ),
        ExecutionBackend.PYTORCH: NativeCompilationCapability(
            backend=ExecutionBackend.PYTORCH,
            compiler_id="pytorch_compile",
            compiler="PyTorch torch.compile",
            parameters=[
                ParameterDefinition(
                    id="enabled",
                    display_name="Enable torch.compile",
                    type="boolean",
                    default=False,
                    description="Wrap the native module with PyTorch's compiler before execution.",
                ),
                ParameterDefinition(
                    id="mode",
                    display_name="Compile mode",
                    type="enum",
                    default="default",
                    options=["default", "reduce-overhead", "max-autotune"],
                    description="Select the optimization strategy used by torch.compile.",
                ),
                ParameterDefinition(
                    id="fullgraph",
                    display_name="Require full graph",
                    type="boolean",
                    default=False,
                    description="Fail instead of graph-breaking when the full model cannot be captured.",
                ),
                ParameterDefinition(
                    id="dynamic",
                    display_name="Dynamic shapes",
                    type="boolean",
                    default=False,
                    description="Allow torch.compile to model dynamic input shapes.",
                ),
            ],
            catalog=get_compilation_catalog_for_implementation(compilation_catalog, "pytorch"),
        ),
        ExecutionBackend.JAX: NativeCompilationCapability(
            backend=ExecutionBackend.JAX,
            compiler_id="jax_jit",
            compiler="JAX jax.jit",
            parameters=[
                ParameterDefinition(
                    id="enabled",
                    display_name="Enable jax.jit",
                    type="boolean",
                    default=False,
                    description="JIT-transform the native Flax apply function before execution.",
                )
            ],
        ),
    }
    if statuses is None:
        return definitions

    result: dict[ExecutionBackend, NativeCompilationCapability] = {}
    for backend, capability in definitions.items():
        status = statuses.get(backend)
        native_compilers = getattr(status, "native_compilers", {})
        aliases = {
            "tensorflow_function": ("tensorflow_function", "tensorflow_jit"),
            "pytorch_compile": ("pytorch_compile", "torch_compile"),
            "jax_jit": ("jax_jit",),
        }[capability.compiler_id]
        available = any(bool(native_compilers.get(identifier, False)) for identifier in aliases)
        reason_code = None if available else "compiler_unavailable"
        reason = (
            None if available else f"{capability.compiler} is not operational for this backend."
        )
        result[backend] = capability.model_copy(
            update={"available": available, "reason_code": reason_code, "reason": reason}
        )
    return result

###############################################################################
def get_compilation_capabilities(
    catalog: CompilationCatalog,
    statuses: Mapping[ExecutionBackend, object] | None = None,
) -> CompilationCapabilities:
    return CompilationCapabilities(
        keras=catalog,
        native=get_native_compilation_capabilities(statuses, catalog),
    )
