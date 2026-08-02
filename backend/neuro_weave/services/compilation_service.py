from __future__ import annotations

from neuro_weave.catalog.loader import NodeCatalog
from neuro_weave.compilation.compiler import compile_to_ir
from neuro_weave.domain.compilation import CompilationResult
from neuro_weave.domain.enums import TargetFramework
from neuro_weave.domain.graph import ModelGraph
from neuro_weave.exporters.keras_exporter import KerasSourceExporter
from neuro_weave.exporters.pytorch_exporter import PyTorchSourceExporter
from neuro_weave.validation.validator import GraphValidator


class CompilationService:
    def __init__(self, catalog: NodeCatalog, validator: GraphValidator) -> None:
        self._catalog = catalog
        self._validator = validator
        self._exporters = {
            TargetFramework.KERAS: KerasSourceExporter(),
            TargetFramework.TENSORFLOW_KERAS: KerasSourceExporter(),
            TargetFramework.PYTORCH: PyTorchSourceExporter(),
        }

    def compile(self, graph: ModelGraph, framework: TargetFramework) -> CompilationResult:
        validation = self._validator.validate(graph, framework)
        if not validation.valid:
            return CompilationResult(
                valid=False,
                framework=framework,
                diagnostics=validation.diagnostics,
            )
        ir = compile_to_ir(graph, framework, self._catalog)
        code = self._exporters[framework].generate(graph, ir)
        return CompilationResult(
            valid=True,
            framework=framework,
            diagnostics=validation.diagnostics,
            code=code,
            ir=ir,
        )
