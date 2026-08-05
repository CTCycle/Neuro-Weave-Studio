from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from compilation.compiler import compile_to_ir
from domain.compilation import CompilationResult
from domain.enums import TargetFramework
from domain.graph import ModelGraph
from exporters.keras_exporter import KerasSourceExporter
from exporters.pytorch_exporter import PyTorchSourceExporter
from services.validation_service import ValidationService


class CompilationService:
    def __init__(self, catalog: NodeCatalog, validator: ValidationService) -> None:
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
