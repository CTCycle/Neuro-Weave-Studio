from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOTS = (
    Path(__file__).resolve().parents[1] / "server",
    Path(__file__).resolve().parents[1] / "scripts",
    Path(__file__).resolve(),
)

###############################################################################
def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(
                path
                for path in root.rglob("*.py")
                if not set(path.parts).intersection(
                    {".venv", "site-packages", "__pycache__", ".pytest_cache", ".ruff_cache"}
                )
            )
    return sorted(set(files))

###############################################################################
def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents

###############################################################################
def _nested_definitions(
    tree: ast.AST,
    path: Path,
    parents: dict[ast.AST, ast.AST],
    *,
    allow_worker_imports: bool = False,
) -> list[str]:
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = parents.get(node)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                violations.append(f"{path}:{node.lineno}: nested definition")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parent = parents.get(node)
            if not isinstance(parent, ast.Module) and not allow_worker_imports:
                violations.append(f"{path}:{node.lineno}: non-top-level import")
    return violations

###############################################################################
def test_backend_python_structure_contract() -> None:
    violations: list[str] = []
    for path in _python_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 1000:
            violations.append(f"{path}: file exceeds 1000 lines")
        tree = ast.parse("\n".join(lines), filename=str(path))
        violations.extend(
            _nested_definitions(
                tree,
                path,
                _parent_map(tree),
                allow_worker_imports=path.name in {"ml_worker.py", "keras_worker.py"},
            )
        )
    assert not violations, "\n".join(violations)
