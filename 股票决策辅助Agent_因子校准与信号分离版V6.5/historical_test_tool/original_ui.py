from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import sys
import types


def _keep_original_ui_node(node: ast.stmt) -> bool:
    """Keep original UI definitions but never execute old test-tool imports."""

    if isinstance(node, ast.ImportFrom):
        return not str(node.module or "").startswith("historical_test_tool")
    if isinstance(node, ast.Import):
        return not any(str(alias.name).startswith("historical_test_tool") for alias in node.names)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return True
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == "DISPLAY_DECIMALS" for target in node.targets)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id == "DISPLAY_DECIMALS"
    return False


@lru_cache(maxsize=1)
def load_original_ui() -> types.ModuleType:
    """Load V6.5's renderer functions without executing its login/main program."""

    project_root = Path(__file__).resolve().parent.parent
    if not (project_root / "app.py").exists() and (project_root / "model_v64" / "app.py").exists():
        project_root = project_root / "model_v64"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    source_path = project_root / "app.py"
    source = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(source_path))

    kept = [node for node in parsed.body if _keep_original_ui_node(node)]

    module_tree = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(module_tree)
    module = types.ModuleType("historical_original_v64_ui")
    module.__file__ = str(source_path)
    exec(compile(module_tree, str(source_path), "exec"), module.__dict__)
    return module
