#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Enforce dependency-light imports for export-model definitions."""

import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
EXPORT_MODEL_DIR = SRC_DIR / "fast_gpu_asr" / "export" / "model"
EXPORT_MODEL_PACKAGE = "fast_gpu_asr.export.model"
ALLOWED_MODEL_IMPORT_ROOTS = sys.stdlib_module_names | {"torch"}
BLOCKED_MODEL_IMPORT_ROOTS = (
    "icefall",
    "k2",
    "kaldi_native_fbank",
    "lhotse",
    "nemo",
    "onnx",
    "onnxruntime",
    "onnxscript",
)


def run_isolated(script: str) -> None:
    """Run a fresh interpreter against the working-tree package.

    Parameters
    ----------
    script : str
        Python source executed after prepending the repository's source directory
        to the isolated interpreter's import path.

    Raises
    ------
    AssertionError
        Raised on a nonzero exit, including captured output for diagnosis.
    subprocess.TimeoutExpired
        Raised when the interpreter does not finish within 30 seconds.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys\nsys.path.insert(0, {str(SRC_DIR)!r})\n{script}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Isolated import failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def export_model_modules() -> tuple[tuple[str, Path], ...]:
    """Discover every Python module in the condensed export-model package.

    Returns
    -------
    tuple[tuple[str, Path], ...]
        Qualified import names and absolute source paths, ordered by path.
        Package initializers use the package name without ``.__init__``.
    """

    modules = []
    for module_path in sorted(EXPORT_MODEL_DIR.rglob("*.py")):
        module_parts = module_path.relative_to(SRC_DIR).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        modules.append((".".join(module_parts), module_path))
    return tuple(modules)


def unexpected_import_references(
    module_name: str, module_path: Path
) -> tuple[tuple[int, str], ...]:
    """Return imports outside the dependency-light model package boundary.

    Parameters
    ----------
    module_name : str
        Qualified import name used to resolve relative references.
    module_path : Path
        Source file to inspect, including deferred and conditional imports.

    Returns
    -------
    tuple[tuple[int, str], ...]
        Sorted source line numbers and disallowed module names. Calls to
        ``__import__`` or ``importlib.import_module`` use ``<dynamic import>``.

    Notes
    -----
    This source-level guard recognizes direct imports and ordinary aliases for
    dynamic import helpers; it is not a complete Python name resolver. The
    isolated-interpreter test separately checks actual import-time dependencies.
    """

    tree = ast.parse(module_path.read_text(encoding="utf8"), filename=str(module_path))
    package_name = (
        module_name
        if module_path.name == "__init__.py"
        else module_name.rpartition(".")[0]
    )
    nodes = tuple(ast.walk(tree))
    importlib_names = {
        alias.asname or alias.name
        for node in nodes
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "importlib"
    }
    import_module_names = {"__import__"}
    import_module_names.update(
        alias.asname or alias.name
        for node in nodes
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    )

    unexpected_references = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imports = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            import_name = node.module or ""
            if node.level:
                import_name = resolve_name("." * node.level + import_name, package_name)
            imports = (import_name,)
        elif isinstance(node, ast.Call):
            named_dynamic_import = (
                isinstance(node.func, ast.Name) and node.func.id in import_module_names
            )
            module_dynamic_import = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_names
                and node.func.attr == "import_module"
            )
            if named_dynamic_import or module_dynamic_import:
                unexpected_references.append((node.lineno, "<dynamic import>"))
            continue
        else:
            continue

        unexpected_references.extend(
            (node.lineno, imported_name)
            for imported_name in imports
            if imported_name.split(".")[0] not in ALLOWED_MODEL_IMPORT_ROOTS
            and imported_name != "fast_gpu_asr.constants"
            and imported_name != EXPORT_MODEL_PACKAGE
            and not imported_name.startswith(f"{EXPORT_MODEL_PACKAGE}.")
        )
    return tuple(sorted(unexpected_references))


def test_export_model_sources_stay_within_dependency_boundary() -> None:
    modules = export_model_modules()
    assert modules
    unexpected_by_module = {
        module_name: references
        for module_name, path in modules
        if (references := unexpected_import_references(module_name, path))
    }
    assert not unexpected_by_module, unexpected_by_module


@pytest.mark.parametrize("filename", ("probe.py", "__init__.py"))
def test_import_guard_handles_relative_and_deferred_imports(
    tmp_path: Path, filename: str
) -> None:
    module_path = tmp_path / filename
    module_path.write_text(
        """import math
import torch
from ....constants import INT32_MAX
from . import decoder
import importlib as imports
from importlib import import_module as load_module
if False:
    import nemo.collections
def deferred():
    from ....utils import get_engine
    imports.import_module("sentencepiece")
    load_module("tensorrt")
    __import__("cupy")
    unrelated.import_module()
""",
        encoding="utf8",
    )

    module_name = "fast_gpu_asr.export.model.parakeet"
    if filename != "__init__.py":
        module_name += ".probe"
    assert unexpected_import_references(module_name, module_path) == (
        (8, "nemo.collections"),
        (10, "fast_gpu_asr.utils"),
        (11, "<dynamic import>"),
        (12, "<dynamic import>"),
        (13, "<dynamic import>"),
    )


def test_export_model_modules_do_not_import_external_stacks() -> None:
    modules = export_model_modules()
    module_files = {name: str(path) for name, path in modules}
    assert len(module_files) == len(modules) > 0

    script = f"""
import importlib
import importlib.abc
import sys

blocked_roots = {BLOCKED_MODEL_IMPORT_ROOTS!r}
blocked_imports = []

class OptionalImportBlocker(importlib.abc.MetaPathFinder):
    '''Reject external training and ONNX dependencies during import.'''

    def find_spec(self, fullname, path=None, target=None):
        '''Record blocked imports even when callers catch ImportError.'''

        if fullname.split('.')[0] in blocked_roots:
            blocked_imports.append(fullname)
            raise ImportError(f'blocked optional dependency: {{fullname}}')
        return None

sys.meta_path.insert(0, OptionalImportBlocker())

for module_name, expected_file in {module_files!r}.items():
    module = importlib.import_module(module_name)
    assert module.__file__ == expected_file, (module_name, module.__file__)
assert blocked_imports == [], blocked_imports
loaded_optional = sorted(set(sys.modules) & set(blocked_roots))
assert not loaded_optional, loaded_optional

blocker_probe = blocked_roots[0]
try:
    importlib.import_module(blocker_probe)
except ImportError as error:
    assert str(error) == f'blocked optional dependency: {{blocker_probe}}', str(error)
else:
    raise AssertionError('optional import blocker did not reject its probe')
assert blocked_imports == [blocker_probe], blocked_imports
"""
    run_isolated(script)
