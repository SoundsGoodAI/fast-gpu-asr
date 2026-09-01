#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for export-model import boundaries."""

import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
EXPORT_MODEL_DIR = SRC_DIR / "fast_gpu_asr" / "export" / "model"
BLOCKED_EXPORT_ROOTS = (
    "icefall",
    "k2",
    "kaldi_native_fbank",
    "lhotse",
    "nemo",
    "onnx",
    "onnxruntime",
    "onnxscript",
)


def run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    """Run a fresh interpreter against the working-tree package."""

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys\nsys.path.insert(0, {str(SRC_DIR)!r})\n{script}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Isolated import failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def export_model_modules() -> tuple[tuple[str, Path], ...]:
    """Discover every Python module in the condensed export-model package."""

    modules = []
    for module_path in sorted(EXPORT_MODEL_DIR.rglob("*.py")):
        module_parts = module_path.relative_to(SRC_DIR).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        modules.append((".".join(module_parts), module_path.resolve()))
    return tuple(modules)


def blocker_source(roots: tuple[str, ...]) -> str:
    """Return source that rejects imports from the supplied package roots."""

    return f"""
import importlib.abc
import sys

blocked_roots = {roots!r}
blocked_imports = []

class OptionalImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked_roots:
            blocked_imports.append(fullname)
            raise ImportError(f'blocked optional dependency: {{fullname}}')
        return None

sys.meta_path.insert(0, OptionalImportBlocker())
"""


def test_export_model_modules_do_not_import_optional_stacks() -> None:
    """Keep condensed PyTorch definitions independent of upstream frameworks."""

    modules = export_model_modules()
    module_names = tuple(module_name for module_name, _ in modules)
    assert module_names
    assert len(module_names) == len(set(module_names))

    script = (
        blocker_source(BLOCKED_EXPORT_ROOTS)
        + f"""
import importlib
from pathlib import Path

module_names = {module_names!r}
module_files = {{}}
for module_name in module_names:
    module = importlib.import_module(module_name)
    module_files[module_name] = str(Path(module.__file__).resolve())
loaded_optional = [
    name for name in sys.modules if name.split('.')[0] in {BLOCKED_EXPORT_ROOTS!r}
]
print(json.dumps({{
    'module_files': module_files,
    'blocked_attempts': blocked_imports,
    'loaded_optional': sorted(loaded_optional),
}}))
"""
    )
    result = run_isolated("import json\n" + script)
    payload = json.loads(result.stdout)

    assert payload["module_files"] == {
        module_name: str(module_path) for module_name, module_path in modules
    }
    assert payload["blocked_attempts"] == []
    assert payload["loaded_optional"] == []
