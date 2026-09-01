#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the shared native-plugin compilation helper."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorrt_plugin_utils as plugin_utils


class FakeTmpPathFactory:
    """Provide the one isolated directory operation used by the helper."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []

    def mktemp(self, basename: str) -> Path:
        self.calls.append(basename)
        output_dir = self.root / f"{basename}-{len(self.calls)}"
        output_dir.mkdir()
        return output_dir


class FakeInitializer:
    """Record ABI declarations and return configured registration results."""

    def __init__(self, results: tuple[bool, ...] = (True, True)) -> None:
        self.argtypes = None
        self.restype = None
        self.results = results
        self.call_signatures: list[tuple[object, object]] = []

    def __call__(self) -> bool:
        self.call_signatures.append((self.argtypes, self.restype))
        return self.results[len(self.call_signatures) - 1]


@dataclass
class HelperEnvironment:
    """Mocked dependency tree around one real helper invocation."""

    tmp_path_factory: FakeTmpPathFactory
    cuda_pathfinder: SimpleNamespace
    cupy: SimpleNamespace
    source_path: Path
    tensorrt_library_path: Path
    cuda_library_paths: dict[str, Path]
    header_calls: list[str]
    library_calls: list[str]


@pytest.fixture
def helper_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> HelperEnvironment:
    """Create discoverable fake sources, headers, and shared libraries."""

    repository_root = tmp_path / "repository"
    helper_path = (
        repository_root / "tests" / "plugins" / Path(plugin_utils.__file__).name
    )
    source_dir = repository_root / "src" / "fast_gpu_asr" / "tensorrt_plugins"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "plugin.cu"
    source_path.write_text("// test plugin\n", encoding="utf8")
    monkeypatch.setattr(plugin_utils, "__file__", str(helper_path))

    tensorrt_dir = tmp_path / "tensorrt_libs"
    tensorrt_dir.mkdir()
    tensorrt_library_path = tensorrt_dir / "libnvinfer.so.11"
    tensorrt_library_path.write_bytes(b"TensorRT")
    tensorrt_libs = SimpleNamespace(__file__=str(tensorrt_dir / "__init__.py"))

    include_dir = tmp_path / "cuda" / "include"
    library_dir = tmp_path / "cuda" / "lib"
    include_dir.mkdir(parents=True)
    library_dir.mkdir()
    cuda_library_paths = {
        name: library_dir / f"lib{name}.so.13" for name in ("cudart", "cublas")
    }
    for library_path in cuda_library_paths.values():
        library_path.write_bytes(b"CUDA")

    header_calls: list[str] = []
    library_calls: list[str] = []

    def find_header(name: str) -> str:
        header_calls.append(name)
        return str(include_dir)

    def load_library(name: str) -> SimpleNamespace:
        library_calls.append(name)
        return SimpleNamespace(abs_path=str(cuda_library_paths[name]))

    cuda_pathfinder = SimpleNamespace(
        find_nvidia_binary_utility=lambda name: (
            "/cuda/bin/nvcc" if name == "nvcc" else None
        ),
        find_nvidia_header_directory=find_header,
        load_nvidia_dynamic_lib=load_library,
    )
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(compute_capability="90"))
    )
    dependencies = {
        "cuda.pathfinder": cuda_pathfinder,
        "cupy": cupy,
        "tensorrt": SimpleNamespace(),
        "tensorrt_libs": tensorrt_libs,
    }
    monkeypatch.setattr(
        plugin_utils.pytest,
        "importorskip",
        lambda name: dependencies[name],
    )

    def unexpected_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("unexpected compiler or loader call")

    monkeypatch.setattr(plugin_utils.subprocess, "run", unexpected_call)
    monkeypatch.setattr(plugin_utils.ctypes, "CDLL", unexpected_call)
    return HelperEnvironment(
        FakeTmpPathFactory(tmp_path),
        cuda_pathfinder,
        cupy,
        source_path,
        tensorrt_library_path,
        cuda_library_paths,
        header_calls,
        library_calls,
    )


def compile_fake_plugin(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    initializer: FakeInitializer,
    *,
    write_output: bytes | None = b"\x7fELF",
) -> tuple[object, list[tuple[str, ...]], list[tuple[str, int]]]:
    """Run the helper with a fake compiler and loader, retaining exact calls."""

    commands: list[tuple[str, ...]] = []
    load_calls: list[tuple[str, int]] = []

    def fake_run(command: tuple[str, ...], *, check: bool) -> None:
        assert check
        commands.append(command)
        if write_output is not None:
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(write_output)

    library = SimpleNamespace(initPlugin=initializer)

    def fake_cdll(path: str, mode: int) -> object:
        load_calls.append((path, mode))
        return library

    monkeypatch.setattr(plugin_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(plugin_utils.ctypes, "CDLL", fake_cdll)
    result = plugin_utils.compile_and_load_plugin(
        helper_environment.tmp_path_factory,
        "plugin.cu",
        "initPlugin",
        ("cudart", "cublas"),
    )
    return result, commands, load_calls


def test_compile_and_load_plugin_uses_production_flags_and_exact_dependencies(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile once for the active GPU, load globally, and register twice."""

    initializer = FakeInitializer()
    library, commands, load_calls = compile_fake_plugin(
        helper_environment,
        monkeypatch,
        initializer,
    )

    assert len(commands) == 1
    output_path = (
        helper_environment.tmp_path_factory.root / "tensorrt_plugins-1" / "plugin.so"
    )
    cuda_library_dir = next(iter(helper_environment.cuda_library_paths.values())).parent
    assert commands[0] == (
        "/cuda/bin/nvcc",
        "--std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "--generate-code=arch=compute_90,code=sm_90",
        "-shared",
        "-I",
        str(cuda_library_dir.parent / "include"),
        str(helper_environment.source_path),
        "-o",
        str(output_path),
        "-L",
        str(helper_environment.tensorrt_library_path.parent),
        "-L",
        str(cuda_library_dir),
        "-l:libnvinfer.so.11",
        "-l:libcudart.so.13",
        "-l:libcublas.so.13",
    )
    assert load_calls == [(str(output_path), ctypes.RTLD_GLOBAL)]
    assert initializer.call_signatures == [
        ((), ctypes.c_bool),
        ((), ctypes.c_bool),
    ]
    assert library.initPlugin is initializer
    assert helper_environment.tmp_path_factory.calls == ["tensorrt_plugins"]
    assert helper_environment.header_calls == ["cudart", "cublas"]
    assert helper_environment.library_calls == ["cudart", "cublas"]


@pytest.mark.parametrize(
    ("results", "message"),
    (
        pytest.param((False,), "Plugin registration failed", id="first-call"),
        pytest.param(
            (True, False),
            "Plugin registration is not idempotent",
            id="second-call",
        ),
    ),
)
def test_compile_and_load_plugin_rejects_registration_failure(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    results: tuple[bool, ...],
    message: str,
) -> None:
    """Reject both initial and repeated registration failures."""

    initializer = FakeInitializer(results)
    with pytest.raises(RuntimeError, match=message):
        compile_fake_plugin(helper_environment, monkeypatch, initializer)

    assert initializer.call_signatures == [((), ctypes.c_bool)] * len(results)


@pytest.mark.parametrize(
    ("write_output", "error_type", "message"),
    (
        pytest.param(
            None, FileNotFoundError, "Compiler output not found", id="missing"
        ),
        pytest.param(b"", RuntimeError, "Compiler output is empty", id="empty"),
    ),
)
def test_compile_and_load_plugin_rejects_missing_or_empty_output(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    write_output: bytes | None,
    error_type: type[Exception],
    message: str,
) -> None:
    """Do not load a compiler invocation that produced no usable artifact."""

    with pytest.raises(error_type, match=message):
        compile_fake_plugin(
            helper_environment,
            monkeypatch,
            FakeInitializer(),
            write_output=write_output,
        )


def test_compile_and_load_plugin_rejects_missing_header(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail instead of silently falling back to unrelated system headers."""

    monkeypatch.setattr(
        helper_environment.cuda_pathfinder,
        "find_nvidia_header_directory",
        lambda name: None if name == "cublas" else "/cuda/include",
    )

    with pytest.raises(RuntimeError, match="CUDA headers for cublas were not found"):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            "plugin.cu",
            "initPlugin",
            ("cudart", "cublas"),
        )

    assert helper_environment.library_calls == ["cudart"]


def test_compile_and_load_plugin_rejects_missing_tensorrt_library(
    helper_environment: HelperEnvironment,
) -> None:
    """Require the exact TensorRT wheel library before invoking nvcc."""

    helper_environment.tensorrt_library_path.unlink()

    with pytest.raises(FileNotFoundError, match="TensorRT library not found"):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            "plugin.cu",
            "initPlugin",
            ("cudart",),
        )

    assert helper_environment.header_calls == []
    assert helper_environment.library_calls == []
    assert helper_environment.tmp_path_factory.calls == []


@pytest.mark.parametrize(
    ("source_name", "error_type", "message"),
    (
        pytest.param(
            "../plugin.cu",
            ValueError,
            "Plugin source must be a filename",
            id="outside-source-directory",
        ),
        pytest.param(
            "missing.cu",
            FileNotFoundError,
            "Plugin source not found",
            id="missing",
        ),
    ),
)
def test_compile_and_load_plugin_rejects_invalid_source(
    helper_environment: HelperEnvironment,
    source_name: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Validate repository inputs before probing the optional CUDA toolchain."""

    with pytest.raises(error_type, match=message):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            source_name,
            "initPlugin",
            ("cudart",),
        )

    assert helper_environment.tmp_path_factory.calls == []
    assert helper_environment.header_calls == []
    assert helper_environment.library_calls == []


def test_compile_and_load_plugin_rejects_stale_cuda_library_path(
    helper_environment: HelperEnvironment,
) -> None:
    """Do not let the linker substitute a system library for a stale result."""

    helper_environment.cuda_library_paths["cublas"].unlink()

    with pytest.raises(FileNotFoundError, match="CUDA library not found"):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            "plugin.cu",
            "initPlugin",
            ("cudart", "cublas"),
        )

    assert helper_environment.header_calls == ["cudart", "cublas"]
    assert helper_environment.library_calls == ["cudart", "cublas"]
    assert helper_environment.tmp_path_factory.calls == []


def test_compile_and_load_plugin_rejects_malformed_compute_capability(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a malformed architecture before constructing an nvcc command."""

    monkeypatch.setattr(
        helper_environment.cupy.cuda,
        "Device",
        lambda: SimpleNamespace(compute_capability="9.0"),
    )

    with pytest.raises(RuntimeError, match="Invalid CUDA compute capability"):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            "plugin.cu",
            "initPlugin",
            ("cudart",),
        )

    assert helper_environment.tmp_path_factory.calls == []


def test_compile_and_load_plugin_skips_when_nvcc_is_unavailable(
    helper_environment: HelperEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an unavailable optional compiler as a test skip."""

    monkeypatch.setattr(
        helper_environment.cuda_pathfinder,
        "find_nvidia_binary_utility",
        lambda _name: None,
    )

    with pytest.raises(pytest.skip.Exception, match="nvcc is required"):
        plugin_utils.compile_and_load_plugin(
            helper_environment.tmp_path_factory,
            "plugin.cu",
            "initPlugin",
            ("cudart",),
        )

    assert helper_environment.tmp_path_factory.calls == []
