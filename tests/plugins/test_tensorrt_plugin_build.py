#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for native TensorRT plugin build orchestration."""

import subprocess
from concurrent.futures import ThreadPoolExecutor as NativeThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

import fast_gpu_asr.tensorrt_plugins.build as build_module
from fast_gpu_asr.tensorrt_plugins.build import build_plugin

TEST_PLUGIN_BUILDS = (("first.cu", ("cudart",)), ("second.cu", ("cublas",)))


@dataclass(frozen=True)
class PluginBuildCall:
    """Arguments passed to one mocked plugin compilation."""

    nvcc_path: Path
    source_dir: Path
    output_dir: Path
    tensorrt_library_path: Path
    cuda_include_dirs: tuple[Path, ...]
    source_name: str
    cuda_library_paths: tuple[Path, ...]


def write_fake_plugin(path: Path, payload: bytes = b"compiled") -> None:
    """Write a nonempty stand-in for one compiler-produced shared library."""

    path.write_bytes(b"\x7fELF" + payload)


def write_installed_plugins(source_dir: Path) -> dict[str, bytes]:
    """Create an existing plugin set and return its exact contents."""

    contents = {
        Path(source_name).with_suffix(".so").name: f"installed {source_name}".encode()
        for source_name, _ in TEST_PLUGIN_BUILDS
    }
    for name, payload in contents.items():
        (source_dir / name).write_bytes(payload)
    return contents


def record_build_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record unexpected plugin compilations during dependency discovery tests."""

    attempts: list[str] = []

    def record_build(
        nvcc_path: Path,
        source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            source_dir,
            output_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        attempts.append(source_name)

    monkeypatch.setattr(build_module, "build_plugin", record_build)
    return attempts


def test_build_plugin_uses_exact_headers_and_libraries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke nvcc once with exact include paths and direct-library linkage."""

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    commands: list[tuple[tuple[str, ...], Path, bool]] = []

    def fake_run(command: tuple[str, ...], *, cwd: Path, check: bool) -> None:
        commands.append((command, cwd, check))

    monkeypatch.setattr(build_module.subprocess, "run", fake_run)
    build_plugin(
        Path("/cuda/bin/nvcc"),
        source_dir,
        output_dir,
        Path("/tensorrt/lib/libnvinfer.so.11"),
        (Path("/cuda/include"), Path("/cublas/include")),
        "plugin.cu",
        (Path("/cuda/lib/libcudart.so.13"), Path("/cuda/lib/libcublas.so.13")),
    )

    assert len(commands) == 1
    command, cwd, check = commands[0]
    assert cwd == source_dir
    assert check
    assert command == (
        "/cuda/bin/nvcc",
        *build_module.NVCC_OPTIONS,
        "-I",
        "/cuda/include",
        "-I",
        "/cublas/include",
        "plugin.cu",
        "-o",
        str(output_dir / "plugin.so"),
        "-L",
        "/tensorrt/lib",
        "-L",
        "/cuda/lib",
        "-l:libnvinfer.so.11",
        "-l:libcudart.so.13",
        "-l:libcublas.so.13",
    )


def prepare_build_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point build discovery at a temporary source and dependency tree."""

    source_dir = tmp_path / "plugins"
    source_dir.mkdir()
    for source_name, _ in TEST_PLUGIN_BUILDS:
        (source_dir / source_name).write_text("// test source\n", encoding="utf8")
    monkeypatch.setattr(build_module, "__file__", str(source_dir / "build.py"))

    tensorrt_dir = tmp_path / "tensorrt_libs"
    tensorrt_dir.mkdir()
    (tensorrt_dir / "libnvinfer.so.11").touch()
    monkeypatch.setattr(
        build_module.tensorrt_libs,
        "__file__",
        str(tensorrt_dir / "__init__.py"),
    )
    monkeypatch.setattr(
        build_module,
        "find_nvidia_binary_utility",
        lambda name: "/cuda/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(
        build_module,
        "find_nvidia_header_directory",
        lambda name: f"/cuda/{name}/include",
    )
    monkeypatch.setattr(
        build_module,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(abs_path=f"/cuda/{name}/lib{name}.so.13"),
    )
    monkeypatch.setattr(
        build_module,
        "PLUGIN_BUILDS",
        TEST_PLUGIN_BUILDS,
    )
    monkeypatch.setattr(build_module, "CUDA_BUILD_LIBRARIES", ("cudart", "cublas"))
    return source_dir


def test_build_main_installs_only_complete_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install every output only after all plugin compilations succeed."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    build_calls: list[PluginBuildCall] = []
    calls_lock = Lock()

    def fake_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        with calls_lock:
            build_calls.append(
                PluginBuildCall(
                    nvcc_path,
                    plugin_source_dir,
                    output_dir,
                    tensorrt_library_path,
                    cuda_include_dirs,
                    source_name,
                    cuda_library_paths,
                )
            )
        write_fake_plugin(
            output_dir / Path(source_name).with_suffix(".so").name,
            source_name.encode(),
        )

    monkeypatch.setattr(build_module, "build_plugin", fake_build_plugin)

    build_module.main()

    assert (source_dir / "first.so").read_bytes() == b"\x7fELFfirst.cu"
    assert (source_dir / "second.so").read_bytes() == b"\x7fELFsecond.cu"
    output_dirs = {call.output_dir for call in build_calls}
    assert len(output_dirs) == 1
    output_dir = output_dirs.pop()
    assert sorted(build_calls, key=lambda call: call.source_name) == [
        PluginBuildCall(
            Path("/cuda/bin/nvcc"),
            source_dir,
            output_dir,
            tmp_path / "tensorrt_libs" / "libnvinfer.so.11",
            (Path("/cuda/cudart/include"), Path("/cuda/cublas/include")),
            "first.cu",
            (Path("/cuda/cudart/libcudart.so.13"),),
        ),
        PluginBuildCall(
            Path("/cuda/bin/nvcc"),
            source_dir,
            output_dir,
            tmp_path / "tensorrt_libs" / "libnvinfer.so.11",
            (Path("/cuda/cudart/include"), Path("/cuda/cublas/include")),
            "second.cu",
            (Path("/cuda/cublas/libcublas.so.13"),),
        ),
    ]
    assert not output_dir.exists()


def test_build_main_deduplicates_cuda_include_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass each discovered CUDA include directory to nvcc only once."""

    prepare_build_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        build_module,
        "find_nvidia_header_directory",
        lambda _name: "/cuda/include",
    )
    include_arguments: list[tuple[Path, ...]] = []
    calls_lock = Lock()

    def fake_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_library_paths,
        )
        with calls_lock:
            include_arguments.append(cuda_include_dirs)
        write_fake_plugin(output_dir / Path(source_name).with_suffix(".so").name)

    monkeypatch.setattr(build_module, "build_plugin", fake_build_plugin)

    build_module.main()

    assert include_arguments == [(Path("/cuda/include"),)] * len(TEST_PLUGIN_BUILDS)


@pytest.mark.parametrize(
    ("available_cpus", "expected_workers"),
    ((None, 1), (1, 1), (8, len(TEST_PLUGIN_BUILDS))),
)
def test_build_main_limits_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_cpus: int | None,
    expected_workers: int,
) -> None:
    """Bound workers by plugin count and handle an unavailable CPU count."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    worker_counts: list[int] = []

    def make_executor(max_workers: int) -> NativeThreadPoolExecutor:
        worker_counts.append(max_workers)
        return NativeThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(build_module, "cpu_count", lambda: available_cpus)
    monkeypatch.setattr(build_module, "ThreadPoolExecutor", make_executor)

    def fake_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        write_fake_plugin(output_dir / Path(source_name).with_suffix(".so").name)

    monkeypatch.setattr(build_module, "build_plugin", fake_build_plugin)

    build_module.main()

    assert worker_counts == [expected_workers]
    assert {path.name for path in source_dir.glob("*.so")} == {
        "first.so",
        "second.so",
    }


def test_build_main_keeps_installed_plugins_when_one_build_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave every installed plugin unchanged when one compiler fails."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    installed_plugins = write_installed_plugins(source_dir)

    def failing_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        if source_name == "second.cu":
            raise subprocess.CalledProcessError(1, source_name)
        (output_dir / "first.so").touch()

    monkeypatch.setattr(build_module, "build_plugin", failing_build_plugin)

    with pytest.raises(subprocess.CalledProcessError):
        build_module.main()

    for name, payload in installed_plugins.items():
        assert (source_dir / name).read_bytes() == payload
    assert not list(source_dir.glob(".plugin-build-*"))


def test_build_main_rejects_missing_nvcc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail dependency discovery before scheduling any compilation."""

    prepare_build_environment(tmp_path, monkeypatch)
    build_attempts = record_build_attempts(monkeypatch)
    monkeypatch.setattr(build_module, "find_nvidia_binary_utility", lambda _: None)

    with pytest.raises(RuntimeError, match="nvcc was not found"):
        build_module.main()

    assert build_attempts == []


def test_build_main_rejects_missing_tensorrt_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before compilation when TensorRT cannot be linked."""

    prepare_build_environment(tmp_path, monkeypatch)
    build_attempts = record_build_attempts(monkeypatch)
    tensorrt_library = (
        Path(build_module.tensorrt_libs.__file__).parent / "libnvinfer.so.11"
    )
    tensorrt_library.unlink()

    with pytest.raises(FileNotFoundError, match="TensorRT library not found"):
        build_module.main()

    assert build_attempts == []


def test_build_main_rejects_missing_cuda_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the exact CUDA component whose headers are unavailable."""

    prepare_build_environment(tmp_path, monkeypatch)
    build_attempts = record_build_attempts(monkeypatch)
    monkeypatch.setattr(
        build_module,
        "find_nvidia_header_directory",
        lambda name: None if name == "cublas" else f"/cuda/{name}/include",
    )

    with pytest.raises(RuntimeError, match="CUDA headers for cublas were not found"):
        build_module.main()

    assert build_attempts == []


def test_build_main_propagates_cuda_library_discovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop before compilation when a required CUDA library cannot be loaded."""

    prepare_build_environment(tmp_path, monkeypatch)
    build_attempts = record_build_attempts(monkeypatch)

    def load_cuda_library(name: str) -> SimpleNamespace:
        if name == "cublas":
            raise RuntimeError("CUDA library cublas was not found")
        return SimpleNamespace(abs_path=f"/cuda/{name}/lib{name}.so.13")

    monkeypatch.setattr(build_module, "load_nvidia_dynamic_lib", load_cuda_library)

    with pytest.raises(RuntimeError, match="CUDA library cublas was not found"):
        build_module.main()

    assert build_attempts == []


def test_build_main_keeps_installed_plugins_when_output_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install nothing when a compiler exits without one expected artifact."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    installed_plugins = write_installed_plugins(source_dir)

    def incomplete_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        if source_name == "first.cu":
            write_fake_plugin(output_dir / "first.so")

    monkeypatch.setattr(build_module, "build_plugin", incomplete_build_plugin)

    with pytest.raises(RuntimeError, match="Compiler output not found"):
        build_module.main()

    for name, payload in installed_plugins.items():
        assert (source_dir / name).read_bytes() == payload
    assert not list(source_dir.glob(".plugin-build-*"))


def test_build_main_keeps_installed_plugins_when_output_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install nothing when one compiler output contains no bytes."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    installed_plugins = write_installed_plugins(source_dir)

    def empty_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        output_path = output_dir / Path(source_name).with_suffix(".so").name
        if source_name == "second.cu":
            output_path.write_bytes(b"")
        else:
            write_fake_plugin(output_path)

    monkeypatch.setattr(build_module, "build_plugin", empty_build_plugin)

    with pytest.raises(RuntimeError, match="Compiler output is empty"):
        build_module.main()

    for name, payload in installed_plugins.items():
        assert (source_dir / name).read_bytes() == payload
    assert not list(source_dir.glob(".plugin-build-*"))


def test_build_main_replaces_existing_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace every installed plugin after a complete successful build."""

    source_dir = prepare_build_environment(tmp_path, monkeypatch)
    for name in ("first.so", "second.so"):
        write_fake_plugin(source_dir / name, b"old")

    def fake_build_plugin(
        nvcc_path: Path,
        plugin_source_dir: Path,
        output_dir: Path,
        tensorrt_library_path: Path,
        cuda_include_dirs: tuple[Path, ...],
        source_name: str,
        cuda_library_paths: tuple[Path, ...],
    ) -> None:
        del (
            nvcc_path,
            plugin_source_dir,
            tensorrt_library_path,
            cuda_include_dirs,
            cuda_library_paths,
        )
        write_fake_plugin(
            output_dir / Path(source_name).with_suffix(".so").name, b"new"
        )

    monkeypatch.setattr(build_module, "build_plugin", fake_build_plugin)

    build_module.main()

    for name in ("first.so", "second.so"):
        assert (source_dir / name).read_bytes() == b"\x7fELFnew"
