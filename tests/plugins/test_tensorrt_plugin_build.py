#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for native TensorRT plugin discovery, compilation, and installation."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import fast_gpu_asr.tensorrt_plugins.build as build_module

TEST_PLUGIN_BUILDS = (
    ("first.cu", ("cudart",)),
    ("second.cu", ("cublas", "cufft", "cudart")),
)
TEST_CUDA_BUILD_LIBRARIES = ("cudart", "cublas", "cufft")


def compile_output(command: tuple[str, ...], cwd: Path, check: bool) -> None:
    """Write sentinel compiler output instead of invoking nvcc.

    Parameters
    ----------
    command : tuple[str, ...]
        Compiler arguments containing an output path after ``-o``.
    cwd : Path
        Source directory passed to the subprocess.
    check : bool
        Whether the caller requests propagation of compiler failures.

    Notes
    -----
    The first plugin also emits an unlisted library to exercise manifest-only
    installation.
    """

    output = Path(command[command.index("-o") + 1])
    assert cwd.is_dir() and check
    output.write_bytes(output.name.encode())
    if output.name == "first.so":
        (output.parent / "unlisted.so").write_bytes(b"not a manifest entry")


@pytest.fixture
def build_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Isolate build files and dependency discovery while retaining real threads.

    Parameters
    ----------
    tmp_path : Path
        Per-test root for plugin sources and the TensorRT library placeholder.
    monkeypatch : pytest.MonkeyPatch
        Restores dependency discovery, subprocess, and executor bindings.

    Returns
    -------
    SimpleNamespace
        Source paths, original installed contents, and configurable mocks.
        Discovery and nvcc execution are mocked; compilation wiring, executor
        lifetime, and installation run through the production functions.
    """

    source_dir = tmp_path / "plugins"
    source_dir.mkdir()
    for source_name, _ in TEST_PLUGIN_BUILDS:
        (source_dir / source_name).write_text("// test source\n", encoding="utf8")
    installed = {
        "first.so": b"old first",
        "second.so": b"old second",
        "unrelated.so": b"untouched",
    }
    for name, contents in installed.items():
        (source_dir / name).write_bytes(contents)
    library_dir = tmp_path / "tensorrt_libs"
    library_dir.mkdir()
    tensorrt_library = library_dir / "libnvinfer.so.11"
    tensorrt_library.touch()
    environment = SimpleNamespace(
        source_dir=source_dir,
        tensorrt_library=tensorrt_library,
        installed=installed,
        nvcc=Mock(return_value="/cuda/bin/nvcc"),
        headers=Mock(side_effect=lambda name: f"/cuda/{name}/include"),
        libraries=Mock(
            side_effect=lambda name: SimpleNamespace(
                abs_path=f"/cuda/{name}/lib{name}.so.13"
            )
        ),
        compiler=Mock(side_effect=compile_output),
        executor=Mock(wraps=ThreadPoolExecutor),
    )
    monkeypatch.setattr(build_module, "__file__", str(source_dir / "build.py"))
    monkeypatch.setattr(
        build_module.tensorrt_libs, "__file__", str(library_dir / "__init__.py")
    )
    monkeypatch.setattr(build_module, "PLUGIN_BUILDS", TEST_PLUGIN_BUILDS)
    monkeypatch.setattr(build_module, "CUDA_BUILD_LIBRARIES", TEST_CUDA_BUILD_LIBRARIES)
    monkeypatch.setattr(build_module, "cpu_count", lambda: 2)
    monkeypatch.setattr(build_module, "find_nvidia_binary_utility", environment.nvcc)
    monkeypatch.setattr(
        build_module, "find_nvidia_header_directory", environment.headers
    )
    monkeypatch.setattr(build_module, "load_nvidia_dynamic_lib", environment.libraries)
    monkeypatch.setattr(build_module, "ThreadPoolExecutor", environment.executor)
    monkeypatch.setattr(build_module.subprocess, "run", environment.compiler)
    return environment


def assert_build_unchanged(environment: SimpleNamespace) -> None:
    """Check installed bytes and temporary-directory cleanup after a failed build.

    Parameters
    ----------
    environment : SimpleNamespace
        Build fixture containing source paths and the original plugin contents.
    """

    assert {
        path.name: path.read_bytes() for path in environment.source_dir.glob("*.so")
    } == environment.installed
    assert not list(environment.source_dir.glob(".plugin-build-*"))


def test_build_plugin_uses_exact_headers_and_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Mock()
    monkeypatch.setattr(build_module.subprocess, "run", run)
    output_dir = tmp_path / "output"

    build_module.build_plugin(
        Path("/cuda/bin/nvcc"),
        tmp_path,
        output_dir,
        Path("/tensorrt/lib/libnvinfer.so.11"),
        (Path("/cuda/include"), Path("/cublas/include")),
        "plugin.cu",
        (Path("/cuda/lib/libcudart.so.13"), Path("/cuda/lib/libcublas.so.13")),
    )

    run.assert_called_once_with(
        (
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
        ),
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    ("available_cpus", "expected_workers"), ((None, 1), (1, 1), (8, 2))
)
def test_main_builds_and_replaces_only_manifest_plugins(
    build_environment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    available_cpus: int | None,
    expected_workers: int,
) -> None:
    env = build_environment
    monkeypatch.setattr(build_module, "cpu_count", lambda: available_cpus)

    build_module.main()

    env.executor.assert_called_once_with(max_workers=expected_workers)
    env.nvcc.assert_called_once_with("nvcc")
    expected_discovery = [call(name) for name in TEST_CUDA_BUILD_LIBRARIES]
    assert (
        env.headers.call_args_list == env.libraries.call_args_list == expected_discovery
    )
    command = env.compiler.call_args.args[0]
    output_dir = Path(command[command.index("-o") + 1]).parent
    assert output_dir.parent == env.source_dir
    assert output_dir.name.startswith(".plugin-build-")
    include_options = tuple(
        option
        for name in TEST_CUDA_BUILD_LIBRARIES
        for option in ("-I", f"/cuda/{name}/include")
    )
    expected_calls = [
        call(
            (
                "/cuda/bin/nvcc",
                *build_module.NVCC_OPTIONS,
                *include_options,
                source_name,
                "-o",
                str(output_dir / Path(source_name).with_suffix(".so")),
                "-L",
                str(env.tensorrt_library.parent),
                *(option for name in libraries for option in ("-L", f"/cuda/{name}")),
                "-l:libnvinfer.so.11",
                *(f"-l:lib{name}.so.13" for name in libraries),
            ),
            cwd=env.source_dir,
            check=True,
        )
        for source_name, libraries in TEST_PLUGIN_BUILDS
    ]
    env.compiler.assert_has_calls(expected_calls, any_order=True)
    assert env.compiler.call_count == len(expected_calls)
    assert {path.name: path.read_bytes() for path in env.source_dir.glob("*.so")} == {
        "first.so": b"first.so",
        "second.so": b"second.so",
        "unrelated.so": b"untouched",
    }
    assert not output_dir.exists()


def test_main_deduplicates_cuda_include_directories(
    build_environment: SimpleNamespace,
) -> None:
    env = build_environment
    env.headers.side_effect = lambda _name: "/cuda/include"

    build_module.main()

    assert env.compiler.call_count == 2
    for invocation in env.compiler.call_args_list:
        command = invocation.args[0]
        assert command.count("-I") == command.count("/cuda/include") == 1


@pytest.mark.parametrize(
    ("dependency", "component"),
    (
        ("tensorrt", None),
        ("nvcc", None),
        *(("headers", name) for name in TEST_CUDA_BUILD_LIBRARIES),
        *(("library", name) for name in TEST_CUDA_BUILD_LIBRARIES),
    ),
)
def test_main_stops_before_compilation_when_discovery_fails(
    build_environment: SimpleNamespace, dependency: str, component: str | None
) -> None:
    env = build_environment
    header_count = library_count = 0
    if dependency == "tensorrt":
        env.tensorrt_library.unlink()
        failure = FileNotFoundError(
            f"TensorRT library not found: {env.tensorrt_library}"
        )
    elif dependency == "nvcc":
        env.nvcc.return_value = None
        failure = RuntimeError("CUDA compiler nvcc was not found.")
    else:
        index = TEST_CUDA_BUILD_LIBRARIES.index(component)
        header_count = index + 1
        library_count = index if dependency == "headers" else index + 1
        if dependency == "headers":
            env.headers.side_effect = [
                *(
                    f"/cuda/{name}/include"
                    for name in TEST_CUDA_BUILD_LIBRARIES[:index]
                ),
                None,
            ]
            failure = RuntimeError(f"CUDA headers for {component} were not found.")
        else:
            failure = RuntimeError(f"CUDA library {component} was not found")
            env.libraries.side_effect = [
                *(
                    SimpleNamespace(abs_path=f"/cuda/{name}/lib{name}.so.13")
                    for name in TEST_CUDA_BUILD_LIBRARIES[:index]
                ),
                failure,
            ]

    with pytest.raises(type(failure)) as error:
        build_module.main()

    assert str(error.value) == str(failure)
    if dependency == "library":
        assert error.value is failure
    if dependency == "tensorrt":
        env.nvcc.assert_not_called()
    else:
        env.nvcc.assert_called_once_with("nvcc")
    assert env.headers.call_args_list == [
        call(name) for name in TEST_CUDA_BUILD_LIBRARIES[:header_count]
    ]
    assert env.libraries.call_args_list == [
        call(name) for name in TEST_CUDA_BUILD_LIBRARIES[:library_count]
    ]
    env.executor.assert_not_called()
    env.compiler.assert_not_called()
    assert_build_unchanged(env)


def test_main_preserves_plugins_and_propagates_compiler_failure(
    build_environment: SimpleNamespace,
) -> None:
    env = build_environment
    failure = subprocess.CalledProcessError(2, ("nvcc", "second.cu"))

    def fail_second(command, cwd, check) -> None:
        """Produce the first output and fail compilation of the second.

        Parameters
        ----------
        command : tuple[str, ...]
            Compiler arguments identifying the source file.
        cwd : Path
            Source directory forwarded to the successful compiler stand-in.
        check : bool
            Subprocess failure policy forwarded unchanged.
        """

        if "second.cu" in command:
            raise failure
        compile_output(command, cwd, check)

    env.compiler.side_effect = fail_second

    with pytest.raises(subprocess.CalledProcessError) as error:
        build_module.main()

    assert error.value is failure
    assert env.compiler.call_count == 2
    assert_build_unchanged(env)


def test_main_drains_workers_before_cleanup_on_failure(
    build_environment: SimpleNamespace,
) -> None:
    env = build_environment
    second_started, first_failed, release_second, second_finished = (
        Event() for _ in range(4)
    )
    failure = subprocess.CalledProcessError(1, ("nvcc", "first.cu"))

    def controlled_compile(command, cwd, check) -> None:
        """Hold the second compiler until the first has failed.

        Parameters
        ----------
        command : tuple[str, ...]
            Compiler arguments distinguishing the two worker jobs.
        cwd : Path
            Source directory passed through to the compiler stand-in.
        check : bool
            Subprocess failure policy passed through unchanged.
        """

        if "first.cu" in command:
            assert second_started.wait(timeout=10), "Second compiler did not start."
            first_failed.set()
            raise failure
        second_started.set()
        assert release_second.wait(timeout=10), "Second compiler was not released."
        compile_output(command, cwd, check)
        second_finished.set()

    def run_build() -> None:
        """Run the real build and require worker completion before it returns.

        Raises
        ------
        subprocess.CalledProcessError
            Expected first-worker compilation failure.
        AssertionError
            If the caller exits before the blocked second compiler finishes.
        """

        try:
            build_module.main()
        finally:
            assert second_finished.is_set(), (
                "Build exited with a compiler still running."
            )

    env.compiler.side_effect = controlled_compile
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(run_build)
        try:
            assert first_failed.wait(timeout=10), "First compiler did not fail."
        finally:
            release_second.set()
        with pytest.raises(subprocess.CalledProcessError) as error:
            future.result(timeout=10)

    assert error.value is failure
    assert_build_unchanged(env)


@pytest.mark.parametrize(
    ("invalid_output", "message"),
    (
        ("missing", "Compiler output not found"),
        ("empty", "Compiler output is empty"),
        ("directory", "Compiler output not found"),
        ("symlink", "Compiler output is a symbolic link"),
    ),
)
@pytest.mark.parametrize("invalid_name", ("first.so", "second.so"))
def test_main_checks_every_output_before_installing(
    build_environment: SimpleNamespace,
    invalid_output: str,
    message: str,
    invalid_name: str,
) -> None:
    env = build_environment

    def invalid_compile(command, cwd, check) -> None:
        """Create one invalid artifact while compiling its peer normally.

        Parameters
        ----------
        command : tuple[str, ...]
            Compiler arguments carrying the output path.
        cwd : Path
            Source directory passed to the successful compiler stand-in.
        check : bool
            Subprocess failure policy passed through unchanged.
        """

        output = Path(command[command.index("-o") + 1])
        if output.name != invalid_name:
            compile_output(command, cwd, check)
        elif invalid_output == "empty":
            output.touch()
        elif invalid_output == "directory":
            output.mkdir()
        elif invalid_output == "symlink":
            target = output.with_suffix(".target")
            target.write_bytes(b"compiled")
            output.symlink_to(target.name)

    env.compiler.side_effect = invalid_compile

    with pytest.raises(RuntimeError, match=message) as error:
        build_module.main()

    assert str(error.value).endswith(f"/{invalid_name}")
    assert env.compiler.call_count == 2
    assert_build_unchanged(env)
