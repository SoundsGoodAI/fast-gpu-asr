#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Shared compilation helpers for native TensorRT plugin tests."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from fast_gpu_asr.tensorrt_plugins.constants import (
    CUDA_ARCHITECTURE_OPTIONS,
    NVCC_OPTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def compile_and_load_plugin(
    tmp_path_factory: pytest.TempPathFactory,
    source_name: str,
    initializer_name: str,
    cuda_libraries: Sequence[str],
) -> ctypes.CDLL:
    """Compile, load, and initialize one plugin for the active CUDA device.

    The helper deliberately resolves the compiler, headers, and exact shared
    libraries through ``cuda-pathfinder``. This mirrors wheel installations,
    where CUDA development components come from Python packages rather than a
    system-wide toolkit. Tests compile only for the active device architecture
    to avoid paying the production fat-binary build cost for every test module.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory used to isolate the compiled test library.
    source_name : str
        Plugin source filename in ``src/fast_gpu_asr/tensorrt_plugins``.
    initializer_name : str
        Exported registration function called after loading the library.
    cuda_libraries : Sequence[str]
        CUDA libraries required by the plugin, such as ``cudart`` or ``cublas``.

    Returns
    -------
    ctypes.CDLL
        Process-lifetime handle for the initialized plugin library.

    Raises
    ------
    FileNotFoundError
        Raised when the requested source, TensorRT library, or compiler output
        is missing.
    RuntimeError
        Raised when CUDA headers are unavailable, the active compute capability
        is malformed, compiler output is empty, or registration fails.
    ValueError
        Raised when ``source_name`` is not a filename relative to the plugin
        source directory.
    """

    source_dir = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fast_gpu_asr"
        / "tensorrt_plugins"
    )
    if Path(source_name).name != source_name:
        raise ValueError(f"Plugin source must be a filename: {source_name}")
    source_path = source_dir / source_name
    if not source_path.is_file():
        raise FileNotFoundError(f"Plugin source not found: {source_path}")

    cuda_pathfinder = pytest.importorskip("cuda.pathfinder")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("tensorrt")
    tensorrt_libs = pytest.importorskip("tensorrt_libs")

    nvcc = cuda_pathfinder.find_nvidia_binary_utility("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required to compile the native plugin test library.")

    tensorrt_library_path = (
        Path(tensorrt_libs.__file__).resolve().parent / "libnvinfer.so.11"
    )
    if not tensorrt_library_path.is_file():
        raise FileNotFoundError(f"TensorRT library not found: {tensorrt_library_path}")

    cuda_include_dirs: list[Path] = []
    cuda_library_paths: list[Path] = []
    for name in cuda_libraries:
        include_dir = cuda_pathfinder.find_nvidia_header_directory(name)
        if include_dir is None:
            raise RuntimeError(f"CUDA headers for {name} were not found.")
        include_path = Path(include_dir)
        if include_path not in cuda_include_dirs:
            cuda_include_dirs.append(include_path)
        library_path = Path(cuda_pathfinder.load_nvidia_dynamic_lib(name).abs_path)
        if not library_path.is_file():
            raise FileNotFoundError(f"CUDA library not found: {library_path}")
        cuda_library_paths.append(library_path)

    library_paths = (tensorrt_library_path, *cuda_library_paths)
    library_dirs = tuple(dict.fromkeys(path.parent for path in library_paths))
    compute_capability = str(cp.cuda.Device().compute_capability)
    if not compute_capability.isdecimal():
        raise RuntimeError(f"Invalid CUDA compute capability: {compute_capability!r}")
    architecture_start = NVCC_OPTIONS.index(CUDA_ARCHITECTURE_OPTIONS[0])
    architecture_end = architecture_start + len(CUDA_ARCHITECTURE_OPTIONS)
    nvcc_options = (
        *NVCC_OPTIONS[:architecture_start],
        (
            f"--generate-code=arch=compute_{compute_capability},"
            f"code=sm_{compute_capability}"
        ),
        *NVCC_OPTIONS[architecture_end:],
    )
    output_path = (
        tmp_path_factory.mktemp("tensorrt_plugins")
        / source_path.with_suffix(".so").name
    )
    subprocess.run(
        (
            str(nvcc),
            *nvcc_options,
            *(option for path in cuda_include_dirs for option in ("-I", str(path))),
            str(source_path),
            "-o",
            str(output_path),
            *(
                option
                for library_dir in library_dirs
                for option in ("-L", str(library_dir))
            ),
            *(f"-l:{path.name}" for path in library_paths),
        ),
        check=True,
    )
    if not output_path.is_file():
        raise FileNotFoundError(f"Compiler output not found: {output_path}")
    if output_path.stat().st_size == 0:
        raise RuntimeError(f"Compiler output is empty: {output_path}")

    library = ctypes.CDLL(str(output_path), mode=ctypes.RTLD_GLOBAL)
    initializer = getattr(library, initializer_name)
    initializer.argtypes = ()
    initializer.restype = ctypes.c_bool
    if not initializer():
        raise RuntimeError(f"Plugin registration failed: {output_path}")
    if not initializer():
        raise RuntimeError(f"Plugin registration is not idempotent: {output_path}")
    return library
