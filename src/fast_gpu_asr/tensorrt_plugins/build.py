#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Build the native TensorRT plugins shipped with fast-gpu-asr.

All plugins are compiled in parallel into temporary files and installed beside
their sources only after every compiler process succeeds.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorrt_libs
from cuda.pathfinder import (
    find_nvidia_binary_utility,
    find_nvidia_header_directory,
    load_nvidia_dynamic_lib,
)

from .constants import CUDA_BUILD_LIBRARIES, NVCC_OPTIONS, PLUGIN_BUILDS


def build_plugin(
    nvcc_path: Path,
    source_dir: Path,
    output_dir: Path,
    tensorrt_library_path: Path,
    cuda_include_dirs: tuple[Path, ...],
    source_name: str,
    cuda_library_paths: tuple[Path, ...],
) -> None:
    """Compile one TensorRT plugin into ``output_dir``.

    Parameters
    ----------
    nvcc_path : Path
        Absolute path to the CUDA compiler driver.
    source_dir : Path
        Directory containing the plugin source and shared headers.
    output_dir : Path
        Temporary directory that receives the compiled shared library.
    tensorrt_library_path : Path
        Exact TensorRT shared library against which the plugin is linked.
    cuda_include_dirs : tuple[Path, ...]
        CUDA include directories located by ``cuda-pathfinder``.
    source_name : str
        Plugin source filename relative to ``source_dir``.
    cuda_library_paths : tuple[Path, ...]
        Exact CUDA shared-library paths required by the plugin.

    Raises
    ------
    subprocess.CalledProcessError
        Raised when compilation or linking fails.
    """

    output_path = output_dir / Path(source_name).with_suffix(".so").name
    include_options = tuple(
        option
        for include_dir in cuda_include_dirs
        for option in ("-I", str(include_dir))
    )
    library_paths = (tensorrt_library_path, *cuda_library_paths)
    library_dirs = tuple(dict.fromkeys(path.parent for path in library_paths))
    library_options = (
        *(
            option
            for library_dir in library_dirs
            for option in ("-L", str(library_dir))
        ),
        *(f"-l:{path.name}" for path in library_paths),
    )
    subprocess.run(
        (
            str(nvcc_path),
            *NVCC_OPTIONS,
            *include_options,
            source_name,
            "-o",
            str(output_path),
            *library_options,
        ),
        cwd=source_dir,
        check=True,
    )


def main() -> None:
    """Compile every plugin, then replace each installed library in place.

    Raises
    ------
    RuntimeError
        Raised when the CUDA compiler, required CUDA dependencies, or expected
        compiler outputs cannot be located.
    FileNotFoundError
        Raised when the TensorRT runtime library is unavailable.
    subprocess.CalledProcessError
        Raised when any plugin fails to compile or link.
    """

    source_dir = Path(__file__).resolve().parent
    tensorrt_lib_dir = Path(tensorrt_libs.__file__).resolve().parent
    tensorrt_library_path = tensorrt_lib_dir / "libnvinfer.so.11"
    if not tensorrt_library_path.is_file():
        raise FileNotFoundError(f"TensorRT library not found: {tensorrt_library_path}")

    nvcc = find_nvidia_binary_utility("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA compiler nvcc was not found.")
    nvcc_path = Path(nvcc)

    cuda_include_dirs: list[Path] = []
    cuda_library_paths: dict[str, Path] = {}
    for library_name in CUDA_BUILD_LIBRARIES:
        include_dir = find_nvidia_header_directory(library_name)
        if include_dir is None:
            raise RuntimeError(f"CUDA headers for {library_name} were not found.")
        include_path = Path(include_dir)
        if include_path not in cuda_include_dirs:
            cuda_include_dirs.append(include_path)

        cuda_library_paths[library_name] = Path(
            load_nvidia_dynamic_lib(library_name).abs_path
        )
    cuda_include_dirs = tuple(cuda_include_dirs)

    with TemporaryDirectory(prefix=".plugin-build-", dir=source_dir) as temporary_dir:
        output_dir = Path(temporary_dir)
        max_workers = min(len(PLUGIN_BUILDS), cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = tuple(
                executor.submit(
                    build_plugin,
                    nvcc_path,
                    source_dir,
                    output_dir,
                    tensorrt_library_path,
                    cuda_include_dirs,
                    source_name,
                    tuple(cuda_library_paths[name] for name in libraries),
                )
                for source_name, libraries in PLUGIN_BUILDS
            )
            for future in futures:
                future.result()

        output_paths = tuple(
            output_dir / Path(source_name).with_suffix(".so").name
            for source_name, _ in PLUGIN_BUILDS
        )
        for output_path in output_paths:
            if not output_path.is_file():
                raise RuntimeError(f"Compiler output not found: {output_path}")
            if output_path.stat().st_size == 0:
                raise RuntimeError(f"Compiler output is empty: {output_path}")

        for output_path in output_paths:
            output_path.replace(source_dir / output_path.name)


if __name__ == "__main__":
    main()
