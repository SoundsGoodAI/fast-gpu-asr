#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Build-time sources and runtime loading for native TensorRT plugins.

The packaged shared libraries implement specialized Zipformer and Parakeet
operators that TensorRT cannot represent as efficiently with standard layers.
Call :func:`load_tensorrt_plugins` before parsing plugin-bearing ONNX graphs or
deserializing their engines. The separate :mod:`build` module compiles the CUDA
C++ sources into the shared libraries distributed with the package.
"""

import ctypes
from functools import cache
from importlib import import_module
from pathlib import Path

from cuda.pathfinder import load_nvidia_dynamic_lib

from .constants import CUDA_RUNTIME_LIBRARIES, PLUGIN_INITIALIZERS


@cache
def load_tensorrt_plugins() -> None:
    """Load and register every packaged TensorRT plugin.

    TensorRT and the required CUDA shared libraries are loaded first so native
    plugin dependencies resolve from NVIDIA wheels, Conda, or a system CUDA
    Toolkit. Each plugin library is then loaded with process-global symbol
    visibility and its exported registration entry point is invoked. The first
    successful call is cached, making subsequent calls no-ops.

    Raises
    ------
    RuntimeError
        Raised when a required CUDA library or plugin library cannot be loaded,
        a plugin does not export its expected initializer, or TensorRT creator
        registration fails.
    """

    import_module("tensorrt_libs")
    for library_name in CUDA_RUNTIME_LIBRARIES:
        load_nvidia_dynamic_lib(library_name)

    plugin_dir = Path(__file__).resolve().parent
    for library_name, initializer_name in PLUGIN_INITIALIZERS:
        library_path = plugin_dir / library_name
        try:
            library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            initializer = getattr(library, initializer_name)
        except OSError as error:
            raise RuntimeError(
                f"Failed to load TensorRT plugin library {library_path}: {error}"
            ) from error
        except AttributeError as error:
            raise RuntimeError(
                f"TensorRT plugin library {library_path} does not export "
                f"{initializer_name}."
            ) from error

        initializer.argtypes = ()
        initializer.restype = ctypes.c_bool
        if not initializer():
            raise RuntimeError(
                f"Failed to register TensorRT plugins from {library_path}."
            )
