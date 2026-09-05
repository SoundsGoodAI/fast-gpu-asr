#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for loading and registering packaged TensorRT plugins."""

import ctypes
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import fast_gpu_asr.tensorrt_plugins as plugins_module

PLUGIN_DIRECTORY = Path(plugins_module.__file__).resolve().parent
PLUGIN_LIBRARIES = tuple(
    library_name for library_name, _ in plugins_module.PLUGIN_INITIALIZERS
)
PLUGIN_LOAD_CALLS = [
    call(str(PLUGIN_DIRECTORY / name), mode=ctypes.RTLD_GLOBAL)
    for name in PLUGIN_LIBRARIES
]
PLUGIN_REGISTRATION_CHECK = """
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[1]))

import fast_gpu_asr.constants as constants
import fast_gpu_asr.tensorrt_plugins as plugins
import tensorrt as trt

plugin_dir = Path(plugins.__file__).resolve().parent
assert plugin_dir == Path(sys.argv[1]).resolve(), plugin_dir
plugin_names = sorted(
    value for name, value in vars(constants).items()
    if name.endswith("_PLUGIN_NAME")
)
assert plugin_names, "No TensorRT plugin names are declared."
plugins.load_tensorrt_plugins()
gc.collect()
registries = {
    "runtime": trt.get_plugin_registry(),
    "builder": trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD),
}
for registry_name, registry in registries.items():
    missing = [
        name for name in plugin_names
        if registry.get_creator(name, "1", constants.TENSORRT_PLUGIN_NAMESPACE) is None
    ]
    assert not missing, f"Missing {registry_name} TensorRT creators: {missing}."
"""


class FakeInitializer:
    """Check the ctypes signature at registration time and count calls."""

    def __init__(self):
        """Initialize writable ctypes metadata and registration controls."""

        self.argtypes = None
        self.restype = None
        self.result = True
        self.on_call = None
        self.calls = 0

    def __call__(self):
        """Validate the configured ABI and invoke the optional registration hook.

        Returns
        -------
        bool
            Configured registration result after counting the call and invoking its
            hook.
        """

        assert self.argtypes == ()
        assert self.restype is ctypes.c_bool
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        return self.result


@pytest.fixture(autouse=True)
def clear_plugin_loader_cache() -> Iterator[None]:
    """Isolate successful-load caching between tests.

    Yields
    ------
    None
        Test execution with the successful-load cache cleared before and after.
    """

    plugins_module.load_tensorrt_plugins.cache_clear()
    yield
    plugins_module.load_tensorrt_plugins.cache_clear()


@pytest.fixture
def loader(monkeypatch):
    """Stub external loads with a separate library and initializer for each entry.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture restoring loader dependencies after each test.

    Returns
    -------
    SimpleNamespace
        Dependency mocks, per-library fake handles, and controllable initializers.
    """

    initializers = {name: FakeInitializer() for name in PLUGIN_LIBRARIES}
    libraries = {
        name: SimpleNamespace(**{symbol: initializers[name]})
        for name, symbol in plugins_module.PLUGIN_INITIALIZERS
    }
    loader = SimpleNamespace(
        import_module=Mock(return_value=None),
        load_cuda=Mock(return_value=None),
        cdll=Mock(side_effect=lambda path, mode: libraries[Path(path).name]),
        libraries=libraries,
        initializers=initializers,
    )
    monkeypatch.setattr(plugins_module, "import_module", loader.import_module)
    monkeypatch.setattr(plugins_module, "load_nvidia_dynamic_lib", loader.load_cuda)
    monkeypatch.setattr(
        plugins_module,
        "ctypes",
        SimpleNamespace(
            CDLL=loader.cdll, RTLD_GLOBAL=ctypes.RTLD_GLOBAL, c_bool=ctypes.c_bool
        ),
    )
    return loader


@pytest.mark.cuda
def test_packaged_plugins_register_actual_creators_in_fresh_process() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", PLUGIN_REGISTRATION_CHECK, str(PLUGIN_DIRECTORY)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loads_dependencies_then_registers_each_library_once(loader) -> None:
    events = []
    loader.import_module.side_effect = lambda name: events.append(("import", name))
    loader.load_cuda.side_effect = lambda name: events.append(("cuda", name))

    def open_library(path, mode):
        """Record a library-open event and return its isolated fake handle.

        Parameters
        ----------
        path : str
            Shared-library path requested by the production loader.
        mode : int
            ctypes loading flags supplied by the production loader.

        Returns
        -------
        SimpleNamespace
            Fake library exposing the initializer requested by the production manifest.
        """

        name = Path(path).name
        events.append(("open", name))
        return loader.libraries[name]

    loader.cdll.side_effect = open_library
    for name, initializer in loader.initializers.items():
        initializer.on_call = lambda name=name: events.append(("initialize", name))

    assert plugins_module.load_tensorrt_plugins() is None
    assert plugins_module.load_tensorrt_plugins() is None

    expected = [("import", "tensorrt_libs")]
    expected.extend(("cuda", name) for name in plugins_module.CUDA_RUNTIME_LIBRARIES)
    for name in PLUGIN_LIBRARIES:
        expected.extend([("open", name), ("initialize", name)])
    assert events == expected
    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS
    assert all(initializer.calls == 1 for initializer in loader.initializers.values())


def test_failed_library_load_is_retried_then_cached(loader) -> None:
    loader.cdll.side_effect = [OSError("transient failure"), *loader.libraries.values()]

    with pytest.raises(RuntimeError, match="Failed to load TensorRT plugin library"):
        plugins_module.load_tensorrt_plugins()
    plugins_module.load_tensorrt_plugins()
    plugins_module.load_tensorrt_plugins()

    assert loader.cdll.call_args_list == [PLUGIN_LOAD_CALLS[0], *PLUGIN_LOAD_CALLS]
    assert loader.import_module.call_args_list == [call("tensorrt_libs")] * 2
    assert (
        loader.load_cuda.call_args_list
        == [call(name) for name in plugins_module.CUDA_RUNTIME_LIBRARIES] * 2
    )
    assert all(initializer.calls == 1 for initializer in loader.initializers.values())


@pytest.mark.parametrize(
    "error_type", [ImportError, ModuleNotFoundError, OSError, RuntimeError]
)
def test_reports_dependency_import_failure(loader, error_type) -> None:
    error = error_type("tensorrt_libs")
    loader.import_module.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    loader.import_module.assert_called_once_with("tensorrt_libs")
    loader.load_cuda.assert_not_called()
    loader.cdll.assert_not_called()
    assert str(exc_info.value) == (
        "Failed to load the TensorRT and CUDA runtime dependencies required by "
        "native plugins: tensorrt_libs"
    )
    assert exc_info.value.__cause__ is error


def test_propagates_unexpected_dependency_error(loader) -> None:
    error = ValueError("invalid dependency request")
    loader.import_module.side_effect = error

    with pytest.raises(ValueError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert exc_info.value is error
    loader.import_module.assert_called_once_with("tensorrt_libs")
    loader.load_cuda.assert_not_called()
    loader.cdll.assert_not_called()


@pytest.mark.parametrize("failed_library", plugins_module.CUDA_RUNTIME_LIBRARIES)
@pytest.mark.parametrize("error_type", [ImportError, OSError, RuntimeError])
def test_reports_each_cuda_dependency_failure(
    loader, failed_library, error_type
) -> None:
    error = error_type(f"missing {failed_library}")
    failed_index = plugins_module.CUDA_RUNTIME_LIBRARIES.index(failed_library)
    loader.load_cuda.side_effect = [None] * failed_index + [error]

    with pytest.raises(RuntimeError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    loader.import_module.assert_called_once_with("tensorrt_libs")
    assert loader.load_cuda.call_args_list == [
        call(name) for name in plugins_module.CUDA_RUNTIME_LIBRARIES[: failed_index + 1]
    ]
    loader.cdll.assert_not_called()
    assert str(exc_info.value) == (
        "Failed to load the TensorRT and CUDA runtime dependencies required by "
        f"native plugins: {error}"
    )
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "failed_index", range(len(PLUGIN_LIBRARIES)), ids=PLUGIN_LIBRARIES
)
def test_reports_library_load_failure(loader, failed_index) -> None:
    error = OSError("cannot load")
    loader.cdll.side_effect = list(loader.libraries.values())[:failed_index] + [error]

    with pytest.raises(RuntimeError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS[: failed_index + 1]
    failed_path = PLUGIN_DIRECTORY / PLUGIN_LIBRARIES[failed_index]
    assert str(exc_info.value) == (
        f"Failed to load TensorRT plugin library {failed_path}: {error}"
    )
    assert exc_info.value.__cause__ is error


def test_propagates_unexpected_library_loader_error(loader) -> None:
    error = ValueError("invalid loader arguments")
    loader.cdll.side_effect = error

    with pytest.raises(ValueError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert exc_info.value is error
    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS[:1]


@pytest.mark.parametrize(
    "failed_index", range(len(PLUGIN_LIBRARIES)), ids=PLUGIN_LIBRARIES
)
def test_reports_missing_initializer(loader, failed_index) -> None:
    library, symbol = plugins_module.PLUGIN_INITIALIZERS[failed_index]
    delattr(loader.libraries[library], symbol)

    with pytest.raises(RuntimeError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS[: failed_index + 1]
    failed_path = PLUGIN_DIRECTORY / library
    assert str(exc_info.value) == (
        f"TensorRT plugin library {failed_path} does not export {symbol}."
    )
    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert symbol in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    "failed_index", range(len(PLUGIN_LIBRARIES)), ids=PLUGIN_LIBRARIES
)
def test_reports_registration_failure(loader, failed_index) -> None:
    failed_library = PLUGIN_LIBRARIES[failed_index]
    loader.initializers[failed_library].result = False

    with pytest.raises(RuntimeError) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS[: failed_index + 1]
    assert [initializer.calls for initializer in loader.initializers.values()] == (
        [1] * (failed_index + 1) + [0] * (len(PLUGIN_LIBRARIES) - failed_index - 1)
    )
    assert str(exc_info.value) == (
        f"Failed to register TensorRT plugins from {PLUGIN_DIRECTORY / failed_library}."
    )
    assert exc_info.value.__cause__ is None


def test_failed_registration_is_retried_then_cached(loader) -> None:
    initializer = loader.initializers[PLUGIN_LIBRARIES[-1]]
    initializer.result = False

    with pytest.raises(RuntimeError, match="Failed to register TensorRT plugins"):
        plugins_module.load_tensorrt_plugins()
    initializer.result = True
    plugins_module.load_tensorrt_plugins()
    plugins_module.load_tensorrt_plugins()

    assert loader.cdll.call_args_list == PLUGIN_LOAD_CALLS * 2
    assert all(initializer.calls == 2 for initializer in loader.initializers.values())
