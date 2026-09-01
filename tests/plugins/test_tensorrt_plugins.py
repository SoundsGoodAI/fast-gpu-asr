#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for loading and registering packaged TensorRT plugins."""

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import fast_gpu_asr.tensorrt_plugins as plugins_module

PLUGIN_LIBRARIES = tuple(
    library_name for library_name, _ in plugins_module.PLUGIN_INITIALIZERS
)


class FakeInitializer:
    """Record ctypes signature assignment and registration calls."""

    def __init__(self, result: bool = True) -> None:
        self.argtypes = None
        self.restype = None
        self.result = result
        self.calls = 0
        self.call_signatures: list[tuple[object, object]] = []

    def __call__(self) -> bool:
        self.calls += 1
        self.call_signatures.append((self.argtypes, self.restype))
        return self.result


def bypass_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace TensorRT and CUDA dependency loading with successful no-ops."""

    monkeypatch.setattr(plugins_module, "import_module", lambda _: None)
    monkeypatch.setattr(plugins_module, "load_nvidia_dynamic_lib", lambda _: None)


def libraries_through(library_name: str) -> list[str]:
    """Return the ordered plugin-library prefix ending at ``library_name``."""

    return list(PLUGIN_LIBRARIES[: PLUGIN_LIBRARIES.index(library_name) + 1])


@pytest.fixture(autouse=True)
def clear_plugin_loader_cache() -> Iterator[None]:
    """Keep cached plugin registration isolated between tests."""

    plugins_module.load_tensorrt_plugins.cache_clear()
    yield
    plugins_module.load_tensorrt_plugins.cache_clear()


def test_load_tensorrt_plugins_registers_each_library_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    cuda_libraries: list[str] = []
    initializers = {
        initializer_name: FakeInitializer()
        for _, initializer_name in plugins_module.PLUGIN_INITIALIZERS
    }

    monkeypatch.setattr(
        plugins_module,
        "import_module",
        lambda name: imported.append(name),
    )
    monkeypatch.setattr(
        plugins_module,
        "load_nvidia_dynamic_lib",
        lambda name: cuda_libraries.append(name),
    )
    monkeypatch.setattr(
        plugins_module.ctypes,
        "CDLL",
        lambda path, mode: SimpleNamespace(**initializers),
    )

    plugins_module.load_tensorrt_plugins()
    plugins_module.load_tensorrt_plugins()

    assert imported == ["tensorrt_libs"]
    assert cuda_libraries == list(plugins_module.CUDA_RUNTIME_LIBRARIES)
    assert all(initializer.calls == 1 for initializer in initializers.values())
    assert all(
        initializer.call_signatures == [((), plugins_module.ctypes.c_bool)]
        for initializer in initializers.values()
    )


def test_load_tensorrt_plugins_uses_exact_library_initializer_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_events: list[tuple[Path, int]] = []
    initializers: dict[str, FakeInitializer] = {}
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        library_path = Path(path)
        library_name = library_path.name
        load_events.append((library_path, mode))
        initializer = FakeInitializer()
        initializers[library_name] = initializer
        return SimpleNamespace(**{expected_initializers[library_name]: initializer})

    bypass_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    plugins_module.load_tensorrt_plugins()

    plugin_dir = Path(plugins_module.__file__).resolve().parent
    assert load_events == [
        (plugin_dir / library_name, plugins_module.ctypes.RTLD_GLOBAL)
        for library_name, _ in plugins_module.PLUGIN_INITIALIZERS
    ]
    assert all(initializer.calls == 1 for initializer in initializers.values())


def test_failed_plugin_load_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)
    imported: list[str] = []
    cuda_libraries: list[str] = []
    initializers: list[FakeInitializer] = []
    attempts = 0

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        nonlocal attempts
        del mode
        attempts += 1
        if attempts == 1:
            raise OSError("transient failure")
        library_name = Path(path).name
        initializer = FakeInitializer()
        initializers.append(initializer)
        return SimpleNamespace(**{expected_initializers[library_name]: initializer})

    monkeypatch.setattr(
        plugins_module,
        "import_module",
        lambda name: imported.append(name),
    )
    monkeypatch.setattr(
        plugins_module,
        "load_nvidia_dynamic_lib",
        lambda name: cuda_libraries.append(name),
    )
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(RuntimeError, match="transient failure") as exc_info:
        plugins_module.load_tensorrt_plugins()
    plugins_module.load_tensorrt_plugins()

    assert attempts == len(plugins_module.PLUGIN_INITIALIZERS) + 1
    assert imported == ["tensorrt_libs", "tensorrt_libs"]
    assert cuda_libraries == list(plugins_module.CUDA_RUNTIME_LIBRARIES) * 2
    assert len(initializers) == len(plugins_module.PLUGIN_INITIALIZERS)
    assert all(initializer.calls == 1 for initializer in initializers)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_load_tensorrt_plugins_reports_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_libraries: list[str] = []
    plugin_opened = False

    def import_tensorrt_libraries(_: str) -> None:
        raise ModuleNotFoundError("tensorrt_libs")

    def fake_cdll(*args: object, **kwargs: object) -> None:
        nonlocal plugin_opened
        del args, kwargs
        plugin_opened = True

    monkeypatch.setattr(
        plugins_module,
        "import_module",
        import_tensorrt_libraries,
    )
    monkeypatch.setattr(
        plugins_module,
        "load_nvidia_dynamic_lib",
        lambda name: cuda_libraries.append(name),
    )
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(
        RuntimeError, match="TensorRT and CUDA runtime dependencies"
    ) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert cuda_libraries == []
    assert not plugin_opened
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert str(exc_info.value.__cause__) == "tensorrt_libs"


@pytest.mark.parametrize("failed_library", plugins_module.CUDA_RUNTIME_LIBRARIES)
@pytest.mark.parametrize("error_type", (OSError, RuntimeError))
def test_load_tensorrt_plugins_reports_each_cuda_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
    failed_library: str,
    error_type: type[OSError] | type[RuntimeError],
) -> None:
    attempted_libraries: list[str] = []
    plugin_opened = False

    def load_cuda_library(name: str) -> None:
        attempted_libraries.append(name)
        if name == failed_library:
            raise error_type(f"missing {name}")

    def open_plugin(*args: object, **kwargs: object) -> None:
        nonlocal plugin_opened
        del args, kwargs
        plugin_opened = True

    monkeypatch.setattr(plugins_module, "import_module", lambda _: None)
    monkeypatch.setattr(plugins_module, "load_nvidia_dynamic_lib", load_cuda_library)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", open_plugin)

    with pytest.raises(
        RuntimeError, match=rf"missing {re.escape(failed_library)}"
    ) as exc_info:
        plugins_module.load_tensorrt_plugins()

    failed_index = plugins_module.CUDA_RUNTIME_LIBRARIES.index(failed_library)
    assert attempted_libraries == list(
        plugins_module.CUDA_RUNTIME_LIBRARIES[: failed_index + 1]
    )
    assert not plugin_opened
    assert isinstance(exc_info.value.__cause__, error_type)


@pytest.mark.parametrize("failed_library", PLUGIN_LIBRARIES)
def test_load_tensorrt_plugins_reports_library_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    failed_library: str,
) -> None:
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)
    opened_libraries: list[str] = []

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        del mode
        library_name = Path(path).name
        opened_libraries.append(library_name)
        if library_name == failed_library:
            raise OSError("cannot load")
        return SimpleNamespace(
            **{expected_initializers[library_name]: FakeInitializer()}
        )

    bypass_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(
        RuntimeError,
        match=rf"Failed to load TensorRT plugin library .*{re.escape(failed_library)}",
    ) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert opened_libraries == libraries_through(failed_library)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "cannot load"


@pytest.mark.parametrize(
    ("failed_library", "failed_initializer"), plugins_module.PLUGIN_INITIALIZERS
)
def test_load_tensorrt_plugins_reports_missing_initializer(
    monkeypatch: pytest.MonkeyPatch,
    failed_library: str,
    failed_initializer: str,
) -> None:
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)
    opened_libraries: list[str] = []

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        del mode
        library_name = Path(path).name
        opened_libraries.append(library_name)
        if library_name == failed_library:
            return SimpleNamespace()
        return SimpleNamespace(
            **{expected_initializers[library_name]: FakeInitializer()}
        )

    bypass_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(
        RuntimeError, match=rf"does not export {re.escape(failed_initializer)}"
    ) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert opened_libraries == libraries_through(failed_library)
    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert failed_initializer in str(exc_info.value.__cause__)


@pytest.mark.parametrize("failed_library", PLUGIN_LIBRARIES)
def test_load_tensorrt_plugins_reports_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
    failed_library: str,
) -> None:
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)
    opened_libraries: list[str] = []
    initializers: dict[str, FakeInitializer] = {}

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        del mode
        library_name = Path(path).name
        opened_libraries.append(library_name)
        initializer_name = expected_initializers[library_name]
        initializer = FakeInitializer(library_name != failed_library)
        initializers[library_name] = initializer
        return SimpleNamespace(**{initializer_name: initializer})

    bypass_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Failed to register TensorRT plugins from .*"
            rf"{re.escape(failed_library)}"
        ),
    ) as exc_info:
        plugins_module.load_tensorrt_plugins()

    assert opened_libraries == libraries_through(failed_library)
    assert all(initializer.calls == 1 for initializer in initializers.values())
    assert all(
        initializer.call_signatures == [((), plugins_module.ctypes.c_bool)]
        for initializer in initializers.values()
    )
    assert not initializers[failed_library].result
    assert exc_info.value.__cause__ is None


def test_registration_failure_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_initializers = dict(plugins_module.PLUGIN_INITIALIZERS)
    failing_library = plugins_module.PLUGIN_INITIALIZERS[-1][0]
    initializers: dict[str, FakeInitializer] = {}

    def fake_cdll(path: str, mode: int) -> SimpleNamespace:
        del mode
        library_name = Path(path).name
        initializer = initializers.setdefault(
            library_name,
            FakeInitializer(result=library_name != failing_library),
        )
        return SimpleNamespace(**{expected_initializers[library_name]: initializer})

    bypass_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(plugins_module.ctypes, "CDLL", fake_cdll)

    with pytest.raises(RuntimeError, match="Failed to register TensorRT plugins"):
        plugins_module.load_tensorrt_plugins()
    initializers[failing_library].result = True
    plugins_module.load_tensorrt_plugins()

    assert initializers[failing_library].calls == 2
    assert all(initializer.calls == 2 for initializer in initializers.values())
