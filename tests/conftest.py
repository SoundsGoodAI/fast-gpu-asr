#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Configure hardware requirements for CUDA-dependent pytest tests.

The ``cuda`` marker requires any working CUDA device. The ``sm80`` marker is
more restrictive and requires compute capability 8.0 or newer. Tests that the
available hardware cannot execute are skipped during collection so the CPU test
suite remains usable on hosts without an NVIDIA GPU.
"""

import pytest


def cuda_compute_capability() -> int | None:
    """Return the active CUDA device's encoded compute capability.

    Returns
    -------
    int or None
        Compute capability encoded as ``major * 10 + minor``, such as ``80``
        for SM80 or ``120`` for SM120. ``None`` indicates that CuPy is not
        installed or that no CUDA device can be queried through the runtime.
    """

    try:
        import cupy as cp
    except ImportError:
        return None

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
        return int(cp.cuda.Device().compute_capability)
    except cp.cuda.runtime.CUDARuntimeError:
        return None


def pytest_configure(config: pytest.Config) -> None:
    """Register the CUDA markers used by the test suite.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration receiving the marker declarations.
    """

    config.addinivalue_line("markers", "cuda: requires a working CUDA device")
    config.addinivalue_line(
        "markers",
        "sm80: requires a working CUDA device with compute capability 8.0 or newer",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip collected tests whose CUDA requirements are not satisfied.

    Parameters
    ----------
    items : list[pytest.Item]
        Collected tests, including markers inherited from modules, classes, and
        parametrized cases.

    Notes
    -----
    CUDA probing is deferred until at least one selected test has a hardware
    marker. An ``sm80`` marker includes the general CUDA requirement and takes
    precedence so its more specific skip reason is preserved.
    """

    # Keep targeted CPU-only test runs from initializing the CUDA runtime.
    if not any(
        item.get_closest_marker("cuda") is not None
        or item.get_closest_marker("sm80") is not None
        for item in items
    ):
        return

    capability = cuda_compute_capability()
    cuda_unavailable = pytest.mark.skip(reason="A working CUDA device is required.")
    sm80_unavailable = pytest.mark.skip(
        reason="A working SM80 or newer CUDA device is required."
    )
    for item in items:
        if item.get_closest_marker("sm80") is not None and (
            capability is None or capability < 80
        ):
            item.add_marker(sm80_unavailable)
        elif item.get_closest_marker("cuda") is not None and capability is None:
            item.add_marker(cuda_unavailable)
