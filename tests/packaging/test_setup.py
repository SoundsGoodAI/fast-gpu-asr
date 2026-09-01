#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for platform-specific wheel metadata and artifact validation."""

import os
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import setuptools
from setuptools import Distribution
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist
from setuptools.errors import FileError

from fast_gpu_asr.tensorrt_plugins.constants import PLUGIN_BUILDS

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
SETUP_PATH = REPOSITORY_DIR / "setup.py"
PLUGIN_DIR = REPOSITORY_DIR / "src" / "fast_gpu_asr" / "tensorrt_plugins"


@dataclass(frozen=True)
class SetupExecution:
    """Classes and setup arguments produced by one execution of ``setup.py``."""

    distribution_type: type[Distribution]
    wheel_type: type[bdist_wheel]
    source_distribution_type: type[sdist]
    setup_calls: tuple[dict[str, Any], ...]


@pytest.fixture
def setup_execution(monkeypatch: pytest.MonkeyPatch) -> SetupExecution:
    """Execute ``setup.py`` without invoking setuptools' command parser."""

    setup_calls: list[dict[str, Any]] = []

    def capture_setup(**kwargs: Any) -> None:
        setup_calls.append(kwargs)

    monkeypatch.setattr(setuptools, "setup", capture_setup)
    namespace = runpy.run_path(
        str(SETUP_PATH),
        run_name="fast_gpu_asr_test_setup",
    )

    distribution_type = namespace.get("BinaryDistribution")
    wheel_type = namespace.get("BinaryWheel")
    source_distribution_type = namespace.get("UnsupportedSourceDistribution")
    if not isinstance(distribution_type, type) or not issubclass(
        distribution_type, Distribution
    ):
        raise TypeError("BinaryDistribution is not a setuptools Distribution class.")
    if not isinstance(wheel_type, type) or not issubclass(wheel_type, bdist_wheel):
        raise TypeError("BinaryWheel is not a bdist_wheel command class.")
    if not isinstance(source_distribution_type, type) or not issubclass(
        source_distribution_type, sdist
    ):
        raise TypeError("UnsupportedSourceDistribution is not an sdist command class.")

    return SetupExecution(
        distribution_type=distribution_type,
        wheel_type=wheel_type,
        source_distribution_type=source_distribution_type,
        setup_calls=tuple(setup_calls),
    )


@pytest.fixture
def wheel_command(setup_execution: SetupExecution, tmp_path: Path) -> bdist_wheel:
    """Create a binary-wheel command using an isolated plugin directory."""

    command = setup_execution.wheel_type(setup_execution.distribution_type())
    command.plugin_dir = tmp_path
    return command


@pytest.fixture
def base_wheel_runs(monkeypatch: pytest.MonkeyPatch) -> list[bdist_wheel]:
    """Record delegation to setuptools' base wheel command."""

    calls: list[bdist_wheel] = []

    def record_run(command: bdist_wheel) -> None:
        calls.append(command)

    monkeypatch.setattr(bdist_wheel, "run", record_run)
    return calls


def write_complete_plugin_set(command: bdist_wheel) -> None:
    """Write nonempty source and library files for every declared plugin."""

    for name in command.plugin_names:
        (command.plugin_dir / f"{name}.cu").write_bytes(b"source")
        (command.plugin_dir / f"{name}.so").write_bytes(b"\x7fELFplugin")


def test_setup_registers_binary_distribution_commands(
    setup_execution: SetupExecution,
) -> None:
    """Register the custom distribution, wheel, and rejected sdist commands."""

    assert len(setup_execution.setup_calls) == 1
    setup_arguments = setup_execution.setup_calls[0]
    assert setup_arguments["distclass"] is setup_execution.distribution_type
    assert setup_arguments["cmdclass"] == {
        "bdist_wheel": setup_execution.wheel_type,
        "sdist": setup_execution.source_distribution_type,
    }


def test_binary_distribution_marks_wheel_as_platform_specific(
    setup_execution: SetupExecution,
) -> None:
    """Mark the distribution as containing native extension modules."""

    distribution = setup_execution.distribution_type()

    assert distribution.has_ext_modules() is True


def test_source_distribution_is_rejected(setup_execution: SetupExecution) -> None:
    """Direct users to the native wheel pipeline instead of building an sdist."""

    command = setup_execution.source_distribution_type(
        setup_execution.distribution_type()
    )

    with pytest.raises(FileError) as error:
        command.run()

    assert str(error.value) == (
        "fast-gpu-asr does not support source distributions; build a native wheel "
        "with scripts/build_wheel.sh."
    )


@pytest.mark.parametrize(
    "base_tag",
    (
        ("cp312", "cp312", "manylinux_2_27_x86_64"),
        ("cp314", "cp314", "linux_aarch64"),
    ),
)
def test_binary_wheel_uses_python_abi_independent_tag(
    wheel_command: bdist_wheel,
    monkeypatch: pytest.MonkeyPatch,
    base_tag: tuple[str, str, str],
) -> None:
    """Replace Python and ABI tags while preserving setuptools' platform tag."""

    calls: list[bdist_wheel] = []

    def get_base_tag(command: bdist_wheel) -> tuple[str, str, str]:
        calls.append(command)
        return base_tag

    monkeypatch.setattr(bdist_wheel, "get_tag", get_base_tag)

    assert wheel_command.get_tag() == ("py3", "none", base_tag[2])
    assert calls == [wheel_command]


def test_binary_wheel_manifest_matches_plugin_builds(
    setup_execution: SetupExecution,
) -> None:
    """Derive wheel artifact names from the production plugin build manifest."""

    expected_names = {Path(source_name).stem for source_name, _ in PLUGIN_BUILDS}

    assert expected_names
    assert setup_execution.wheel_type.plugin_names == expected_names
    assert setup_execution.wheel_type.plugin_dir == PLUGIN_DIR


def test_binary_wheel_accepts_complete_plugin_set(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
) -> None:
    """Delegate exactly once when every plugin artifact is valid."""

    write_complete_plugin_set(wheel_command)

    wheel_command.run()

    assert base_wheel_runs == [wheel_command]
    for name in wheel_command.plugin_names:
        assert (wheel_command.plugin_dir / f"{name}.cu").read_bytes() == b"source"
        assert (
            wheel_command.plugin_dir / f"{name}.so"
        ).read_bytes() == b"\x7fELFplugin"


@pytest.mark.parametrize(
    "artifact_change",
    ("missing_source", "missing_library", "unexpected_source", "unexpected_library"),
)
def test_binary_wheel_rejects_incomplete_or_stale_plugin_set(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
    artifact_change: str,
) -> None:
    """Reject missing and unexpected artifacts before invoking setuptools."""

    write_complete_plugin_set(wheel_command)
    plugin_name = min(wheel_command.plugin_names)
    if artifact_change == "missing_source":
        affected_name = f"{plugin_name}.cu"
        (wheel_command.plugin_dir / affected_name).unlink()
    elif artifact_change == "missing_library":
        affected_name = f"{plugin_name}.so"
        (wheel_command.plugin_dir / affected_name).unlink()
    elif artifact_change == "unexpected_source":
        affected_name = "stale.cu"
        (wheel_command.plugin_dir / affected_name).write_bytes(b"stale source")
    else:
        affected_name = "stale.so"
        (wheel_command.plugin_dir / affected_name).write_bytes(b"stale library")

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert "do not match the build manifest" in str(error.value)
    assert Path(affected_name).stem in str(error.value)
    assert base_wheel_runs == []


@pytest.mark.parametrize("suffix", (".cu", ".so"))
def test_binary_wheel_rejects_empty_plugin_artifact(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
    suffix: str,
) -> None:
    """Reject empty source and library artifacts before invoking setuptools."""

    write_complete_plugin_set(wheel_command)
    artifact_name = min(wheel_command.plugin_names) + suffix
    (wheel_command.plugin_dir / artifact_name).write_bytes(b"")

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert "plugin artifacts must not be empty" in str(error.value)
    assert artifact_name in str(error.value)
    assert base_wheel_runs == []


@pytest.mark.parametrize("suffix", (".cu", ".so"))
@pytest.mark.parametrize(
    "artifact_kind",
    ("directory", "broken_symlink", "valid_symlink", "fifo"),
)
def test_binary_wheel_rejects_nonregular_plugin_artifact(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
    tmp_path: Path,
    suffix: str,
    artifact_kind: str,
) -> None:
    """Reject directories, symlinks, and special files before wheel creation."""

    write_complete_plugin_set(wheel_command)
    artifact_name = min(wheel_command.plugin_names) + suffix
    artifact_path = wheel_command.plugin_dir / artifact_name
    artifact_path.unlink()
    if artifact_kind == "directory":
        artifact_path.mkdir()
    elif artifact_kind == "broken_symlink":
        artifact_path.symlink_to(tmp_path / "missing-artifact")
    elif artifact_kind == "valid_symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(b"external")
        artifact_path.symlink_to(target)
    else:
        os.mkfifo(artifact_path)

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert "regular, non-symlink files" in str(error.value)
    assert artifact_name in str(error.value)
    assert base_wheel_runs == []


def test_binary_wheel_rejects_empty_plugin_set(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
) -> None:
    """Reject a wheel build when every native artifact is absent."""

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert "do not match the build manifest" in str(error.value)
    assert base_wheel_runs == []
