#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for platform-specific wheel metadata and artifact validation."""

import builtins
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
import zipfile
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path
from typing import Any

import pytest
import setuptools
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from setuptools import Distribution
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist
from setuptools.errors import FileError

from fast_gpu_asr.tensorrt_plugins.constants import PLUGIN_BUILDS

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
SETUP_PATH = REPOSITORY_DIR / "setup.py"
PLUGIN_DIR = REPOSITORY_DIR / "src" / "fast_gpu_asr" / "tensorrt_plugins"
PLUGIN_NAMES = tuple(sorted(Path(source_name).stem for source_name, _ in PLUGIN_BUILDS))
BUILD_WHEEL_SCRIPT = """
import importlib
import importlib.abc
import sys
import tomllib
from pathlib import Path

class RuntimeImportBlocker(importlib.abc.MetaPathFinder):
    '''Keep isolated builds independent of the runtime and GPU dependencies.'''

    def find_spec(self, fullname, path=None, target=None):
        '''Reject both direct and importlib-based runtime imports.'''

        if fullname.split('.')[0] in ('fast_gpu_asr', 'torch', 'cupy', 'tensorrt'):
            raise AssertionError(f'Wheel build imported runtime dependency {fullname}')
        return None

sys.meta_path.insert(0, RuntimeImportBlocker())

build_system = tomllib.loads(
    Path("pyproject.toml").read_text(encoding="utf8")
)["build-system"]
backend_module_name, separator, backend_object_path = build_system[
    "build-backend"
].partition(":")
backend = importlib.import_module(backend_module_name)
if separator:
    for attribute in backend_object_path.split("."):
        backend = getattr(backend, attribute)
backend.build_wheel(sys.argv[1])
"""


@pytest.fixture
def setup_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture setup registration without importing the runtime package.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Isolate the setup callback and runtime-import guard.

    Returns
    -------
    dict[str, Any]
        Keyword arguments supplied to the single ``setuptools.setup`` call.
    """

    setup_calls: list[dict[str, Any]] = []
    original_import = builtins.__import__

    def reject_runtime_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Reject runtime imports while delegating build-tool imports unchanged."""

        if name == "fast_gpu_asr" or name.startswith("fast_gpu_asr."):
            raise AssertionError(f"setup.py imported runtime package {name!r}.")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(setuptools, "setup", lambda **kwargs: setup_calls.append(kwargs))
        patch.setattr(builtins, "__import__", reject_runtime_import)
        runpy.run_path(str(SETUP_PATH), run_name="fast_gpu_asr_test_setup")

    assert len(setup_calls) == 1
    return setup_calls[0]


@pytest.fixture
def wheel_command(setup_options: dict[str, Any], tmp_path: Path) -> bdist_wheel:
    """Create a wheel command with complete, nonempty plugin artifacts.

    Parameters
    ----------
    setup_options : dict[str, Any]
        Registered wheel-command and distribution classes.
    tmp_path : Path
        Directory receiving placeholder sources and libraries.

    Returns
    -------
    bdist_wheel
        Command configured to validate the temporary artifacts. Library bytes
        are opaque placeholders, not executable CUDA shared objects.
    """

    command = setup_options["cmdclass"]["bdist_wheel"](setup_options["distclass"]())
    command.plugin_dir = tmp_path
    for name in PLUGIN_NAMES:
        (tmp_path / f"{name}.cu").write_bytes(b"source")
        (tmp_path / f"{name}.so").write_bytes(b"plugin")
    return command


@pytest.fixture
def base_wheel_runs(monkeypatch: pytest.MonkeyPatch) -> list[bdist_wheel]:
    """Record delegation to setuptools' base wheel command.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replace the base command's build operation for this test.

    Returns
    -------
    list[bdist_wheel]
        Commands passed to the base implementation, in call order.
    """

    calls: list[bdist_wheel] = []

    monkeypatch.setattr(bdist_wheel, "run", lambda command: calls.append(command))
    return calls


def create_wheel_test_project(tmp_path: Path) -> Path:
    """Copy packaging inputs and add distinct placeholder plugin libraries.

    Parameters
    ----------
    tmp_path : Path
        Parent directory for the isolated project copy.

    Returns
    -------
    Path
        Project directory containing real package sources and metadata, without
        generated artifacts. Placeholder libraries test packaging, not loading.
    """

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    for name in ("LICENSE", "NOTICE", "README.md", "pyproject.toml", "setup.py"):
        shutil.copy2(REPOSITORY_DIR / name, project_dir / name)
    source_dir = project_dir / "src"
    source_dir.mkdir()
    shutil.copytree(
        REPOSITORY_DIR / "src" / "fast_gpu_asr",
        source_dir / "fast_gpu_asr",
        ignore=shutil.ignore_patterns(
            "*.so", "*.pyc", "*.pyo", "__pycache__", "*.egg-info", ".plugin-build-*"
        ),
    )
    plugin_dir = project_dir / "src" / "fast_gpu_asr" / "tensorrt_plugins"
    for plugin_name in PLUGIN_NAMES:
        (plugin_dir / f"{plugin_name}.so").write_bytes(plugin_name.encode("ascii"))
    return project_dir


def run_wheel_build(
    project_dir: Path, wheel_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Build one test wheel through the configured PEP 517 backend.

    Parameters
    ----------
    project_dir : Path
        Isolated copy of the project containing placeholder plugin libraries.
    wheel_dir : Path
        New destination directory for the wheel.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Backend exit status and captured output, including expected failures.

    Notes
    -----
    The fresh interpreter uses installed build tools without downloading any
    dependencies. A finder rejects runtime-package and GPU-library imports.
    """

    wheel_dir.mkdir()
    return subprocess.run(
        (sys.executable, "-I", "-c", BUILD_WHEEL_SCRIPT, str(wheel_dir)),
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_setup_registers_binary_distribution_commands(
    setup_options: dict[str, Any],
) -> None:
    assert set(setup_options) == {"cmdclass", "distclass"}
    assert issubclass(setup_options["distclass"], Distribution)
    assert setup_options["distclass"]().has_ext_modules() is True
    commands = setup_options["cmdclass"]
    assert set(commands) == {"bdist_wheel", "sdist"}
    assert issubclass(commands["bdist_wheel"], bdist_wheel)
    assert issubclass(commands["sdist"], sdist)
    assert commands["bdist_wheel"].plugin_names == set(PLUGIN_NAMES)
    assert commands["bdist_wheel"].plugin_dir == PLUGIN_DIR


def test_source_distribution_is_rejected(
    setup_options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    command = setup_options["cmdclass"]["sdist"](setup_options["distclass"]())
    monkeypatch.setattr(
        sdist, "run", lambda _: pytest.fail("sdist must not invoke the base command")
    )

    with pytest.raises(FileError) as error:
        command.run()

    assert str(error.value) == (
        "fast-gpu-asr does not support source distributions; build a native wheel "
        "with scripts/build_wheel.sh."
    )


@pytest.mark.parametrize(
    "base_tag",
    (("cp312", "cp312", "manylinux_2_27_x86_64"), ("cp314", "cp314", "linux_aarch64")),
    ids=("manylinux-x86_64", "linux-aarch64"),
)
def test_binary_wheel_uses_python_abi_independent_tag(
    wheel_command: bdist_wheel,
    monkeypatch: pytest.MonkeyPatch,
    base_tag: tuple[str, str, str],
) -> None:
    def get_base_tag(command: bdist_wheel) -> tuple[str, str, str]:
        """Supply a platform tag while checking which command delegates to it."""

        assert command is wheel_command
        return base_tag

    monkeypatch.setattr(bdist_wheel, "get_tag", get_base_tag)

    assert wheel_command.get_tag() == ("py3", "none", base_tag[2])


def test_real_wheel_contains_validated_artifacts_and_binary_metadata(
    tmp_path: Path,
) -> None:
    project_dir = create_wheel_test_project(tmp_path)
    source_dir = project_dir / "src"
    expected_files = {
        path.relative_to(source_dir).as_posix(): path.read_bytes()
        for path in (source_dir / "fast_gpu_asr").rglob("*")
        if path.is_file()
    }
    for name in ("__pycache__/stale.pyc", "stale.pyo", "stale.egg-info/PKG-INFO"):
        path = source_dir / "fast_gpu_asr" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale build artifact")
    project = tomllib.loads(
        (project_dir / "pyproject.toml").read_text(encoding="utf8")
    )["project"]
    wheel_dir = tmp_path / "wheel"
    result = run_wheel_build(project_dir, wheel_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = wheel_dir.glob("*.whl")
    with zipfile.ZipFile(wheel_path) as archive:
        archive_names = archive.namelist()
        assert len(archive_names) == len(set(archive_names))
        actual_files = {
            name: archive.read(name)
            for name in archive_names
            if name.startswith("fast_gpu_asr/") or name.endswith((".cu", ".h", ".so"))
        }
        assert actual_files == expected_files
        assert not any(
            "__pycache__/" in name
            or ".egg-info/" in name
            or name.endswith((".pyc", ".pyo"))
            for name in archive_names
        )
        (dist_info,) = {
            name.split("/")[0]
            for name in archive_names
            if name.split("/")[0].endswith(".dist-info")
        }
        wheel_metadata = Parser().parsestr(
            archive.read(f"{dist_info}/WHEEL").decode("utf8")
        )
        project_metadata = Parser().parsestr(
            archive.read(f"{dist_info}/METADATA").decode("utf8")
        )
        entry_points = ConfigParser(interpolation=None)
        entry_points.optionxform = str
        entry_points.read_string(
            archive.read(f"{dist_info}/entry_points.txt").decode("utf8")
        )
        assert set(project_metadata.get_all("License-File", [])) == {
            "LICENSE",
            "NOTICE",
        }
        for name in ("LICENSE", "NOTICE"):
            assert (
                archive.read(f"{dist_info}/licenses/{name}")
                == (project_dir / name).read_bytes()
            )

    distribution_name, version, build_tag, tags = parse_wheel_filename(wheel_path.name)
    assert distribution_name == canonicalize_name(project["name"])
    assert str(version) == project["version"]
    assert build_tag == ()
    assert {tag.interpreter for tag in tags} == {"py3"}
    assert {tag.abi for tag in tags} == {"none"}
    assert all(tag.platform != "any" for tag in tags)
    assert wheel_metadata["Root-Is-Purelib"] == "false"
    assert sorted(wheel_metadata.get_all("Tag", [])) == sorted(str(tag) for tag in tags)
    assert project_metadata["Name"] == project["name"]
    assert project_metadata["Version"] == project["version"]
    assert project_metadata["Summary"] == project["description"]
    assert project_metadata["License-Expression"] == project["license"]
    assert sorted(project_metadata.get_all("Classifier", [])) == sorted(
        project["classifiers"]
    )
    assert sorted(project_metadata.get_all("Project-URL", [])) == sorted(
        f"{name}, {url}" for name, url in project["urls"].items()
    )
    metadata_requirements = [
        Requirement(requirement)
        for requirement in project_metadata.get_all("Requires-Dist", [])
    ]
    expected_requirements = {
        Requirement(requirement) for requirement in project["dependencies"]
    }
    for extra, requirements in project["optional-dependencies"].items():
        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            extra_marker = f'extra == "{extra}"'
            requirement.marker = Marker(
                f"({requirement.marker}) and {extra_marker}"
                if requirement.marker is not None
                else extra_marker
            )
            expected_requirements.add(requirement)
    assert sorted(metadata_requirements, key=str) == sorted(
        expected_requirements, key=str
    )
    assert sorted(project_metadata.get_all("Provides-Extra", [])) == sorted(
        project["optional-dependencies"]
    )
    assert SpecifierSet(project_metadata["Requires-Python"]) == SpecifierSet(
        project["requires-python"]
    )
    assert project_metadata.get_payload() == (project_dir / "README.md").read_text(
        encoding="utf8"
    )
    assert entry_points.sections() == ["console_scripts"]
    assert dict(entry_points["console_scripts"]) == project["scripts"]


def test_real_wheel_build_runs_plugin_artifact_validation(tmp_path: Path) -> None:
    project_dir = create_wheel_test_project(tmp_path)
    missing_library = PLUGIN_NAMES[-1] + ".so"
    (
        project_dir / "src" / "fast_gpu_asr" / "tensorrt_plugins" / missing_library
    ).unlink()
    wheel_dir = tmp_path / "wheel"

    result = run_wheel_build(project_dir, wheel_dir)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "plugin artifacts do not match the build manifest" in output
    assert not list(wheel_dir.glob("*.whl"))


def test_binary_wheel_accepts_complete_plugin_set(
    wheel_command: bdist_wheel, base_wheel_runs: list[bdist_wheel]
) -> None:
    wheel_command.run()

    assert base_wheel_runs == [wheel_command]


def test_binary_wheel_propagates_base_build_failure(
    wheel_command: bdist_wheel, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("wheel build failed")

    def fail_build(command: bdist_wheel) -> None:
        """Raise the original build failure after checking delegation."""

        assert command is wheel_command
        raise failure

    monkeypatch.setattr(bdist_wheel, "run", fail_build)

    with pytest.raises(RuntimeError) as error:
        wheel_command.run()

    assert error.value is failure


@pytest.mark.parametrize("change", ("missing", "extra", "substituted"))
@pytest.mark.parametrize("suffix", (".cu", ".so"))
def test_binary_wheel_rejects_manifest_mismatch(
    wheel_command: bdist_wheel,
    base_wheel_runs: list[bdist_wheel],
    change: str,
    suffix: str,
) -> None:
    source_names = set(PLUGIN_NAMES)
    library_names = set(PLUGIN_NAMES)
    affected_names = source_names if suffix == ".cu" else library_names
    if change in ("missing", "substituted"):
        (wheel_command.plugin_dir / f"{PLUGIN_NAMES[-1]}{suffix}").unlink()
        affected_names.remove(PLUGIN_NAMES[-1])
    if change in ("extra", "substituted"):
        (wheel_command.plugin_dir / f"stale{suffix}").write_bytes(b"stale")
        affected_names.add("stale")

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert str(error.value) == (
        "TensorRT plugin artifacts do not match the build manifest: "
        f"expected={list(PLUGIN_NAMES)}, sources={sorted(source_names)}, "
        f"libraries={sorted(library_names)}."
    )
    assert base_wheel_runs == []


@pytest.mark.parametrize(
    "artifact_kind", ("directory", "broken_symlink", "valid_symlink", "fifo")
)
def test_binary_wheel_reports_nonregular_artifacts(
    wheel_command: bdist_wheel, base_wheel_runs: list[bdist_wheel], artifact_kind: str
) -> None:
    artifact_names = [f"{PLUGIN_NAMES[-1]}.so", f"{PLUGIN_NAMES[0]}.cu"]
    for name in artifact_names:
        path = wheel_command.plugin_dir / name
        path.unlink()
        if artifact_kind == "directory":
            path.mkdir()
        elif artifact_kind == "fifo":
            os.mkfifo(path)
        else:
            target = path.with_suffix(".target")
            if artifact_kind == "valid_symlink":
                target.write_bytes(b"plugin")
            path.symlink_to(target)

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert str(error.value) == (
        "TensorRT plugin artifacts must be regular, non-symlink files: "
        f"{sorted(artifact_names)}."
    )
    assert base_wheel_runs == []


def test_binary_wheel_reports_every_empty_artifact(
    wheel_command: bdist_wheel, base_wheel_runs: list[bdist_wheel]
) -> None:
    artifact_names = [f"{PLUGIN_NAMES[-1]}.cu", f"{PLUGIN_NAMES[0]}.so"]
    for artifact_name in artifact_names:
        (wheel_command.plugin_dir / artifact_name).write_bytes(b"")

    with pytest.raises(FileError) as error:
        wheel_command.run()

    assert str(error.value) == (
        f"TensorRT plugin artifacts must not be empty: {sorted(artifact_names)}."
    )
    assert base_wheel_runs == []


def test_binary_wheel_rejects_empty_plugin_set(
    wheel_command: bdist_wheel, base_wheel_runs: list[bdist_wheel]
) -> None:
    wheel_command.plugin_dir = wheel_command.plugin_dir / "empty"
    wheel_command.plugin_dir.mkdir()

    with pytest.raises(FileError, match="do not match the build manifest"):
        wheel_command.run()

    assert base_wheel_runs == []
