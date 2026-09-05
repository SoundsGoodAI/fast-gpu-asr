#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for wheel build orchestration and destination preservation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_wheel.sh"
RAW_WHEEL = "fast_gpu_asr-0-py3-none-linux_x86_64.whl"
REPAIRED_WHEEL = "fast_gpu_asr-0-py3-none-manylinux_2_27_x86_64.whl"
SOURCE_FILES = {
    "LICENSE": "license",
    "NOTICE": "upstream notices",
    "README.md": "readme",
    "pyproject.toml": "metadata",
    "setup.py": "setup",
    "src/fast_gpu_asr/__init__.py": "package",
    "src/fast_gpu_asr/tensorrt_plugins/plugin.cu": "source",
    "src/fast_gpu_asr/tensorrt_plugins/plugin.h": "header",
}


class CommandRecord(TypedDict):
    """One command observed by the fake wheel toolchain."""

    tool: str
    arguments: list[str]
    cwd: str
    source_files: NotRequired[dict[str, str]]


@pytest.fixture
def wheel_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the real build script into a repository with fake build tools.

    Parameters
    ----------
    tmp_path : Path
        Parent directory for the repository and executable command shims.
    monkeypatch : pytest.MonkeyPatch
        Prepend the shim directory to ``PATH`` and clear failure controls.

    Returns
    -------
    Path
        Repository path containing spaces, to exercise shell quoting. Shims
        record commands and create opaque artifacts without compiling CUDA or
        running auditwheel; all other shell operations run normally.
    """

    repository_dir = tmp_path / "repository with spaces"
    scripts_dir = repository_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, scripts_dir / SCRIPT_PATH.name)
    for name, content in {**SOURCE_FILES, "unrelated.txt": "exclude me"}.items():
        path = repository_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    shim = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

tool = Path(sys.argv[0]).name
arguments = sys.argv[1:]
record = {"tool": tool, "arguments": arguments, "cwd": os.getcwd()}
if tool == "uv" and arguments[:2] == ["build", "--wheel"]:
    record["source_files"] = {
        path.relative_to(Path.cwd()).as_posix(): path.read_text(encoding="utf8")
        for path in Path.cwd().rglob("*") if path.is_file()
    }
with Path(os.environ["COMMAND_LOG"]).open("a", encoding="utf8") as log:
    log.write(json.dumps(record) + "\\n")

if tool == "uv":
    if arguments[:1] == ["run"]:
        stage = "plugin"
    elif arguments[:2] == ["build", "--wheel"]:
        stage = "wheel"
    else:
        raise SystemExit(90)
    if os.environ.get("FAIL_STAGE") == stage:
        raise SystemExit({"plugin": 31, "wheel": 32}[stage])
    if stage == "plugin":
        Path("src/fast_gpu_asr/tensorrt_plugins/plugin.so").write_text(
            "compiled plugin", encoding="utf8"
        )
    else:
        output_dir = Path(arguments[arguments.index("--out-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "directory.whl").mkdir()
        (output_dir / "nested").mkdir()
        (output_dir / "nested" / "ignored.whl").write_bytes(b"nested wheel")
        for index in range(int(os.environ.get("RAW_WHEEL_COUNT", "1"))):
            wheel_path = output_dir / (
                f"fast_gpu_asr-{index}-py3-none-linux_x86_64.whl"
            )
            wheel_path.write_bytes(b"raw wheel")
elif tool == "uvx":
    if os.environ.get("FAIL_STAGE") == "repair":
        raise SystemExit(33)
    output_dir = Path(arguments[arguments.index("--wheel-dir") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "directory.whl").mkdir()
    (output_dir / "nested").mkdir()
    (output_dir / "nested" / "ignored.whl").write_bytes(b"nested wheel")
    for index in range(int(os.environ.get("REPAIRED_WHEEL_COUNT", "1"))):
        wheel_path = output_dir / (
            f"fast_gpu_asr-{index}-py3-none-manylinux_2_27_x86_64.whl"
        )
        wheel_path.write_bytes(b"repaired wheel")
elif tool == "mv":
    if os.environ.get("FAIL_STAGE") == "publish":
        raise SystemExit(34)
    if len(arguments) != 3 or arguments[0] != "--":
        raise SystemExit(91)
    os.replace(arguments[1], arguments[2])
else:
    raise SystemExit(92)
"""
    for tool in ("uv", "uvx", "mv"):
        path = bin_dir / tool
        path.write_text(shim, encoding="utf8")
        path.chmod(0o755)
    for variable in ("FAIL_STAGE", "RAW_WHEEL_COUNT", "REPAIRED_WHEEL_COUNT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return repository_dir


@pytest.fixture
def wheel_dir(tmp_path: Path) -> Path:
    """Seed a destination with existing wheels and unrelated files.

    Parameters
    ----------
    tmp_path : Path
        Parent directory for the wheel destination.

    Returns
    -------
    Path
        Destination containing a replaceable wheel, a stale package wheel,
        unrelated files, and an archived wheel that must survive cleanup.
    """

    path = tmp_path / "wheel output"
    (path / "archive").mkdir(parents=True)
    for name, content in {
        REPAIRED_WHEEL: b"known-good",
        "fast_gpu_asr-stale.whl": b"stale",
        "unrelated.whl": b"unrelated",
        "keep.txt": b"keep",
        "archive/fast_gpu_asr-archived.whl": b"archived",
    }.items():
        (path / name).write_bytes(content)
    return path


def read_files(directory: Path) -> dict[str, bytes]:
    """Snapshot file contents to detect lost or changed artifacts.

    Parameters
    ----------
    directory : Path
        Directory whose files are read recursively.

    Returns
    -------
    dict[str, bytes]
        Relative POSIX paths mapped to exact file contents.
    """

    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def run_build_wheel(
    repository_dir: Path,
    arguments: tuple[str, ...] = (),
    **environment_overrides: str,
) -> tuple[subprocess.CompletedProcess[str], list[CommandRecord]]:
    """Run outside the temporary repository and collect shim invocations.

    Parameters
    ----------
    repository_dir : Path
        Repository prepared by ``wheel_project``.
    arguments : tuple[str, ...]
        Arguments forwarded to the shell script without splitting.
    **environment_overrides : str
        Shim controls for stage failures or emitted wheel counts.

    Returns
    -------
    tuple[subprocess.CompletedProcess[str], list[CommandRecord]]
        Exit status and output, plus command records in execution order.
    """

    command_log = repository_dir.parent / "commands.jsonl"
    command_log.unlink(missing_ok=True)
    result = subprocess.run(
        (str(repository_dir / "scripts" / SCRIPT_PATH.name), *arguments),
        capture_output=True,
        cwd=repository_dir.parent,
        env={**os.environ, **environment_overrides, "COMMAND_LOG": str(command_log)},
        text=True,
        timeout=30,
    )
    commands = (
        [
            json.loads(line)
            for line in command_log.read_text(encoding="utf8").splitlines()
        ]
        if command_log.exists()
        else []
    )
    return result, commands


def test_build_wheel_repairs_with_expected_policy_and_exclusions(
    wheel_project: Path, wheel_dir: Path
) -> None:
    expected_files = read_files(wheel_dir)
    expected_files[REPAIRED_WHEEL] = b"repaired wheel"
    del expected_files["fast_gpu_asr-stale.whl"]

    result, commands = run_build_wheel(wheel_project, (str(wheel_dir),))

    assert result.returncode == 0, result.stderr
    assert read_files(wheel_dir) == expected_files
    assert not list(wheel_dir.glob(".build-wheel.*"))
    assert [command["tool"] for command in commands] == ["uv", "uv", "uvx", "mv"]
    plugin, build, repair, publish = commands
    assert plugin["arguments"] == [
        "run",
        "--frozen",
        "python",
        "-m",
        "fast_gpu_asr.tensorrt_plugins.build",
    ]
    raw_dir = Path(build["arguments"][-1])
    build_dir = raw_dir.parent
    repaired_dir = Path(repair["arguments"][7])
    assert build_dir.parent == wheel_dir
    assert build_dir.name.startswith(".build-wheel.")
    assert build["arguments"] == ["build", "--wheel", "--out-dir", str(raw_dir)]
    assert Path(build["cwd"]) == build_dir / "source"
    assert build["source_files"] == {
        **SOURCE_FILES,
        "src/fast_gpu_asr/tensorrt_plugins/plugin.so": "compiled plugin",
    }
    assert repaired_dir.parent == build_dir
    assert repaired_dir != raw_dir
    assert repair["arguments"] == [
        "--from",
        "auditwheel",
        "auditwheel",
        "repair",
        "--plat",
        "manylinux_2_27_x86_64",
        "--wheel-dir",
        str(repaired_dir),
        "--exclude",
        "libnvinfer.so.11",
        "--exclude",
        "libcudart.so.13",
        "--exclude",
        "libcublas.so.13",
        "--exclude",
        "libcublasLt.so.13",
        "--exclude",
        "libcufft.so.12",
        str(raw_dir / RAW_WHEEL),
    ]
    assert publish["arguments"] == [
        "--",
        str(repaired_dir / REPAIRED_WHEEL),
        str(wheel_dir / REPAIRED_WHEEL),
    ]
    assert [command["cwd"] for command in (plugin, repair, publish)] == [
        str(wheel_project)
    ] * 3


@pytest.mark.parametrize(
    ("stage", "exit_code", "expected_tools"),
    (
        ("plugin", 31, ["uv"]),
        ("wheel", 32, ["uv", "uv"]),
        ("repair", 33, ["uv", "uv", "uvx"]),
        ("publish", 34, ["uv", "uv", "uvx", "mv"]),
    ),
)
def test_build_wheel_preserves_destination_when_command_fails(
    wheel_project: Path,
    wheel_dir: Path,
    stage: str,
    exit_code: int,
    expected_tools: list[str],
) -> None:
    original_files = read_files(wheel_dir)

    result, commands = run_build_wheel(
        wheel_project, (str(wheel_dir),), FAIL_STAGE=stage
    )

    assert result.returncode == exit_code, result.stderr
    assert [command["tool"] for command in commands] == expected_tools
    assert read_files(wheel_dir) == original_files
    assert not list(wheel_dir.glob(".build-wheel.*"))


@pytest.mark.parametrize(
    ("stage", "expected_tools"),
    (("raw", ["uv", "uv"]), ("repaired", ["uv", "uv", "uvx"])),
)
@pytest.mark.parametrize("count", (0, 2))
def test_build_wheel_rejects_invalid_artifact_count(
    wheel_project: Path,
    wheel_dir: Path,
    stage: str,
    expected_tools: list[str],
    count: int,
) -> None:
    original_files = read_files(wheel_dir)

    result, commands = run_build_wheel(
        wheel_project, (str(wheel_dir),), **{f"{stage.upper()}_WHEEL_COUNT": str(count)}
    )

    assert result.returncode == 1
    assert result.stderr == f"Expected exactly one {stage} wheel, found {count}.\n"
    assert [command["tool"] for command in commands] == expected_tools
    assert read_files(wheel_dir) == original_files
    assert not list(wheel_dir.glob(".build-wheel.*"))


@pytest.mark.parametrize(
    "arguments", ((), ("wheel output",)), ids=("default", "relative")
)
def test_build_wheel_resolves_output_directory(
    wheel_project: Path, arguments: tuple[str, ...]
) -> None:
    destination = (
        wheel_project.parent / arguments[0] if arguments else wheel_project / "dist"
    )

    result, commands = run_build_wheel(wheel_project, arguments)

    assert result.returncode == 0, result.stderr
    assert read_files(destination) == {REPAIRED_WHEEL: b"repaired wheel"}
    assert commands[0]["cwd"] == str(wheel_project)
    assert not list(destination.glob(".build-wheel.*"))


def test_build_wheel_rejects_output_path_that_is_a_file(wheel_project: Path) -> None:
    destination = wheel_project.parent / "dist"
    destination.write_bytes(b"keep")

    result, commands = run_build_wheel(wheel_project, (str(destination),))

    assert result.returncode != 0
    assert commands == []
    assert destination.read_bytes() == b"keep"


def test_build_wheel_preserves_destination_when_source_copy_fails(
    wheel_project: Path, wheel_dir: Path
) -> None:
    (wheel_project / "README.md").unlink()
    original_files = read_files(wheel_dir)

    result, commands = run_build_wheel(wheel_project, (str(wheel_dir),))

    assert result.returncode != 0
    assert "README.md" in result.stderr
    assert [command["tool"] for command in commands] == ["uv"]
    assert read_files(wheel_dir) == original_files
    assert not list(wheel_dir.glob(".build-wheel.*"))


@pytest.mark.parametrize("arguments", (("",), ("one", "two")))
def test_build_wheel_rejects_invalid_arguments(
    wheel_project: Path, arguments: tuple[str, ...]
) -> None:
    original_files = read_files(wheel_project.parent)

    result, commands = run_build_wheel(wheel_project, arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "Usage: build_wheel.sh [wheel-directory]\n"
    assert commands == []
    assert read_files(wheel_project.parent) == original_files
