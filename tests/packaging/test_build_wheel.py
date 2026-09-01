#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for transactional binary-wheel build orchestration."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NotRequired, TypedDict

import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_DIR / "scripts" / "build_wheel.sh"
COMMAND_TIMEOUT_SECONDS = 30


class CommandRecord(TypedDict):
    """One external command observed by the fake wheel toolchain."""

    tool: str
    arguments: list[str]
    cwd: str
    cwd_entries: NotRequired[list[str]]
    cwd_files: NotRequired[list[str]]


def write_executable(path: Path, source: str) -> None:
    """Write one executable test command."""

    path.write_text(source, encoding="utf8")
    path.chmod(0o755)


@pytest.fixture
def fake_wheel_toolchain(tmp_path: Path) -> tuple[Path, Path]:
    """Create configurable ``uv``, ``uvx``, and ``mv`` command shims."""

    bin_dir = tmp_path / "bin"
    command_log = tmp_path / "commands.jsonl"
    bin_dir.mkdir()
    shim = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

tool = Path(sys.argv[0]).name
arguments = sys.argv[1:]
record = {"tool": tool, "arguments": arguments, "cwd": os.getcwd()}
if tool == "uv" and arguments[:2] == ["build", "--wheel"]:
    record["cwd_entries"] = sorted(path.name for path in Path.cwd().iterdir())
    record["cwd_files"] = sorted(
        str(path.relative_to(Path.cwd()))
        for path in Path.cwd().rglob("*")
        if path.is_file()
    )
with Path(os.environ["COMMAND_LOG"]).open("a", encoding="utf8") as log:
    log.write(json.dumps(record))
    log.write("\\n")

if tool == "uv":
    if arguments[:1] == ["run"]:
        stage = "plugin"
    elif arguments[:2] == ["build", "--wheel"]:
        stage = "wheel"
    else:
        raise SystemExit(90)
    if os.environ.get("FAIL_STAGE") == stage:
        raise SystemExit({"plugin": 31, "wheel": 32}[stage])
    if stage == "wheel":
        output_dir = Path(arguments[arguments.index("--out-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
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
    write_executable(bin_dir / "uv", shim)
    write_executable(bin_dir / "uvx", shim)
    write_executable(bin_dir / "mv", shim)
    return bin_dir, command_log


def run_build_wheel(
    wheel_dir: Path,
    fake_wheel_toolchain: tuple[Path, Path],
    *,
    script_path: Path = SCRIPT_PATH,
    script_arguments: tuple[str, ...] | None = None,
    working_directory: Path | None = None,
    **environment_overrides: str,
) -> tuple[subprocess.CompletedProcess[str], list[CommandRecord]]:
    """Run the wheel script with the fake commands and return its command log."""

    bin_dir, command_log = fake_wheel_toolchain
    command_log.unlink(missing_ok=True)
    environment = os.environ.copy()
    for variable in ("FAIL_STAGE", "RAW_WHEEL_COUNT", "REPAIRED_WHEEL_COUNT"):
        environment.pop(variable, None)
    environment.update(environment_overrides)
    environment["COMMAND_LOG"] = str(command_log)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    if script_arguments is None:
        script_arguments = (str(wheel_dir),)
    result = subprocess.run(
        (str(script_path), *script_arguments),
        check=False,
        capture_output=True,
        cwd=working_directory,
        env=environment,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
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


def create_minimal_repository(path: Path, include_readme: bool = True) -> Path:
    """Create the source files required by a copied wheel-build script."""

    scripts_dir = path / "scripts"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / SCRIPT_PATH.name
    shutil.copy2(SCRIPT_PATH, script_path)
    for name in ("LICENSE", "pyproject.toml", "setup.py"):
        (path / name).write_text(name, encoding="utf8")
    if include_readme:
        (path / "README.md").write_text("README", encoding="utf8")
    (path / "src").mkdir()
    return script_path


def test_build_wheel_repairs_with_expected_policy_and_exclusions(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
) -> None:
    """Publish one wheel only after applying the complete repair policy."""

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    published_wheel = wheel_dir / "fast_gpu_asr-0-py3-none-manylinux_2_27_x86_64.whl"
    published_wheel.write_bytes(b"known-good")
    stale_wheel = wheel_dir / "fast_gpu_asr-stale.whl"
    stale_wheel.write_bytes(b"stale")
    (wheel_dir / "unrelated.whl").write_bytes(b"unrelated")
    (wheel_dir / "keep.txt").write_text("keep", encoding="utf8")
    archive_dir = wheel_dir / "archive"
    archive_dir.mkdir()
    archived_wheel = archive_dir / "fast_gpu_asr-archived.whl"
    archived_wheel.write_bytes(b"archived")

    result, commands = run_build_wheel(wheel_dir, fake_wheel_toolchain)

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in wheel_dir.glob("*.whl")) == [
        "fast_gpu_asr-0-py3-none-manylinux_2_27_x86_64.whl",
        "unrelated.whl",
    ]
    assert published_wheel.read_bytes() == b"repaired wheel"
    assert not stale_wheel.exists()
    assert (wheel_dir / "keep.txt").read_text(encoding="utf8") == "keep"
    assert archived_wheel.read_bytes() == b"archived"
    assert not list(wheel_dir.glob(".build-wheel.*"))
    assert [command["tool"] for command in commands] == ["uv", "uv", "uvx", "mv"]
    assert commands[0]["arguments"] == [
        "run",
        "--frozen",
        "python",
        "-m",
        "fast_gpu_asr.tensorrt_plugins.build",
    ]
    raw_wheel_dir = Path(commands[1]["arguments"][-1])
    assert commands[1]["arguments"] == [
        "build",
        "--wheel",
        "--out-dir",
        str(raw_wheel_dir),
    ]
    source_dir = Path(commands[1]["cwd"])
    assert source_dir == raw_wheel_dir.parent / "source"
    assert commands[1]["cwd_entries"] == [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "src",
    ]
    assert {
        "src/fast_gpu_asr/__init__.py",
        "src/fast_gpu_asr/tensorrt_plugins/constants.py",
    } <= set(commands[1]["cwd_files"])
    repair_arguments = commands[2]["arguments"]
    assert repair_arguments[:3] == ["--from", "auditwheel", "auditwheel"]
    assert repair_arguments[3:6] == [
        "repair",
        "--plat",
        "manylinux_2_27_x86_64",
    ]
    assert repair_arguments[6] == "--wheel-dir"
    assert Path(repair_arguments[7]).parent == raw_wheel_dir.parent
    assert repair_arguments[8:-1] == [
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
    ]
    assert Path(repair_arguments[-1]).parent == raw_wheel_dir
    assert Path(repair_arguments[-1]).name == (
        "fast_gpu_asr-0-py3-none-linux_x86_64.whl"
    )
    assert commands[0]["cwd"] == str(REPOSITORY_DIR)
    assert commands[2]["cwd"] == str(REPOSITORY_DIR)
    repaired_wheel_dir = Path(repair_arguments[7])
    assert commands[3]["arguments"] == [
        "--",
        str(repaired_wheel_dir / published_wheel.name),
        str(published_wheel),
    ]
    assert commands[3]["cwd"] == str(REPOSITORY_DIR)


@pytest.mark.parametrize(
    ("stage", "exit_code", "expected_tools"),
    (
        ("plugin", 31, ["uv"]),
        ("wheel", 32, ["uv", "uv"]),
        ("repair", 33, ["uv", "uv", "uvx"]),
    ),
)
def test_build_wheel_preserves_destination_when_command_fails(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
    stage: str,
    exit_code: int,
    expected_tools: list[str],
) -> None:
    """Keep a known-good artifact when any fallible build stage fails."""

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    old_wheel = wheel_dir / "fast_gpu_asr-known-good.whl"
    old_wheel.write_bytes(b"known-good")

    result, commands = run_build_wheel(
        wheel_dir, fake_wheel_toolchain, FAIL_STAGE=stage
    )

    assert result.returncode == exit_code
    assert [command["tool"] for command in commands] == expected_tools
    assert old_wheel.read_bytes() == b"known-good"
    assert [path.name for path in wheel_dir.glob("*.whl")] == [old_wheel.name]
    assert not list(wheel_dir.glob(".build-wheel.*"))


def test_build_wheel_preserves_destination_when_publish_fails(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
) -> None:
    """Keep existing package and unrelated wheels when atomic publication fails."""

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    old_wheel = wheel_dir / "fast_gpu_asr-known-good.whl"
    unrelated_wheel = wheel_dir / "another_project.whl"
    old_wheel.write_bytes(b"known-good")
    unrelated_wheel.write_bytes(b"unrelated")

    result, commands = run_build_wheel(
        wheel_dir, fake_wheel_toolchain, FAIL_STAGE="publish"
    )

    assert result.returncode == 34
    assert [command["tool"] for command in commands] == ["uv", "uv", "uvx", "mv"]
    assert old_wheel.read_bytes() == b"known-good"
    assert unrelated_wheel.read_bytes() == b"unrelated"
    assert sorted(path.name for path in wheel_dir.glob("*.whl")) == [
        unrelated_wheel.name,
        old_wheel.name,
    ]
    assert not list(wheel_dir.glob(".build-wheel.*"))


@pytest.mark.parametrize(
    ("variable", "count", "message", "expected_tools"),
    (
        (
            "RAW_WHEEL_COUNT",
            "0",
            "Expected exactly one raw wheel, found 0.",
            ["uv", "uv"],
        ),
        (
            "RAW_WHEEL_COUNT",
            "2",
            "Expected exactly one raw wheel, found 2.",
            ["uv", "uv"],
        ),
        (
            "REPAIRED_WHEEL_COUNT",
            "0",
            "Expected exactly one repaired wheel, found 0.",
            ["uv", "uv", "uvx"],
        ),
        (
            "REPAIRED_WHEEL_COUNT",
            "2",
            "Expected exactly one repaired wheel, found 2.",
            ["uv", "uv", "uvx"],
        ),
    ),
)
def test_build_wheel_rejects_ambiguous_artifact_count(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
    variable: str,
    count: str,
    message: str,
    expected_tools: list[str],
) -> None:
    """Reject missing or ambiguous artifacts without replacing the destination."""

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    old_wheel = wheel_dir / "fast_gpu_asr-known-good.whl"
    old_wheel.write_bytes(b"known-good")

    result, commands = run_build_wheel(
        wheel_dir, fake_wheel_toolchain, **{variable: count}
    )

    assert result.returncode == 1
    assert result.stderr == message + "\n"
    assert [command["tool"] for command in commands] == expected_tools
    assert old_wheel.read_bytes() == b"known-good"
    assert [path.name for path in wheel_dir.glob("*.whl")] == [old_wheel.name]
    assert not list(wheel_dir.glob(".build-wheel.*"))


def test_build_wheel_uses_default_output_directory(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
) -> None:
    """Publish to the repository's ``dist`` directory when no path is given."""

    repository_dir = tmp_path / "repository"
    script_path = create_minimal_repository(repository_dir)
    wheel_dir = repository_dir / "dist"

    result, commands = run_build_wheel(
        wheel_dir,
        fake_wheel_toolchain,
        script_path=script_path,
        script_arguments=(),
        working_directory=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert [path.name for path in wheel_dir.glob("*.whl")] == [
        "fast_gpu_asr-0-py3-none-manylinux_2_27_x86_64.whl"
    ]
    assert commands[0]["cwd"] == str(repository_dir)
    assert not list(wheel_dir.glob(".build-wheel.*"))


def test_build_wheel_resolves_relative_output_from_caller_directory(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
) -> None:
    """Resolve a relative output path from the caller and create it safely."""

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    wheel_dir = caller_dir / "wheel output"

    result, _ = run_build_wheel(
        wheel_dir,
        fake_wheel_toolchain,
        script_arguments=(wheel_dir.name,),
        working_directory=caller_dir,
    )

    assert result.returncode == 0, result.stderr
    assert [path.name for path in wheel_dir.glob("*.whl")] == [
        "fast_gpu_asr-0-py3-none-manylinux_2_27_x86_64.whl"
    ]
    assert not list(wheel_dir.glob(".build-wheel.*"))


def test_build_wheel_preserves_destination_when_source_copy_fails(
    tmp_path: Path,
    fake_wheel_toolchain: tuple[Path, Path],
) -> None:
    """Clean temporary files without replacing a wheel after a copy failure."""

    repository_dir = tmp_path / "repository"
    script_path = create_minimal_repository(repository_dir, include_readme=False)
    wheel_dir = repository_dir / "dist"
    wheel_dir.mkdir()
    old_wheel = wheel_dir / "fast_gpu_asr-known-good.whl"
    old_wheel.write_bytes(b"known-good")

    result, commands = run_build_wheel(
        wheel_dir,
        fake_wheel_toolchain,
        script_path=script_path,
        script_arguments=(),
    )

    assert result.returncode != 0
    assert "README.md" in result.stderr
    assert [command["tool"] for command in commands] == ["uv"]
    assert old_wheel.read_bytes() == b"known-good"
    assert [path.name for path in wheel_dir.glob("*.whl")] == [old_wheel.name]
    assert not list(wheel_dir.glob(".build-wheel.*"))


@pytest.mark.parametrize("arguments", (("",), ("one", "two")))
def test_build_wheel_rejects_invalid_arguments(arguments: tuple[str, ...]) -> None:
    """Reject empty or ambiguous output arguments before touching the filesystem."""

    result = subprocess.run(
        (str(SCRIPT_PATH), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "Usage: build_wheel.sh [wheel-directory]\n"
