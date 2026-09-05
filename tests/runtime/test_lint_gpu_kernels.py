#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for native and embedded CUDA formatting with the real formatter."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from fast_gpu_asr.decoder import lint_gpu_kernels

MODULE_PATH = Path(lint_gpu_kernels.__file__).resolve()
KERNEL = "__global__ void run(float* x){if(x[0]>0){x[0]*=2;}}\n"
FORMATTED_KERNEL = """__global__ void run(float* x)
{
    if (x[0] > 0)
    {
        x[0] *= 2;
    }
}
"""


@pytest.fixture
def cuda_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated source tree using the repository's CUDA style.

    Parameters
    ----------
    tmp_path : Path
        Temporary project root.
    monkeypatch : pytest.MonkeyPatch
        Redirects the command's source discovery and style lookup to this root.

    Returns
    -------
    Path
        Source directory containing all supported native file extensions,
        nested sources, and embedded kernels. An unrelated Python module must
        remain untouched. All paths and command-line overrides are test-local.
    """

    (tmp_path / ".clang-format").write_text(
        (MODULE_PATH.parents[3] / ".clang-format").read_text()
    )
    source_dir = tmp_path / "src" / "fast_gpu_asr"
    (source_dir / "decoder").mkdir(parents=True)
    (source_dir / "kernel.cu").write_text(KERNEL)
    (source_dir / "helper.h").write_text("int value(){return 1;}\n")
    (source_dir / "nested").mkdir()
    (source_dir / "not-a-source.cu").mkdir()
    for suffix in (".cpp", ".cuh", ".hpp"):
        (source_dir / "nested" / f"kernel{suffix}").write_text(KERNEL)
    (source_dir / "decoder" / "gpu_kernels.py").write_text(
        'kernel = r"""' + KERNEL + '"""\n'
    )
    (source_dir / "other.py").write_text('text = r"""' + KERNEL + '"""\n')
    monkeypatch.setattr(
        lint_gpu_kernels,
        "__file__",
        str(source_dir / "decoder" / "lint_gpu_kernels.py"),
    )
    return source_dir


def test_native_formatting_compacts_functions_and_preserves_control_flow() -> None:
    includes = "#include <z.h>\n#include <a.h>\n\n"
    source = includes + "__device__ float square(float x)\n{\nreturn x*x;\n}\n" + KERNEL

    formatted = lint_gpu_kernels.format_cuda(source, Path("kernel.cu"))

    assert formatted == (
        includes
        + "__device__ float square(float x) { return x * x; }\n"
        + FORMATTED_KERNEL
    )
    assert lint_gpu_kernels.format_cuda(formatted, Path("kernel.cu")) == formatted


@pytest.mark.parametrize("indent", ("", "    "), ids=("native", "embedded"))
def test_column_limit_accepts_exactly_100_characters(indent: str) -> None:
    source = "int " + "x" * (95 - len(indent)) + ";\n"
    assert lint_gpu_kernels.format_cuda(source, Path("kernel.cu"), indent) == (
        indent + source
    )
    with pytest.raises(ValueError, match="exceeds 100 characters"):
        lint_gpu_kernels.format_cuda(
            source.replace(";", "x;"), Path("kernel.cu"), indent
        )


def test_column_limit_must_leave_space_after_indentation(cuda_project: Path) -> None:
    (cuda_project.parents[1] / ".clang-format").write_text("ColumnLimit: 4\n")

    with pytest.raises(
        ValueError, match="ColumnLimit must be an integer greater than 4"
    ):
        lint_gpu_kernels.format_cuda(KERNEL, Path("kernel.cu"), "    ")


def test_unicode_separators_do_not_bypass_the_column_limit() -> None:
    source = 'const char* label = "' + "a" * 48 + "\u2028" + "b" * 48 + '";\n'
    with pytest.raises(ValueError, match="exceeds 100 characters"):
        lint_gpu_kernels.format_cuda(source, Path("kernel.cu"))


def test_indentation_preserves_unicode_in_cpp_strings() -> None:
    source = '__device__ const char* label() { return "a\u0085\u2028\u2029b"; }\n'
    assert lint_gpu_kernels.format_cuda(source, Path("kernel.cu"), "    ") == (
        "    " + source
    )


@pytest.mark.parametrize("quote", ('"""', "'''"))
def test_embedded_kernels_preserve_python_and_literal_contents(quote: str) -> None:
    prefix = (
        '"""__global__ documentation"""\n'
        f"# ignored = r{quote}__global__ void ignored(){{}}{quote}\n"
        'metadata = "\u03b1\u0085\u2028\u2029"; first = '
    )
    suffix = (
        'ordinary = "__device__ example"\npattern = r"a\\s+b"\n'
        + f"notes = r{quote}not a CUDA source{quote}\n"
    )
    label = '__device__ const char* label(){return "two words\\n";}'
    source = (
        prefix
        + f"r{quote}{KERNEL}{quote}\n"
        + f"second = R{quote}\n    {label}\n{quote}  # keep this comment\n"
        + suffix
    )

    formatted = lint_gpu_kernels.format_python_kernels(source, Path("gpu_kernels.py"))

    expected = (
        prefix
        + f"r{quote}\n"
        + textwrap.indent(FORMATTED_KERNEL, "    ")
        + f"{quote}\nsecond = R{quote}\n"
        + '    __device__ const char* label() { return "two words\\n"; }\n'
        + f"{quote}  # keep this comment\n"
        + suffix
    )
    assert formatted == expected
    assert (
        lint_gpu_kernels.format_python_kernels(formatted, Path("gpu_kernels.py"))
        == formatted
    )


def test_embedded_column_limit_includes_python_indentation() -> None:
    source = (
        'kernel = r"""\n'
        "    __device__ float scale(const float* values, int key, float factor) {\n"
        "        return (values[key] + values[key + 1] + values[key + 2] "
        "+ values[key + 3]) * factor;\n"
        '    }\n    """\n'
    )

    formatted = lint_gpu_kernels.format_python_kernels(source, Path("gpu_kernels.py"))

    assert 88 < max(map(len, formatted.splitlines())) <= 100
    assert formatted.endswith('    """\n')
    assert (
        lint_gpu_kernels.format_python_kernels(formatted, Path("gpu_kernels.py"))
        == formatted
    )


def test_formatter_uses_path_when_not_beside_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formatter = Path(sys.executable).with_name("clang-format")
    if formatter.is_file():
        monkeypatch.setenv("PATH", str(formatter.parent))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    assert lint_gpu_kernels.format_cuda(KERNEL, Path("kernel.cu")) == FORMATTED_KERNEL


def test_check_fix_and_recheck(
    cuda_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = {
        path: path.read_text() for path in cuda_project.rglob("*") if path.is_file()
    }
    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py"])

    assert lint_gpu_kernels.main() == 1
    diff = capsys.readouterr().out
    for path in original:
        if path.name != "other.py":
            assert f"--- {path.relative_to(cuda_project.parents[1])}\n" in diff
    assert "-" + KERNEL in diff
    assert "+__global__ void run(float* x)\n" in diff
    assert "other.py" not in diff
    assert all(path.read_text() == source for path, source in original.items())

    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--fix"])
    assert lint_gpu_kernels.main() == 0
    for path, source in original.items():
        if path.name == "other.py":
            expected = source
        elif path.suffix == ".py":
            expected = (
                'kernel = r"""\n' + textwrap.indent(FORMATTED_KERNEL, "    ") + '"""\n'
            )
        elif path.name == "helper.h":
            expected = "int value() { return 1; }\n"
        else:
            expected = FORMATTED_KERNEL
        assert path.read_text() == expected, path

    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--check"])
    assert lint_gpu_kernels.main() == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("long-line", "exceeds 100 characters"),
        ("incomplete-cuda", "returned non-zero exit status"),
        ("configuration", "UnknownOption"),
        ("missing-source", "gpu_kernels.py"),
        ("missing-formatter", "clang-format"),
        ("invalid-python", "invalid syntax"),
        ("open-string", "unterminated triple-quoted string"),
    ),
)
def test_fix_does_not_write_when_formatting_fails(
    cuda_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    message: str,
) -> None:
    if failure in ("long-line", "incomplete-cuda"):
        (cuda_project / "nested" / "kernel.hpp").write_text(
            "int " + "x" * 101 + ";\n"
            if failure == "long-line"
            else "__global__ void run() {\n    if (\n"
        )
    elif failure == "missing-source":
        (cuda_project / "decoder" / "gpu_kernels.py").unlink()
    elif failure == "configuration":
        (cuda_project.parents[1] / ".clang-format").write_text(
            "ColumnLimit: 100\nUnknownOption: true\n"
        )
    elif failure in ("invalid-python", "open-string"):
        (cuda_project / "decoder" / "gpu_kernels.py").write_text(
            "if True print('broken')\n"
            if failure == "invalid-python"
            else 'kernel = r"""__global__ void run() {}\n'
        )
    else:
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(sys, "executable", str(cuda_project / "python"))
    original = {
        path: path.read_bytes() for path in cuda_project.rglob("*") if path.is_file()
    }
    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--fix"])

    assert lint_gpu_kernels.main() == 2
    assert any(
        record.levelno == logging.ERROR and message in record.getMessage()
        for record in caplog.records
    )
    assert {
        path: path.read_bytes() for path in cuda_project.rglob("*") if path.is_file()
    } == original


@pytest.mark.parametrize(
    "style",
    ("[", "", "[]", "{}", 'ColumnLimit: "100"', "ColumnLimit: 0"),
)
def test_invalid_style_returns_error_without_writing(
    cuda_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    style: str,
) -> None:
    (cuda_project.parents[1] / ".clang-format").write_text(style)
    original = {
        path: path.read_bytes() for path in cuda_project.rglob("*") if path.is_file()
    }
    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--fix"])

    assert lint_gpu_kernels.main() == 2
    message = "expected" if style == "[" else "ColumnLimit"
    assert any(
        record.levelno == logging.ERROR and message in record.getMessage()
        for record in caplog.records
    )
    assert {
        path: path.read_bytes() for path in cuda_project.rglob("*") if path.is_file()
    } == original


def test_write_error_returns_failure(
    cuda_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        Path, "write_text", Mock(side_effect=PermissionError("read-only source"))
    )
    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--fix"])

    assert lint_gpu_kernels.main() == 2
    assert str(cuda_project / "helper.h") in caplog.text
    assert "read-only source" in caplog.text


def test_standalone_formatter_does_not_import_cuda(cuda_project: Path) -> None:
    module_path = cuda_project / "decoder" / "lint_gpu_kernels.py"
    module_path.write_text(MODULE_PATH.read_text())
    for mode, status in (("--check", 1), ("--fix", 0), ("--check", 0)):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import runpy, sys\n"
                "sys.modules['cupy'] = None\n"
                "sys.modules['fast_gpu_asr'] = None\n"
                f"sys.argv = [{str(module_path)!r}, {mode!r}]\n"
                f"runpy.run_path({str(module_path)!r}, run_name='__main__')\n",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == status, result.stderr
        assert "CUDA source files checked" in result.stderr
        if status == 1:
            assert "--- src/fast_gpu_asr/kernel.cu\n" in result.stdout
        else:
            assert result.stdout == ""


def test_check_and_fix_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["lint_gpu_kernels.py", "--check", "--fix"])

    with pytest.raises(SystemExit) as error:
        lint_gpu_kernels.main()

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
