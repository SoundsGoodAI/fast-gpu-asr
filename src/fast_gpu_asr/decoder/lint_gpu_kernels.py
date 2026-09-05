#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Check or format native plugins and embedded decoder CUDA without a GPU.

Run this file directly from a repository checkout to avoid the package's CUDA
imports. Paths and style configuration are relative to the checkout, not the
working directory.
"""

import argparse
import ast
import difflib
import io
import json
import logging
import subprocess
import sys
import textwrap
import tokenize
from itertools import accumulate
from pathlib import Path

import yaml


def format_cuda(source: str, path: Path, indent: str = "") -> str:
    """Format CUDA/C++ and enforce the configured physical line limit.

    Parameters
    ----------
    source : str
        CUDA/C++ source without Python indentation.
    path : Path
        Filename for formatter diagnostics.
    indent : str
        Prefix added to nonempty lines; counts toward the column limit.

    Returns
    -------
    str
        Formatted, indented source.

    Raises
    ------
    OSError
        The style file cannot be read or clang-format cannot be started.
    subprocess.CalledProcessError
        clang-format rejected the source or style.
    ValueError
        Invalid column limit or an unbreakable line exceeding it.
    yaml.YAMLError
        The style file contains invalid YAML.
    """

    root = Path(__file__).resolve().parents[3]
    style = yaml.safe_load((root / ".clang-format").read_text())
    column_limit = style.get("ColumnLimit") if isinstance(style, dict) else None
    if not isinstance(column_limit, int) or column_limit <= len(indent):
        raise ValueError(f"ColumnLimit must be an integer greater than {len(indent)}.")

    style["ColumnLimit"] = column_limit - len(indent)
    executable = Path(sys.executable).with_name("clang-format")
    result = subprocess.run(
        (
            str(executable) if executable.is_file() else "clang-format",
            "--Werror",
            "--fail-on-incomplete-format",
            f"--style={json.dumps(style)}",
            f"--assume-filename={path.with_suffix('.cu')}",
        ),
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )

    formatted = "".join(
        indent + line if line.strip() else line for line in io.StringIO(result.stdout)
    )
    for line_number, line in enumerate(formatted.split("\n"), 1):
        if len(line) > column_limit:
            raise ValueError(
                f"{path}:{line_number}: line exceeds {column_limit} characters "
                "after formatting; split the long token or comment manually."
            )

    return formatted


def format_python_kernels(source: str, path: Path) -> str:
    """Format raw triple-quoted CUDA strings without rewriting Python code.

    Parameters
    ----------
    source : str
        Python module containing raw CUDA strings with device or kernel functions.
    path : Path
        Python source path used in formatter diagnostics.

    Returns
    -------
    str
        Original Python text with only CUDA string bodies reformatted. Python
        tokens and ordinary strings remain unchanged.

    Notes
    -----
    Only raw literals containing ``__global__`` or ``__device__`` are formatted.
    Python is parsed without importing or executing it. CUDA bodies use four
    spaces of indentation while retaining their original quote delimiters.
    Offsets follow the tokenizer's physical lines, not Unicode line separators.
    """

    ast.parse(source, filename=str(path))
    offsets = list(accumulate(map(len, io.StringIO(source)), initial=0))
    cursor, parts = 0, []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.STRING:
            continue
        if token.string[:4].lower() not in ('r"""', "r'''"):
            continue

        body = ast.literal_eval(token.string)
        if "__global__" not in body and "__device__" not in body:
            continue

        formatted = format_cuda(textwrap.dedent(body).strip("\n") + "\n", path, "    ")
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        closing_line = token.string.rsplit("\n", 1)[-1]
        closing_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
        parts.extend(
            (
                source[cursor:start],
                token.string[:4]
                + "\n"
                + formatted
                + closing_indent
                + token.string[-3:],
            )
        )
        cursor = end

    parts.append(source[cursor:])

    return "".join(parts)


def main() -> int:
    """Check all production CUDA sources, or apply formatting with --fix.

    Returns
    -------
    int
        Zero on success, one for formatting differences in check mode, or two
        for input, formatter, or write errors. All inputs are formatted before
        writing starts; write failures are not rolled back.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check only (default).")
    mode.add_argument("--fix", action="store_true", help="Apply CUDA formatting.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    source_dir = root / "src" / "fast_gpu_asr"
    paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix in (".cu", ".cuh", ".cpp", ".h", ".hpp") and path.is_file()
    )
    paths.append(source_dir / "decoder" / "gpu_kernels.py")
    changes = []
    try:
        for path in paths:
            source = path.read_text()
            formatted = (
                format_python_kernels(source, path)
                if path.suffix == ".py"
                else format_cuda(source, path)
            )
            if source != formatted:
                changes.append((path, source, formatted))

        for path, source, formatted in changes:
            if args.fix:
                path.write_text(formatted)
            else:
                sys.stdout.writelines(
                    difflib.unified_diff(
                        list(io.StringIO(source)),
                        list(io.StringIO(formatted)),
                        fromfile=str(path.relative_to(root)),
                        tofile=str(path.relative_to(root)),
                    )
                )
    except (
        OSError,
        SyntaxError,
        ValueError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as error:
        logger.error("%s: %s", path, getattr(error, "stderr", None) or error)
        return 2
    logger.info(
        "%d CUDA source files checked; %d %s.",
        len(paths),
        len(changes),
        "formatted" if args.fix else "need formatting",
    )

    return int(bool(changes) and not args.fix)


if __name__ == "__main__":
    raise SystemExit(main())
