#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Validate repository automation and generated-artifact policies.

The tests in this module cover contracts that span GitHub Actions, actionlint,
package metadata, the TensorRT plugin manifest, and Git ignore rules. Structured
parsing is used where possible so comments and unrelated text cannot satisfy an
assertion accidentally.
"""

import ast
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from fast_gpu_asr.tensorrt_plugins.constants import PLUGIN_INITIALIZERS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACTIONLINT_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "actionlint.yaml"
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
STANDARD_SELF_HOSTED_LABELS = {"self-hosted", "linux", "x64"}


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML document whose root is a mapping.

    Parameters
    ----------
    path : Path
        Path to the YAML document.

    Returns
    -------
    dict[str, Any]
        Parsed root mapping.

    Raises
    ------
    TypeError
        Raised when the document root is not a mapping.
    """

    document = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(document, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(document)}.")
    return document


def get_job(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a workflow job by its stable identifier.

    Parameters
    ----------
    workflow : dict[str, Any]
        Parsed GitHub Actions workflow.
    name : str
        Job identifier under the workflow's ``jobs`` mapping.

    Returns
    -------
    dict[str, Any]
        Requested job configuration.

    Raises
    ------
    AssertionError
        Raised when the jobs mapping or requested job is missing or malformed.
    """

    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict) or not isinstance(jobs.get(name), dict):
        raise AssertionError(f"Workflow job {name!r} is missing or malformed.")
    return jobs[name]


def get_named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the unique step with a given display name.

    Parameters
    ----------
    job : dict[str, Any]
        Parsed GitHub Actions job.
    name : str
        Exact value of the step's ``name`` field.

    Returns
    -------
    dict[str, Any]
        Matching workflow step.

    Raises
    ------
    AssertionError
        Raised when the steps list is malformed or the name is not unique.
    """

    steps = job.get("steps")
    if not isinstance(steps, list):
        raise AssertionError("Workflow job has no steps list.")
    matches = [
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {name!r} step, found {len(matches)}.")
    return matches[0]


def get_run_script(step: dict[str, Any]) -> str:
    """Return a workflow step's nonempty shell script.

    Parameters
    ----------
    step : dict[str, Any]
        Parsed GitHub Actions step.

    Returns
    -------
    str
        Contents of the step's ``run`` field.

    Raises
    ------
    AssertionError
        Raised when the step does not contain a nonempty shell script.
    """

    script = step.get("run")
    if not isinstance(script, str) or not script.strip():
        raise AssertionError("Expected workflow step to contain a shell script.")
    return script


def get_python_heredoc(script: str) -> str:
    """Extract a unique single-quoted ``PY`` heredoc from a shell script.

    Parameters
    ----------
    script : str
        Shell source containing the embedded Python program.

    Returns
    -------
    str
        Python source between the heredoc delimiters.

    Raises
    ------
    AssertionError
        Raised unless the script contains exactly one matching heredoc.
    """

    matches = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, flags=re.S)
    if len(matches) != 1:
        raise AssertionError(f"Expected one Python heredoc, found {len(matches)}.")
    return matches[0]


def get_literal_string_set(source: str, variable: str) -> set[str]:
    """Read a literal string-set assignment from Python source.

    Parameters
    ----------
    source : str
        Python source to parse.
    variable : str
        Name assigned the expected set literal.

    Returns
    -------
    set[str]
        Strings contained in the assigned set.

    Raises
    ------
    AssertionError
        Raised when the assignment is missing, duplicated, or not a literal set
        containing only strings.
    """

    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise AssertionError(
            f"Expected one assignment to {variable!r}, found {len(assignments)}."
        )
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, set) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"Expected {variable!r} to be a literal string set.")
    return value


def is_immutable_action_reference(reference: str) -> bool:
    """Return whether an action reference is local or pinned immutably.

    Local actions are versioned with the repository. Remote GitHub actions must
    use a full 40-character commit SHA, while Docker actions must use a full
    SHA-256 image digest.

    Parameters
    ----------
    reference : str
        Value of a workflow ``uses`` field.

    Returns
    -------
    bool
        ``True`` when the reference cannot move independently of the workflow.
    """

    if reference.startswith("./"):
        return True
    if reference.startswith("docker://"):
        return (
            re.fullmatch(r"docker://[^@\s]+@sha256:[0-9a-f]{64}", reference) is not None
        )
    return re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference) is not None


def test_actionlint_runner_labels_match_gpu_workflow() -> None:
    """Keep actionlint's custom labels synchronized with the GPU runner.

    The GPU job must retain the standard self-hosted Linux x64 labels and the
    project-specific ``gpu`` label. Only custom labels belong in actionlint's
    configuration.
    """

    actionlint_config = load_yaml_mapping(ACTIONLINT_CONFIG_PATH)
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    gpu_job = get_job(workflow, "gpu-tests")

    runner_labels = gpu_job.get("runs-on")
    if not isinstance(runner_labels, list) or not all(
        isinstance(label, str) for label in runner_labels
    ):
        raise AssertionError("gpu-tests.runs-on must be a list of string labels.")
    assert len(runner_labels) == len(set(runner_labels))
    assert set(runner_labels) >= STANDARD_SELF_HOSTED_LABELS
    assert "gpu" in runner_labels

    self_hosted_runner = actionlint_config.get("self-hosted-runner")
    if not isinstance(self_hosted_runner, dict):
        raise AssertionError("actionlint is missing self-hosted-runner configuration.")
    configured_labels = self_hosted_runner.get("labels")
    if not isinstance(configured_labels, list) or not all(
        isinstance(label, str) for label in configured_labels
    ):
        raise AssertionError("actionlint runner labels must be a list of strings.")
    assert len(configured_labels) == len(set(configured_labels))
    assert set(configured_labels) == set(runner_labels) - STANDARD_SELF_HOSTED_LABELS


def test_gpu_ci_verifies_every_native_plugin_library() -> None:
    """Keep GPU plugin rebuilding and wheel verification synchronized.

    CI must remove stale libraries, rebuild all declared plugins, check the
    resulting library count, and compare the wheel contents with the production
    plugin manifest.
    """

    workflow = load_yaml_mapping(WORKFLOW_PATH)
    gpu_job = get_job(workflow, "gpu-tests")
    expected_libraries = {library_name for library_name, _ in PLUGIN_INITIALIZERS}

    rebuild_script = get_run_script(get_named_step(gpu_job, "Rebuild native plugins"))
    assert "rm -f src/fast_gpu_asr/tensorrt_plugins/*.so" in rebuild_script
    assert (
        "uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build"
        in rebuild_script
    )
    count_matches = re.findall(
        r"find src/fast_gpu_asr/tensorrt_plugins -maxdepth 1 "
        r"-name '\*\.so' \| wc -l\)\" -eq (\d+)",
        rebuild_script,
    )
    assert count_matches == [str(len(expected_libraries))]

    smoke_script = get_run_script(
        get_named_step(gpu_job, "Verify and smoke-test installed wheel")
    )
    smoke_test_source = get_python_heredoc(smoke_script)
    verified_libraries = get_literal_string_set(
        smoke_test_source,
        "expected_libraries",
    )
    assert verified_libraries == expected_libraries
    smoke_test_tree = ast.parse(smoke_test_source)
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "libraries"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotEq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "expected_libraries"
        for node in ast.walk(smoke_test_tree)
    )


def test_ci_runs_hosted_tests_for_every_supported_python() -> None:
    """Run hosted tests for every supported Python minor version.

    The workflow matrix, package classifiers, and ``requires-python`` range must
    describe the same contiguous version set. The GPU job must also wait for the
    quality and complete hosted-test jobs before using the self-hosted runner.
    """

    workflow = load_yaml_mapping(WORKFLOW_PATH)
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf8"))
    python_job = get_job(workflow, "python-tests")
    gpu_job = get_job(workflow, "gpu-tests")

    strategy = python_job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    tested_versions = matrix.get("python-version") if isinstance(matrix, dict) else None
    if not isinstance(tested_versions, list) or not all(
        isinstance(version, str) for version in tested_versions
    ):
        raise AssertionError("python-tests must define a string python-version matrix.")
    assert tested_versions
    assert len(tested_versions) == len(set(tested_versions))

    project = pyproject["project"]
    classifier_prefix = "Programming Language :: Python :: "
    classified_versions = [
        match.group(1)
        for classifier in project["classifiers"]
        if (match := re.fullmatch(classifier_prefix + r"(\d+\.\d+)", classifier))
    ]
    assert set(tested_versions) == set(classified_versions)

    supported_versions = sorted(Version(version) for version in tested_versions)
    version_pairs = [(version.major, version.minor) for version in supported_versions]
    assert all(
        current == (previous[0], previous[1] + 1)
        for previous, current in zip(version_pairs, version_pairs[1:], strict=False)
    )
    requires_python = SpecifierSet(project["requires-python"])
    assert all(version in requires_python for version in supported_versions)
    first_major, first_minor = version_pairs[0]
    last_major, last_minor = version_pairs[-1]
    assert Version(f"{first_major}.{first_minor - 1}") not in requires_python
    assert Version(f"{last_major}.{last_minor + 1}") not in requires_python

    hosted_runner = python_job.get("runs-on")
    assert isinstance(hosted_runner, str)
    assert re.fullmatch(r"ubuntu-(?:latest|\d{2}\.\d{2})", hosted_runner)
    assert get_run_script(get_named_step(python_job, "Run CPU test suite")).strip() == (
        "uv run --frozen pytest -q"
    )
    needs = gpu_job.get("needs")
    if isinstance(needs, str):
        needs = [needs]
    assert isinstance(needs, list) and all(isinstance(job, str) for job in needs)
    assert set(needs) >= {"quality", "python-tests"}


def test_ci_actions_are_pinned_to_commit_hashes() -> None:
    """Prevent mutable remote action references from entering CI.

    Both reusable jobs and individual workflow steps are inspected. Local
    actions remain valid because they are pinned by the repository commit.
    """

    workflow = load_yaml_mapping(WORKFLOW_PATH)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise AssertionError("Workflow jobs are missing or malformed.")

    action_references: list[str] = []
    for job in jobs.values():
        if not isinstance(job, dict):
            raise AssertionError("Workflow job is not a mapping.")
        if isinstance(job.get("uses"), str):
            action_references.append(job["uses"])
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            raise AssertionError("Workflow steps are not a list.")
        action_references.extend(
            step["uses"]
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("uses"), str)
        )

    assert action_references
    mutable_references = [
        reference
        for reference in action_references
        if not is_immutable_action_reference(reference)
    ]
    assert mutable_references == []


def test_generated_plugin_artifacts_are_ignored_and_untracked() -> None:
    """Keep generated native plugin artifacts ignored and untracked.

    Git's own path matching validates every production library and a nested
    temporary build path. A separate index query catches generated artifacts
    that were accidentally committed despite matching an ignore rule.
    """

    generated_paths = (
        *(
            f"src/fast_gpu_asr/tensorrt_plugins/{library_name}"
            for library_name, _ in PLUGIN_INITIALIZERS
        ),
        "src/fast_gpu_asr/tensorrt_plugins/.plugin-build-example/example_plugin.so",
    )
    for path in generated_paths:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--no-index", path),
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"Generated plugin path is not ignored: {path}"

    tracked = subprocess.run(
        (
            "git",
            "ls-files",
            "--",
            "src/fast_gpu_asr/tensorrt_plugins/*.so",
            "src/fast_gpu_asr/tensorrt_plugins/.plugin-build-*",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""
