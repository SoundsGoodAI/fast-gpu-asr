#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Validate CI, package metadata, and generated-artifact policies."""

import ast
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import fast_gpu_asr.constants as asr_constants
from fast_gpu_asr.tensorrt_plugins.constants import PLUGIN_BUILDS, PLUGIN_INITIALIZERS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACTIONLINT_CONFIG_PATH = REPOSITORY_ROOT / ".github" / "actionlint.yaml"
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOW_DIRECTORY / "ci.yml"
LOCAL_ACTION_DIRECTORY = REPOSITORY_ROOT / ".github" / "actions"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
PLUGIN_DIRECTORY = REPOSITORY_ROOT / "src" / "fast_gpu_asr" / "tensorrt_plugins"
STANDARD_SELF_HOSTED_LABELS = {"self-hosted", "linux", "x64"}
MANUAL_GPU_CONDITION = (
    "github.event_name == 'workflow_dispatch' && inputs.run_gpu_tests"
)
PUBLISH_CONDITION = (
    "github.event_name == 'workflow_dispatch' && inputs.run_gpu_tests && "
    "(inputs.publish_to == 'testpypi' || inputs.publish_to == 'pypi') && "
    "startsWith(github.ref, 'refs/tags/v') && "
    "github.repository == 'SoundsGoodAI/fast-gpu-asr'"
)
YAML_BOOLEAN_TAG = "tag:yaml.org,2002:bool"


class GitHubActionsLoader(yaml.SafeLoader):
    """Parse booleans without treating GitHub's ``on`` key as YAML 1.1 syntax."""


GitHubActionsLoader.yaml_implicit_resolvers = {
    character: [
        (tag, resolver) for tag, resolver in resolvers if tag != YAML_BOOLEAN_TAG
    ]
    for character, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
GitHubActionsLoader.add_implicit_resolver(
    YAML_BOOLEAN_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping without interpreting the workflow's ``on`` key.

    Parameters
    ----------
    path : Path
        Workflow, local-action, or actionlint YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed top-level mapping with GitHub-compatible boolean values.
    """

    document = yaml.load(path.read_text(encoding="utf8"), Loader=GitHubActionsLoader)
    assert isinstance(document, dict), path
    return document


def repository_workflow_paths() -> list[Path]:
    """Find workflow definitions under the repository's workflow directory.

    Returns
    -------
    list[Path]
        Sorted paths with either supported YAML suffix.
    """

    return sorted(
        (*WORKFLOW_DIRECTORY.glob("*.yml"), *WORKFLOW_DIRECTORY.glob("*.yaml"))
    )


def workflow_events(workflow: dict[str, Any]) -> set[str]:
    """Read event names from a workflow's trigger declaration.

    Parameters
    ----------
    workflow : dict[str, Any]
        Parsed workflow with a string, list, or mapping-valued ``on`` key.

    Returns
    -------
    set[str]
        Nonempty set of configured trigger names.
    """

    triggers = workflow["on"]
    assert isinstance(triggers, (str, list, dict))
    events = {triggers} if isinstance(triggers, str) else set(triggers)
    assert events and all(isinstance(event, str) for event in events)
    return events


def get_named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    """Find exactly one step with the requested display name.

    Parameters
    ----------
    job : dict[str, Any]
        Parsed job containing its ordered steps.
    name : str
        Exact display name of the required step.

    Returns
    -------
    dict[str, Any]
        Matching step; missing or duplicate matches fail unpacking.
    """

    [step] = [step for step in job["steps"] if step.get("name") == name]
    return step


def assert_mandatory_step(step: dict[str, Any]) -> None:
    """Check that a step runs unconditionally with repository-default execution.

    Parameters
    ----------
    step : dict[str, Any]
        Step whose own condition, error handling, shell, and working directory
        are checked; inherited workflow and job settings are outside this helper.
    """

    assert "if" not in step
    assert step.get("continue-on-error") in (None, False)
    assert step.get("shell") in (None, "bash")
    assert "working-directory" not in step


def get_action_step(job: dict[str, Any], action_name: str) -> dict[str, Any]:
    """Find the unique step that invokes a named remote action.

    Parameters
    ----------
    job : dict[str, Any]
        Parsed job containing its ordered steps.
    action_name : str
        Case-insensitive ``owner/repository`` name, without its version suffix.

    Returns
    -------
    dict[str, Any]
        Matching step; missing or duplicate matches fail unpacking.
    """

    [step] = [
        step
        for step in job["steps"]
        if step.get("uses", "").casefold().startswith(action_name.casefold() + "@")
    ]
    return step


def assert_pytest_is_not_filtered(*scopes: dict[str, Any]) -> None:
    """Reject environment-level pytest arguments that can hide test modules.

    Parameters
    ----------
    *scopes : dict[str, Any]
        Workflow, job, and step mappings whose environments must not define
        ``PYTEST_ADDOPTS``.
    """

    for scope in scopes:
        environment = scope.get("env", {})
        assert isinstance(environment, dict)
        assert "PYTEST_ADDOPTS" not in environment


def assert_required_run_step(
    job: dict[str, Any], name: str, expected_script: str
) -> dict[str, Any]:
    """Find a mandatory step and verify its shell command.

    Parameters
    ----------
    job : dict[str, Any]
        Parsed job containing the step.
    name : str
        Exact display name of the required step.
    expected_script : str
        Expected command after stripping leading and trailing whitespace.

    Returns
    -------
    dict[str, Any]
        Validated step for further environment or ordering checks.
    """

    step = get_named_step(job, name)
    assert_mandatory_step(step)
    assert step["run"].strip() == expected_script
    return step


def get_python_heredoc(script: str) -> str:
    """Extract exactly one single-quoted ``PY`` heredoc from a shell script.

    Parameters
    ----------
    script : str
        Shell source containing the embedded Python program.

    Returns
    -------
    str
        Python source between the heredoc delimiters. Missing or duplicate
        heredocs fail unpacking.
    """

    [source] = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", script, flags=re.S)
    return source


def get_literal_string_set(source: str, variable: str) -> set[str]:
    """Read a unique top-level string-set assignment without executing it.

    Parameters
    ----------
    source : str
        Python program containing a literal set assignment.
    variable : str
        Name of the assigned variable.

    Returns
    -------
    set[str]
        Evaluated string set; missing, duplicate, or nonliteral values fail.
    """

    [assignment] = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == variable for target in node.targets)
    ]
    value = ast.literal_eval(assignment.value)
    assert isinstance(value, set) and all(isinstance(item, str) for item in value)
    return value


def is_immutable_action_reference(reference: str) -> bool:
    """Check whether an action reference satisfies the pinning policy.

    Parameters
    ----------
    reference : str
        Local path, remote action reference, or ``docker://`` image reference.

    Returns
    -------
    bool
        Whether the reference is local without parent traversal, uses a full
        commit SHA, or pins a full image digest. Local paths are resolved and
        checked for existence separately.
    """

    if reference.startswith("./"):
        return ".." not in PurePosixPath(reference).parts
    if reference.startswith("docker://"):
        return is_immutable_container_image(reference.removeprefix("docker://"))
    return re.fullmatch(r"[^@\s]+@[0-9a-fA-F]{40}", reference) is not None


def is_checkout_action_reference(reference: str) -> bool:
    """Recognize the official checkout action regardless of letter case.

    Parameters
    ----------
    reference : str
        Action reference, including its version suffix.

    Returns
    -------
    bool
        Whether the reference uses exactly ``actions/checkout``.
    """

    return reference.casefold().startswith("actions/checkout@")


def is_immutable_container_image(reference: str) -> bool:
    """Check whether a container image is pinned by its SHA-256 digest.

    Parameters
    ----------
    reference : str
        Image name and digest, without a ``docker://`` prefix.

    Returns
    -------
    bool
        Whether the reference ends in a full lowercase SHA-256 digest.
    """

    return re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", reference) is not None


def contains_runtime_error_raise(statements: Iterable[ast.AST]) -> bool:
    """Look for a direct ``raise RuntimeError(...)`` among AST nodes.

    Parameters
    ----------
    statements : Iterable[ast.AST]
        Nodes to inspect. Pass ``ast.walk`` when nested statements must count.

    Returns
    -------
    bool
        Whether any supplied node raises ``RuntimeError`` directly. This is a
        structural check, not proof that the raise is reachable at runtime.
    """

    return any(
        isinstance(statement, ast.Raise)
        and isinstance(statement.exc, ast.Call)
        and isinstance(statement.exc.func, ast.Name)
        and statement.exc.func.id == "RuntimeError"
        for statement in statements
    )


def test_workflow_yaml_loader_does_not_change_pyyaml_defaults(tmp_path: Path) -> None:
    source = "on: [push]\nenabled: true\nlegacy: off\n"
    path = tmp_path / "workflow.yml"
    path.write_text(source, encoding="utf8")

    assert load_yaml_mapping(path) == {"on": ["push"], "enabled": True, "legacy": "off"}
    assert yaml.safe_load(source) == {True: ["push"], "enabled": True, "legacy": False}


def test_actionlint_runner_labels_match_gpu_workflow() -> None:
    actionlint_config = load_yaml_mapping(ACTIONLINT_CONFIG_PATH)
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    assert workflow["jobs"]["gpu-tests"]["runs-on"] == "gpu-t4"
    assert actionlint_config["self-hosted-runner"]["labels"] == ["gpu-t4"]


def test_gpu_ci_requires_explicit_manual_opt_in() -> None:
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    assert workflow_events(workflow) == {"push", "pull_request", "workflow_dispatch"}
    gpu_input = workflow["on"]["workflow_dispatch"]["inputs"]["run_gpu_tests"]
    assert gpu_input["type"] == "boolean"
    assert gpu_input["default"] is False


def test_ci_protects_gpu_runs_and_checkout_credentials() -> None:
    workflow_paths = repository_workflow_paths()
    assert WORKFLOW_PATH in workflow_paths
    ci_workflow = load_yaml_mapping(WORKFLOW_PATH)
    assert "pull_request" in workflow_events(ci_workflow)
    assert ci_workflow["on"]["pull_request"] in (None, {})
    assert ci_workflow.get("permissions") == {"contents": "read"}
    actionlint_config = load_yaml_mapping(ACTIONLINT_CONFIG_PATH)
    configured_labels = actionlint_config["self-hosted-runner"]["labels"]
    restricted_label_names = {
        label.casefold() for label in STANDARD_SELF_HOSTED_LABELS
    } | {label.casefold() for label in configured_labels}

    restricted_jobs = []
    checkout_steps = []
    for workflow_path in workflow_paths:
        workflow = load_yaml_mapping(workflow_path)
        events = workflow_events(workflow)
        assert "pull_request_target" not in events
        jobs = workflow["jobs"]

        if "pull_request" in events:
            for scope in (workflow, *jobs.values()):
                permissions = scope.get("permissions", workflow.get("permissions"))
                if scope is jobs.get("publish"):
                    assert " ".join(scope["if"].split()) == PUBLISH_CONDITION
                    assert permissions == {"id-token": "write"}
                    continue
                if permissions != "read-all":
                    assert isinstance(permissions, dict), workflow_path
                    assert set(permissions.values()) <= {"none", "read"}, workflow_path

        for job_name, job in jobs.items():
            runner = job.get("runs-on")
            if runner is None and "uses" in job:
                runner_labels = []
            else:
                assert isinstance(runner, (str, list)), job_name
                runner_labels = [runner] if isinstance(runner, str) else runner

            if isinstance(runner, list) or any(
                "${{" in label or label.casefold() in restricted_label_names
                for label in runner_labels
            ):
                restricted_jobs.append((workflow_path, job_name))
                assert job.get("continue-on-error") in (None, False)
                timeout = job.get("timeout-minutes")
                assert type(timeout) is int and 0 < timeout <= 60
                assert " ".join(job["if"].split()) == MANUAL_GPU_CONDITION

            checkout_steps.extend(
                step
                for step in job.get("steps", [])
                if is_checkout_action_reference(step.get("uses", ""))
            )

    assert (WORKFLOW_PATH, "gpu-tests") in restricted_jobs
    assert checkout_steps
    for step in checkout_steps:
        assert step["with"]["persist-credentials"] is False


def test_ci_quality_job_enforces_repository_checks() -> None:
    quality_job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["quality"]
    assert re.fullmatch(r"ubuntu-(?:latest|\d{2}\.\d{2})", quality_job["runs-on"])
    timeout = quality_job.get("timeout-minutes")
    assert type(timeout) is int and 0 < timeout <= 20

    expected_steps = {
        "Check lockfile": "uv lock --check",
        "Install development environment": "uv sync --frozen --extra dev",
        "Lint": "uv run --frozen ruff check .",
        "Check formatting": "uv run --frozen ruff format --check .",
        "Check CUDA formatting": (
            "uv run --frozen python "
            "src/fast_gpu_asr/decoder/lint_gpu_kernels.py --check"
        ),
        "Compile Python sources": (
            "uv run --frozen python -m compileall -q src tests scripts"
        ),
    }
    for name, expected_script in expected_steps.items():
        assert_required_run_step(quality_job, name, expected_script)

    actionlint_step = get_named_step(quality_job, "Validate GitHub Actions workflows")
    assert_mandatory_step(actionlint_step)
    actionlint_environment = actionlint_step["env"]
    assert re.fullmatch(r"[0-9a-f]{64}", actionlint_environment["ACTIONLINT_SHA256"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", actionlint_environment["ACTIONLINT_VERSION"])
    actionlint_script = actionlint_step["run"]
    actionlint_commands = {
        line.strip() for line in actionlint_script.splitlines() if line.strip()
    }
    assert (
        'printf \'%s  %s\\n\' "${ACTIONLINT_SHA256}" "${archive}" | sha256sum --check'
    ) in actionlint_commands
    assert '"${RUNNER_TEMP}/actionlint" -color' in actionlint_commands


def test_ci_jobs_checkout_source_and_bootstrap_uv_reproducibly() -> None:
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    tested_versions = workflow["jobs"]["python-tests"]["strategy"]["matrix"][
        "python-version"
    ]
    supported_versions = sorted(Version(version) for version in tested_versions)
    assert supported_versions

    expected_python_versions = {
        "quality": str(supported_versions[0]),
        "python-tests": "${{ matrix.python-version }}",
        "gpu-tests": str(supported_versions[-1]),
    }
    expected_cache_settings = {
        "quality": True,
        "python-tests": True,
        "gpu-tests": False,
    }
    setup_uv_versions = set()
    for job_name, expected_python_version in expected_python_versions.items():
        job = workflow["jobs"][job_name]
        assert job.get("continue-on-error") in (None, False)
        if job_name != "gpu-tests":
            assert "if" not in job

        checkout_step = get_action_step(job, "actions/checkout")
        assert_mandatory_step(checkout_step)
        checkout_inputs = checkout_step["with"]
        assert checkout_inputs.get("persist-credentials") is False
        assert (
            not {"path", "ref", "repository", "sparse-checkout"}
            & checkout_inputs.keys()
        )
        if job_name == "gpu-tests":
            assert checkout_inputs.get("clean") is True

        setup_uv_step = get_action_step(job, "astral-sh/setup-uv")
        assert_mandatory_step(setup_uv_step)
        setup_uv_inputs = setup_uv_step["with"]
        assert setup_uv_inputs.get("python-version") == expected_python_version
        assert setup_uv_inputs.get("enable-cache") is expected_cache_settings[job_name]
        setup_uv_version = setup_uv_inputs["version"]
        assert re.fullmatch(r"\d+\.\d+\.\d+", setup_uv_version)
        setup_uv_versions.add(setup_uv_version)

        steps = job["steps"]
        assert steps.index(checkout_step) < steps.index(setup_uv_step)
        first_uv_step_name = {
            "quality": "Check lockfile",
            "python-tests": "Install development environment",
            "gpu-tests": "Install CUDA, export, and test dependencies",
        }[job_name]
        assert steps.index(setup_uv_step) < steps.index(
            get_named_step(job, first_uv_step_name)
        )
        if job_name == "quality":
            assert steps.index(checkout_step) < steps.index(
                get_named_step(job, "Validate GitHub Actions workflows")
            )
        elif job_name == "gpu-tests":
            assert steps.index(get_named_step(job, "Configure temporary paths")) < (
                steps.index(setup_uv_step)
            )

    assert len(setup_uv_versions) == 1


@pytest.mark.parametrize(
    ("gpu_tests", "ref", "error"),
    (
        ("true", "refs/tags/v0.1.0", None),
        ("false", "refs/tags/v0.1.0", "run_gpu_tests=true"),
        ("true", "refs/heads/main", "version tag v0.1.0"),
        ("true", "refs/heads/v0.1.0", "version tag v0.1.0"),
        ("true", "refs/tags/v0.2.0", "version tag v0.1.0"),
    ),
)
def test_ci_release_request_requires_gpu_tests_and_matching_tag(
    tmp_path: Path, gpu_tests: str, ref: str, error: str | None
) -> None:
    job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["quality"]
    step = get_named_step(job, "Validate release request")
    assert step["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to != 'none'"
    )
    assert step["env"] == {"RUN_GPU_TESTS": "${{ inputs.run_gpu_tests }}"}
    assert step.get("continue-on-error") in (None, False)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    result = subprocess.run(
        (sys.executable, "-I", "-c", get_python_heredoc(step["run"])),
        cwd=tmp_path,
        env={**os.environ, "RUN_GPU_TESTS": gpu_tests, "GITHUB_REF": ref},
        capture_output=True,
        text=True,
        timeout=10,
    )
    if error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert error in result.stderr


def test_ci_publishes_only_its_tested_wheel_with_scoped_credentials() -> None:
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    publish_input = workflow["on"]["workflow_dispatch"]["inputs"]["publish_to"]
    assert publish_input["type"] == "choice"
    assert publish_input["default"] == "none"
    assert publish_input["options"] == ["none", "testpypi", "pypi"]

    gpu_job = workflow["jobs"]["gpu-tests"]
    metadata_step = assert_required_run_step(
        gpu_job,
        "Validate PyPI metadata",
        "uvx --from twine twine check --strict dist/*.whl",
    )
    assert (
        gpu_job["steps"].index(get_named_step(gpu_job, "Build platform wheel"))
        < gpu_job["steps"].index(metadata_step)
        < gpu_job["steps"].index(get_named_step(gpu_job, "Upload wheel"))
    )

    job = workflow["jobs"]["publish"]
    assert job["needs"] == ["gpu-tests"]
    assert " ".join(job["if"].split()) == PUBLISH_CONDITION
    assert job["permissions"] == {"id-token": "write"}
    assert job["environment"]["name"] == "${{ inputs.publish_to }}"
    download = get_action_step(job, "actions/download-artifact")
    publish = get_action_step(job, "pypa/gh-action-pypi-publish")
    assert job["steps"] == [download, publish]
    assert_mandatory_step(download)
    assert_mandatory_step(publish)
    assert download["with"] == {
        "name": get_named_step(gpu_job, "Upload wheel")["with"]["name"],
        "path": "dist",
    }
    assert publish["with"] == {
        "repository-url": (
            "${{ inputs.publish_to == 'pypi' && 'https://upload.pypi.org/legacy/' "
            "|| 'https://test.pypi.org/legacy/' }}"
        ),
        "attestations": True,
    }


def test_gpu_ci_verifies_every_native_plugin_library() -> None:
    gpu_job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["gpu-tests"]
    expected_libraries = {library for library, _ in PLUGIN_INITIALIZERS}
    expected_headers = {
        f"fast_gpu_asr/tensorrt_plugins/{path.name}"
        for path in PLUGIN_DIRECTORY.glob("*.h")
    }
    expected_creators = {
        value
        for name, value in vars(asr_constants).items()
        if name.endswith("_PLUGIN_NAME")
    }
    assert expected_libraries and expected_headers and expected_creators

    rebuild_step = get_named_step(gpu_job, "Rebuild native plugins")
    assert_mandatory_step(rebuild_step)
    assert [
        line.strip() for line in rebuild_step["run"].splitlines() if line.strip()
    ] == [
        "rm -f src/fast_gpu_asr/tensorrt_plugins/*.so",
        "uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build",
        'test "$(find src/fast_gpu_asr/tensorrt_plugins -maxdepth 1 '
        f"-name '*.so' | wc -l)\" -eq {len(expected_libraries)}",
    ]

    smoke_step = get_named_step(gpu_job, "Verify and smoke-test installed wheel")
    assert_mandatory_step(smoke_step)
    smoke_test_source = get_python_heredoc(smoke_step["run"])
    for variable, expected in (
        ("expected_libraries", expected_libraries),
        ("expected_headers", expected_headers),
        ("expected_creators", expected_creators),
    ):
        assert get_literal_string_set(smoke_test_source, variable) == expected, variable
    smoke_test_tree = ast.parse(smoke_test_source)
    header_assignments = [
        ast.unparse(node)
        for node in smoke_test_tree.body
        if isinstance(node, ast.Assign)
    ]
    assert (
        header_assignments.count("missing_headers = expected_headers - archive_names")
        == 1
    )
    for condition in ("libraries != expected_libraries", "missing_headers"):
        [guard] = [
            node
            for node in smoke_test_tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == condition
        ]
        assert contains_runtime_error_raise(guard.body), condition

    [creator_loop] = [
        node
        for node in ast.walk(smoke_test_tree)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "expected_creators"
    ]
    assert contains_runtime_error_raise(ast.walk(creator_loop))


@pytest.mark.parametrize(
    ("driver_version", "accepted"),
    (
        ("535.183.01", False),
        ("579.99", False),
        ("580.65.06", True),
        ("590.48", True),
        (None, False),
    ),
)
def test_gpu_ci_driver_preflight(driver_version: str | None, accepted: bool) -> None:
    gpu_job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["gpu-tests"]
    step = get_named_step(gpu_job, "Verify NVIDIA driver")
    assert_mandatory_step(step)
    script = """nvidia-smi() {
        [[ -n "$DRIVER_VERSION" ]] || return 1
        if (( $# == 0 )); then return 0; fi
        [[ "$*" == "--query-gpu=driver_version --format=csv,noheader" ]] || return 64
        printf '%s\\n' "$DRIVER_VERSION"
    }
    """ + step["run"]
    result = subprocess.run(
        ("bash", "-euo", "pipefail", "-c", script),
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "DRIVER_VERSION": driver_version or ""},
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if not accepted and driver_version is not None:
        assert f"this image has {driver_version}" in result.stderr


@pytest.mark.parametrize("version", ("11.2.1.2", "11.3.0.4"))
def test_gpu_ci_selects_headers_from_installed_tensorrt(
    tmp_path: Path, version: str
) -> None:
    gpu_job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["gpu-tests"]
    step = get_named_step(gpu_job, "Install native build prerequisites")
    assert_mandatory_step(step)
    assert "env" not in step
    metadata_query = (
        'from importlib.metadata import version; print(version("tensorrt-cu13"))'
    )
    script = r"""
    sudo() { [[ "$*" == apt-get* ]]; }
    uv() {
        [[ "$1 $2 $3 $4" == "run --frozen python -c" ]] || return 64
        [[ "$5" == "$METADATA_QUERY" ]] || return 64
        printf '%s\n' "$TENSORRT_VERSION"
    }
    curl() { printf '%s\n' "$@"; }
    tar() { printf '%s\n' "$@"; }
    """ + step["run"]
    github_env = tmp_path / "github-env"
    result = subprocess.run(
        ("bash", "-euo", "pipefail", "-c", script),
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_ENV": str(github_env),
            "CPLUS_INCLUDE_PATH": "/existing/includes",
            "TENSORRT_VERSION": version,
            "METADATA_QUERY": metadata_query,
        },
    )
    assert result.returncode == 0, result.stderr
    release = ".".join(version.split(".")[:2])
    assert (
        f"https://codeload.github.com/NVIDIA/TensorRT/tar.gz/refs/tags/v{release}"
        in result.stdout.splitlines()
    )
    assert "--strip-components=2" in result.stdout.splitlines()
    assert f"TensorRT-{release}/include" in result.stdout.splitlines()
    assert github_env.read_text() == (
        f"CPLUS_INCLUDE_PATH={tmp_path}/tensorrt-headers:/existing/includes\n"
    )


def test_gpu_ci_runs_complete_suite_and_cleans_runner() -> None:
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    gpu_job = workflow["jobs"]["gpu-tests"]
    ordered_stages = (
        "Verify NVIDIA driver",
        "Install CUDA, export, and test dependencies",
        "Install native build prerequisites",
        "Configure Python CUDA toolchain",
        "Verify GPU and TensorRT development headers",
        "Rebuild native plugins",
        "Inspect plugin linkage",
        "Run complete GPU test suite",
        "Require all supported GPU tests to run",
        "Build platform wheel",
        "Verify and smoke-test installed wheel",
        "Upload wheel",
    )
    positions = [
        gpu_job["steps"].index(get_named_step(gpu_job, name)) for name in ordered_stages
    ]
    assert positions == sorted(positions)
    temporary_paths_step = get_named_step(gpu_job, "Configure temporary paths")
    assert_mandatory_step(temporary_paths_step)
    assert [
        line.strip()
        for line in temporary_paths_step["run"].splitlines()
        if line.strip()
    ] == [
        "{",
        'echo "CUPY_CACHE_DIR=${RUNNER_TEMP}/cupy-cache-${GITHUB_RUN_ID}-'
        '${GITHUB_RUN_ATTEMPT}"',
        'echo "UV_CACHE_DIR=${RUNNER_TEMP}/uv-cache-${GITHUB_RUN_ID}-'
        '${GITHUB_RUN_ATTEMPT}"',
        'echo "UV_PROJECT_ENVIRONMENT=${RUNNER_TEMP}/fast-gpu-asr-${GITHUB_RUN_ID}-'
        '${GITHUB_RUN_ATTEMPT}"',
        '} >> "${GITHUB_ENV}"',
    ]
    assert_required_run_step(
        gpu_job,
        "Install CUDA, export, and test dependencies",
        "uv sync --frozen --extra dev",
    )
    gpu_test_step = assert_required_run_step(
        gpu_job,
        "Run complete GPU test suite",
        'uv run --frozen pytest -q --junitxml="${RUNNER_TEMP}/gpu-test-results.xml"',
    )
    assert_pytest_is_not_filtered(workflow, gpu_job, gpu_test_step)

    build_step = get_named_step(gpu_job, "Build platform wheel")
    assert_mandatory_step(build_step)
    build_script = build_step["run"]
    assert [line.strip() for line in build_script.splitlines() if line.strip()] == [
        "rm -rf dist",
        "scripts/build_wheel.sh dist",
    ]

    cleanup_step = get_named_step(gpu_job, "Clean GPU runner artifacts")
    assert cleanup_step.get("if") == "always()"
    assert cleanup_step.get("continue-on-error") in (None, False)
    assert cleanup_step.get("shell") in (None, "bash")
    assert gpu_job["steps"][-1] is cleanup_step
    cleanup_script = re.sub(r"\\\n\s*", " ", cleanup_step["run"])
    cleanup_commands = [
        " ".join(line.split()) for line in cleanup_script.splitlines() if line.strip()
    ]
    [recursive_removal] = [
        command for command in cleanup_commands if command.startswith("rm -rf ")
    ]
    assert {
        "${CUPY_CACHE_DIR}",
        "${UV_CACHE_DIR}",
        "${UV_PROJECT_ENVIRONMENT}",
        "${RUNNER_TEMP}/wheel-site",
        "${RUNNER_TEMP}/tensorrt-headers",
        "${RUNNER_TEMP}/tensorrt-headers.tar.gz",
        "dist",
    } <= set(shlex.split(recursive_removal)[2:])
    assert "rm -f src/fast_gpu_asr/tensorrt_plugins/*.so" in cleanup_commands


@pytest.mark.parametrize(
    ("capability", "report", "error"),
    (
        (
            "90",
            '<testsuites><testsuite skipped="0"/><testsuite skipped="0"/></testsuites>',
            None,
        ),
        (
            "90",
            '<testsuites><testsuite skipped="0"/><testsuite skipped="2"/></testsuites>',
            "2 skipped",
        ),
        ("75", '<testsuite skipped="1"/>', "1 skipped"),
        ("75", '<testsuite skipped="0"/>', None),
        (
            "75",
            '<testsuite skipped="1"><testcase><skipped message="'
            "A working SM80 or newer CUDA device is required."
            '"/></testcase></testsuite>',
            None,
        ),
        (
            "80",
            '<testsuite skipped="1"><testcase><skipped message="'
            "A working SM80 or newer CUDA device is required."
            '"/></testcase></testsuite>',
            "1 skipped",
        ),
        (
            "75",
            '<testsuite skipped="1"><testcase><skipped message="missing nvcc"/>'
            "</testcase></testsuite>",
            "1 skipped",
        ),
        (
            "75",
            '<testsuites><testsuite skipped="1"><testcase><skipped message="'
            "A working SM80 or newer CUDA device is required."
            '"/></testcase></testsuite><testsuite skipped="1"><testcase>'
            '<skipped message="missing nvcc"/></testcase></testsuite></testsuites>',
            "2 skipped, 1 expected",
        ),
        ("74", '<testsuite skipped="0"/>', "requires SM75"),
        ("90", None, "FileNotFoundError"),
    ),
)
def test_gpu_ci_validates_test_reports(
    tmp_path: Path, capability: str, report: str | None, error: str | None
) -> None:
    gpu_job = load_yaml_mapping(WORKFLOW_PATH)["jobs"]["gpu-tests"]
    step = get_named_step(gpu_job, "Require all supported GPU tests to run")
    assert_mandatory_step(step)
    if report is not None:
        (tmp_path / "gpu-test-results.xml").write_text(report, encoding="utf8")
    script = (
        "import sys\nfrom types import SimpleNamespace\n"
        "sys.modules['cupy'] = SimpleNamespace(cuda=SimpleNamespace("
        f"Device=lambda: SimpleNamespace(compute_capability='{capability}')))\n"
        + get_python_heredoc(step["run"])
    )
    result = subprocess.run(
        (sys.executable, "-I", "-c", script),
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
    )
    if error is None:
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
    else:
        assert result.returncode != 0
        assert error in result.stderr


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        ("./.github/actions/local", True),
        ("actions/checkout@" + "a" * 40, True),
        ("docker://python@sha256:" + "b" * 64, True),
        ("actions/checkout@main", False),
        ("actions/checkout@" + "a" * 39, False),
        ("docker://python:latest", False),
        ("docker://python@sha256:" + "b" * 63, False),
        ("./../outside-workspace", False),
        ("actions/checkout@" + "A" * 40, True),
        ("actions/checkout@" + "a" * 40 + " trailing", False),
    ),
)
def test_action_reference_immutability_classifier(
    reference: str, expected: bool
) -> None:
    assert is_immutable_action_reference(reference) is expected


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        ("actions/checkout@" + "a" * 40, True),
        ("Actions/Checkout@" + "a" * 40, True),
        ("actions/checkout-extra@" + "a" * 40, False),
        ("./.github/actions/checkout", False),
    ),
)
def test_checkout_action_reference_classifier(reference: str, expected: bool) -> None:
    assert is_checkout_action_reference(reference) is expected


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        ("ubuntu@sha256:" + "a" * 64, True),
        ("registry.example.com:5000/image@sha256:" + "b" * 64, True),
        ("ubuntu:latest", False),
        ("ubuntu@sha256:" + "a" * 63, False),
        ("ubuntu@sha256:" + "A" * 64, False),
    ),
)
def test_container_image_immutability_classifier(
    reference: str, expected: bool
) -> None:
    assert is_immutable_container_image(reference) is expected


def test_ci_runs_hosted_tests_for_every_supported_python() -> None:
    workflow = load_yaml_mapping(WORKFLOW_PATH)
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf8"))
    python_job = workflow["jobs"]["python-tests"]
    gpu_job = workflow["jobs"]["gpu-tests"]

    assert python_job["strategy"]["fail-fast"] is False
    tested_versions = python_job["strategy"]["matrix"]["python-version"]
    assert isinstance(tested_versions, list)
    assert all(isinstance(version, str) for version in tested_versions)
    assert tested_versions
    assert len(tested_versions) == len(set(tested_versions))

    project = pyproject["project"]
    classifier_prefix = "Programming Language :: Python :: "
    classified_versions = [
        match.group(1)
        for classifier in project["classifiers"]
        if (match := re.fullmatch(classifier_prefix + r"(\d+\.\d+)", classifier))
    ]
    assert len(classified_versions) == len(set(classified_versions))
    assert set(tested_versions) == set(classified_versions)

    supported_versions = sorted(Version(version) for version in tested_versions)
    version_pairs = [(version.major, version.minor) for version in supported_versions]
    assert len({major for major, _ in version_pairs}) == 1
    assert all(
        current == (previous[0], previous[1] + 1)
        for previous, current in zip(version_pairs, version_pairs[1:], strict=False)
    )
    requires_python = SpecifierSet(project["requires-python"])
    first_major, first_minor = version_pairs[0]
    last_major, last_minor = version_pairs[-1]
    assert requires_python == SpecifierSet(
        f">={first_major}.{first_minor},<{last_major}.{last_minor + 1}"
    )
    assert pyproject["tool"]["ruff"]["target-version"] == (
        f"py{first_major}{first_minor}"
    )

    assert re.fullmatch(r"ubuntu-(?:latest|\d{2}\.\d{2})", python_job["runs-on"])
    timeout = python_job.get("timeout-minutes")
    assert type(timeout) is int and 0 < timeout <= 30
    for name, expected_script in {
        "Install development environment": "uv sync --frozen --extra dev",
        "Compile Python sources": (
            "uv run --frozen python -m compileall -q src tests scripts"
        ),
        "Run CPU test suite": "uv run --frozen pytest -q",
    }.items():
        assert_required_run_step(python_job, name, expected_script)
    assert_pytest_is_not_filtered(
        workflow, python_job, get_named_step(python_job, "Run CPU test suite")
    )

    needs = gpu_job["needs"]
    if isinstance(needs, str):
        needs = [needs]
    assert isinstance(needs, list) and all(isinstance(job, str) for job in needs)
    assert set(needs) >= {"quality", "python-tests"}


def test_ci_actions_and_containers_are_immutable() -> None:
    workflow_paths = repository_workflow_paths()
    assert WORKFLOW_PATH in workflow_paths
    local_actions = {
        *LOCAL_ACTION_DIRECTORY.rglob("action.yml"),
        *LOCAL_ACTION_DIRECTORY.rglob("action.yaml"),
    }
    pending_scopes = [
        (path, job)
        for path in workflow_paths
        for job in load_yaml_mapping(path)["jobs"].values()
    ] + [(path, load_yaml_mapping(path)["runs"]) for path in sorted(local_actions)]
    action_references = []
    while pending_scopes:
        source, scope = pending_scopes.pop()
        images = [service["image"] for service in scope.get("services", {}).values()]
        container = scope.get("container")
        if container is not None:
            images.append(
                container if isinstance(container, str) else container["image"]
            )
        for image in images:
            assert is_immutable_container_image(image), (source, image)

        steps = scope.get("steps", [])
        assert isinstance(steps, list) and all(isinstance(step, dict) for step in steps)
        references = [step["uses"] for step in (scope, *steps) if "uses" in step]
        image = scope.get("image")
        if isinstance(image, str) and image.startswith("docker://"):
            references.append(image)
        for reference in references:
            assert is_immutable_action_reference(reference), (source, reference)
            action_references.append(reference)
            if not reference.startswith("./"):
                continue
            local_path = (REPOSITORY_ROOT / reference[2:]).resolve()
            assert local_path.is_relative_to(REPOSITORY_ROOT), (source, reference)
            assert local_path.exists(), (source, reference)
            if local_path.is_dir():
                [definition] = [
                    path
                    for path in (local_path / "action.yml", local_path / "action.yaml")
                    if path.is_file()
                ]
                if definition not in local_actions:
                    local_actions.add(definition)
                    pending_scopes.append(
                        (definition, load_yaml_mapping(definition)["runs"])
                    )
            else:
                assert local_path.parent == WORKFLOW_DIRECTORY
                assert local_path.suffix in (".yml", ".yaml")

    assert action_references


def test_generated_plugin_artifacts_are_ignored_and_untracked() -> None:
    generated_paths = (
        *(
            f"src/fast_gpu_asr/tensorrt_plugins/{library_name}"
            for library_name, _ in PLUGIN_INITIALIZERS
        ),
        "src/fast_gpu_asr/tensorrt_plugins/.plugin-build-example/example_plugin.so",
        "src/fast_gpu_asr/tensorrt_plugins/.plugin-build-example/build.log",
    )
    ignored_artifacts = subprocess.run(
        ("git", "check-ignore", "--no-index", "--verbose", "--stdin"),
        cwd=REPOSITORY_ROOT,
        input="\n".join(generated_paths) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ignored_artifacts.returncode == 0, ignored_artifacts.stderr
    ignored_files = {
        name: rule.split(":", 1)[0]
        for rule, name in (
            line.split("\t", 1) for line in ignored_artifacts.stdout.splitlines()
        )
    }
    assert ignored_files == dict.fromkeys(generated_paths, ".gitignore")

    versioned_paths = {
        *(
            f"src/fast_gpu_asr/tensorrt_plugins/{source_name}"
            for source_name, _ in PLUGIN_BUILDS
        ),
        *(
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in PLUGIN_DIRECTORY.glob("*.h")
        ),
    }
    assert versioned_paths
    ignored_sources = subprocess.run(
        ("git", "check-ignore", "--no-index", "--stdin"),
        cwd=REPOSITORY_ROOT,
        input="\n".join(sorted(versioned_paths)) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ignored_sources.returncode == 1, ignored_sources.stderr
    assert ignored_sources.stdout == ""

    tracked_sources = subprocess.run(
        ("git", "ls-files", "--", *sorted(versioned_paths)),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert set(tracked_sources.stdout.splitlines()) == versioned_paths

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
        timeout=30,
    )
    assert tracked.stdout == ""
