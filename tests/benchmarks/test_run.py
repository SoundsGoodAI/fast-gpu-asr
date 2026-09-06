#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU regression tests for benchmark matrix orchestration."""

import json
import subprocess
import sys
from hashlib import sha256
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO
from unittest.mock import Mock, call

import pytest

from benchmarks import run
from benchmarks.common import BATCHES, GPUS, MODELS, PRECISIONS, file_hash, write_json

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Supply valid CLI arguments without creating files or querying hardware.

    Parameters
    ----------
    tmp_path : Path
        Root for campaign, checkpoint, and output paths.
    monkeypatch : pytest.MonkeyPatch
        Temporarily replaces ``sys.argv``.

    Returns
    -------
    list[str]
        Mutable ``sys.argv`` list with all required checkpoint paths and one
        selected model, precision, and batch size.
    """

    options = {
        "campaign": tmp_path / "campaign.json",
        "datasets-root": tmp_path,
        "scorer": tmp_path,
        "output": tmp_path / "runs",
        "gpu": "H100",
        "models": "zipformer_rnnt",
        "batches": "1",
        "precisions": "fp16",
    }
    for model in MODELS:
        filename = "model.pt" if model.startswith("zipformer") else f"{model}.nemo"
        options[model] = tmp_path / f"{model} with spaces" / filename
    argv = ["run", *[arg for k, v in options.items() for arg in (f"--{k}", str(v))]]
    monkeypatch.setattr(sys, "argv", argv)
    return argv


@pytest.fixture
def matrix_run(
    cli: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Prepare real files and hashes with simulated hardware and subprocesses.

    Parameters
    ----------
    cli : list[str]
        Arguments parsed once; subsequent parser calls return the same namespace.
    tmp_path : Path
        Root for tiny checkpoint, sidecar, plugin, and benchmark artifacts.
    monkeypatch : pytest.MonkeyPatch
        Replaces campaign loading, hardware probes, parsing, and process calls.

    Returns
    -------
    SimpleNamespace
        Mutable ``args``, model-keyed ``checkpoints``, frozen ``campaign`` hashes,
        subprocess ``process`` mock, ``output`` directory, and optional CPU
        ``quota`` file path. Orchestration and artifact I/O remain real.
    """

    args = run.parse_args()
    monkeypatch.setattr(run, "parse_args", lambda: args)
    checkpoints = {model: getattr(args, model) for model in MODELS}
    models = {}
    for model, checkpoint in checkpoints.items():
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(model.encode())
        if model.startswith("zipformer"):
            checkpoint.with_name("config.yaml").write_text("config")
            checkpoint.with_name("bpe.model").write_bytes(b"tokenizer")
        models[model] = {
            "files": {p.name: file_hash(p) for p in checkpoint.parent.iterdir()}
        }
    campaign = {"id": "campaign", "models": models}
    monkeypatch.setattr(run, "load_campaign", Mock(return_value=campaign))
    monkeypatch.setattr(run, "check_gpu", Mock())
    monkeypatch.setattr(
        run,
        "nvidia_query",
        Mock(
            side_effect=[
                [["NVIDIA H100", GPU]],
                [["NVIDIA H100", GPU, "81920", "580", "[N/A]", "Default", "Disabled"]],
            ]
        ),
    )
    runtime = SimpleNamespace(
        runtimeGetVersion=lambda: 13030, driverGetVersion=lambda: 13030
    )
    monkeypatch.setattr(
        run, "cp", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    )
    monkeypatch.setattr(
        run.importlib.metadata, "version", lambda name: f"{name}-version"
    )
    monkeypatch.setattr(run.os, "sched_getaffinity", lambda pid: {4, 2})
    quota = tmp_path / "cpu.max"
    monkeypatch.setattr(
        run,
        "Path",
        lambda path: quota if path == "/sys/fs/cgroup/cpu.max" else Path(path),
    )
    monkeypatch.setattr(subprocess, "check_output", Mock(return_value='{"cpus": []}'))
    root = tmp_path / "source"
    plugins = root / "src/fast_gpu_asr/tensorrt_plugins"
    plugins.mkdir(parents=True)
    (plugins / "feature.so").write_bytes(b"plugin")
    monkeypatch.setattr(run, "ROOT", root)

    def execute(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
        stdout: TextIO,
        stderr: int,
        check: bool,
    ) -> None:
        """Simulate export, collection, or scoring without running external code.

        Export writes a placeholder bundle, collection refreshes the hardware
        snapshot, and scoring marks the result complete without computing metrics.

        Parameters
        ----------
        command : list[str]
            Python module invocation with output or run-directory arguments.
        env : dict[str, str]
            Process environment; must select the expected physical GPU UUID.
        cwd : Path
            Must be the fixture's source root.
        stdout : TextIO
            Open stage log receiving diagnostic text.
        stderr : int
            Must merge errors into the log via ``subprocess.STDOUT``.
        check : bool
            Must be true so real subprocess failures would propagate.
        """

        assert env["CUDA_VISIBLE_DEVICES"] == GPU
        assert cwd == root and stderr == subprocess.STDOUT and check
        stdout.write("stage diagnostics\n")
        if command[2].startswith("fast_gpu_asr.export."):
            bundle = Path(command[command.index("--output-dir") + 1])
            (bundle / "metadata").mkdir(parents=True)
            (bundle / "encoder.trt").write_bytes(b"engine")
            (bundle / "metadata/model_config.yaml").write_text("config")
        else:
            directory = Path(command[command.index("--run-dir") + 1])
            machine = json.loads((directory / "hardware.json").read_text())
            if command[2] == "benchmarks.collect":
                machine["cpu_quota"] = "100000 100000"
                write_json(directory / "hardware.json", machine)
            elif command[2] == "benchmarks.score":
                result = json.loads((directory / "result.json").read_text())
                result.update(status="complete", hardware=machine)
                write_json(directory / "result.json", result)
            else:
                pytest.fail(f"Unexpected subprocess: {command}")

    process = Mock(side_effect=execute)
    monkeypatch.setattr(subprocess, "run", process)
    return SimpleNamespace(
        args=args,
        checkpoints=checkpoints,
        campaign=campaign,
        process=process,
        output=args.output,
        quota=quota,
    )


@pytest.mark.parametrize("discard", (False, True))
def test_matrix_runs_all_stages_in_order(
    matrix_run: SimpleNamespace, discard: bool
) -> None:
    args = matrix_run.args
    args.models, args.precisions, args.batches = list(MODELS), list(PRECISIONS), [1, 2]
    args.discard_engines = discard

    run.main()

    configurations = list(product(args.models, args.precisions, args.batches))
    names = [f"H100-{m}-{p}-b{b}" for m, p, b in configurations]
    assert json.loads((args.output / "matrix.json").read_text()) == names
    calls = matrix_run.process.call_args_list
    assert len(calls) == 3 * len(names)
    for index, (model, precision, batch) in enumerate(configurations):
        directory = args.output / names[index]
        spec = json.loads((directory / "run.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        assert result["status"] == "complete", result
        assert result["spec"] == spec
        assert (spec["model"], spec["precision"], spec["batch_size"], spec["beam"]) == (
            model,
            precision,
            batch,
            6,
        )
        assert result["hardware"]["cpu_quota"] == "100000 100000"
        export = calls[3 * index].args[0]
        family = model.split("_")[0]
        assert export[:3] == [
            sys.executable,
            "-m",
            f"fast_gpu_asr.export.export_{family}",
        ]
        root = Path(run.__file__).resolve().parents[1]
        assert (root / f"src/fast_gpu_asr/export/export_{family}.py").is_file()
        assert dict(zip(export[3::2], export[4::2], strict=True)) == {
            "--model-path": str(matrix_run.checkpoints[model]),
            "--output-dir": str(directory / "bundle"),
            "--batch-size": str(batch),
            "--beam": "6",
            "--decoder-type": "transducer_modified_beam_search",
            "--encoder-precision": precision,
            "--decoder-precision": precision,
            "--min-audio-seconds": "0.1",
            "--opt-audio-seconds": "8",
            "--max-audio-seconds": "40",
            "--optimization-level": "5",
        }
        assert json.loads((directory / "export-command.json").read_text()) == export
        for offset, (stage, option, path) in enumerate(
            (
                ("collect", "datasets-root", args.datasets_root),
                ("score", "scorer", args.scorer),
            ),
            1,
        ):
            command = calls[3 * index + offset].args[0]
            assert command[:3] == [sys.executable, "-m", f"benchmarks.{stage}"]
            assert dict(zip(command[3::2], command[4::2], strict=True)) == {
                "--campaign": str(args.campaign),
                "--run-dir": str(directory),
                f"--{option}": str(path),
            }
        for stage in ("build", "collect", "score"):
            assert (directory / f"{stage}.log").read_text() == "stage diagnostics\n"
        assert json.loads((directory / "bundle-hashes.json").read_text()) == {
            "encoder.trt": sha256(b"engine").hexdigest(),
            "metadata/model_config.yaml": sha256(b"config").hexdigest(),
        }
        assert json.loads((directory / "plugin-hashes.json").read_text()) == {
            "feature.so": sha256(b"plugin").hexdigest(),
        }
        assert (directory / "bundle").exists() is not discard
    assert run.load_campaign.call_args_list == [
        call(args.campaign, check_source=True)
    ] * (len(names) + 1)
    assert all(p.is_file() for p in matrix_run.checkpoints.values())


@pytest.mark.parametrize("model", ("zipformer_rnnt", "parakeet_v3"))
def test_beam_one_uses_greedy(
    matrix_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    matrix_run.args.models = [model]
    monkeypatch.setitem(run.BEAMS, model, 1)
    run.main()
    command = matrix_run.process.call_args_list[0].args[0]
    assert command[command.index("--beam") + 1] == "1"
    assert command[command.index("--decoder-type") + 1] == "transducer_greedy_search"


@pytest.mark.parametrize("quota", (None, "max 100000"))
def test_failed_build_retains_hardware_metadata(
    matrix_run: SimpleNamespace, quota: str | None
) -> None:
    if quota is not None:
        matrix_run.quota.write_text(quota + "\n")
    matrix_run.process.side_effect = subprocess.CalledProcessError(1, "export")
    run.main()
    directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b1"
    result = json.loads((directory / "result.json").read_text())
    machine = json.loads((directory / "hardware.json").read_text())
    assert result["status"] == "failed" and result["stage"] == "build"
    assert result["hardware"] == machine
    assert machine["gpu"]["uuid"] == GPU
    assert machine["gpu"]["power.limit"] == "[N/A]"
    assert machine["cuda_runtime"] == machine["cuda_driver"] == 13030
    assert machine["cpu_affinity"] == [2, 4]
    assert machine["cpu_quota"] == quota
    assert machine["packages"]["cupy-cuda13x"] == "cupy-cuda13x-version"
    assert run.nvidia_query.call_args.args[-1] == GPU
    matrix_run.process.assert_called_once()
    assert not (directory / "collection.json").exists()


@pytest.mark.parametrize("stage", ("build", "collect", "score"))
@pytest.mark.parametrize("interrupted", (False, True))
def test_stage_failure_preserves_evidence_and_cleans_only_bundles(
    matrix_run: SimpleNamespace,
    stage: str,
    interrupted: bool,
) -> None:
    matrix_run.args.batches = [1, 2]
    matrix_run.args.discard_engines = True
    execute = matrix_run.process.side_effect
    failed_call = ("build", "collect", "score").index(stage) + 1

    def fail_stage(*args: object, **kwargs: object) -> None:
        """Run the stage stub, then fail once at the selected process call.

        Artifacts are written before raising ``KeyboardInterrupt`` or
        ``CalledProcessError``, as selected by ``interrupted``.

        Parameters
        ----------
        *args : object
            Positional subprocess arguments forwarded unchanged to ``execute``.
        **kwargs : object
            Keyword subprocess arguments forwarded unchanged to ``execute``.
        """

        execute(*args, **kwargs)
        if matrix_run.process.call_count == failed_call:
            if interrupted:
                raise KeyboardInterrupt("interrupted")
            raise subprocess.CalledProcessError(1, stage)

    matrix_run.process.side_effect = fail_stage
    if interrupted:
        with pytest.raises(KeyboardInterrupt, match="interrupted"):
            run.main()
    else:
        run.main()

    directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b1"
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == "failed" and result["stage"] == stage
    assert result["hardware"] == json.loads((directory / "hardware.json").read_text())
    error = "KeyboardInterrupt" if interrupted else "CalledProcessError"
    assert error in result["error"] and error in result["traceback"]
    assert result["diagnostics"] == "stage diagnostics\n"
    assert not (directory / "bundle").exists()
    assert (directory / "bundle-hashes.json").exists() is (stage != "build")
    assert all(p.is_file() for p in matrix_run.checkpoints.values())
    assert matrix_run.process.call_count == failed_call + (0 if interrupted else 3)
    next_directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b2"
    if interrupted:
        assert not next_directory.exists()
    else:
        assert (
            json.loads((next_directory / "result.json").read_text())["status"]
            == "complete"
        )
    assert len(json.loads((matrix_run.output / "matrix.json").read_text())) == 2


@pytest.mark.parametrize("model", MODELS)
def test_rejects_unverified_checkpoint_path(
    matrix_run: SimpleNamespace, model: str
) -> None:
    matrix_run.args.models = [model]
    checkpoint = matrix_run.checkpoints[model]
    other = checkpoint.with_name("other" + checkpoint.suffix)
    other.write_bytes(b"unverified checkpoint")
    setattr(matrix_run.args, model, other)
    with pytest.raises(ValueError, match="not the frozen model file"):
        run.main()
    run.nvidia_query.assert_not_called()
    matrix_run.process.assert_not_called()
    assert not matrix_run.output.exists()


@pytest.mark.parametrize("filename", ("model.pt", "config.yaml", "bpe.model"))
@pytest.mark.parametrize("during_export", (False, True))
def test_rejects_changed_checkpoint_or_sidecar(
    matrix_run: SimpleNamespace, filename: str, during_export: bool
) -> None:
    path = matrix_run.checkpoints["zipformer_rnnt"].with_name(filename)
    if during_export:
        execute = matrix_run.process.side_effect

        def change_file(*args: object, **kwargs: object) -> None:
            """Change the selected checkpoint or sidecar before export returns.

            Parameters
            ----------
            *args : object
                Positional subprocess arguments forwarded to the stage stub.
            **kwargs : object
                Keyword subprocess arguments forwarded to the stage stub.
            """

            execute(*args, **kwargs)
            path.write_bytes(b"changed")

        matrix_run.process.side_effect = change_file
        run.main()
        directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b1"
        result = json.loads((directory / "result.json").read_text())
        assert result["status"] == "failed" and result["stage"] == "build"
        assert "Checkpoint changed" in result["error"]
        matrix_run.process.assert_called_once()
    else:
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="Checkpoint changed"):
            run.main()
        run.nvidia_query.assert_not_called()
        assert not matrix_run.output.exists()


def test_replaced_campaign_aborts_matrix(matrix_run: SimpleNamespace) -> None:
    run.load_campaign.side_effect = [
        matrix_run.campaign,
        {**matrix_run.campaign, "id": "replacement"},
    ]
    with pytest.raises(ValueError, match="Campaign changed"):
        run.main()
    matrix_run.process.assert_not_called()
    assert list(matrix_run.output.iterdir()) == [matrix_run.output / "matrix.json"]


def test_existing_output_is_preserved(matrix_run: SimpleNamespace) -> None:
    matrix_run.output.mkdir()
    sentinel = matrix_run.output / "matrix.json"
    sentinel.write_text("previous results")
    with pytest.raises(FileExistsError):
        run.main()
    assert sentinel.read_text() == "previous results"
    matrix_run.process.assert_not_called()


def test_wrong_gpu_is_rejected(matrix_run: SimpleNamespace) -> None:
    run.nvidia_query.side_effect = [[["NVIDIA H200", GPU]]]
    with pytest.raises(ValueError, match="Requested H100, found NVIDIA H200"):
        run.main()
    matrix_run.process.assert_not_called()
    assert not matrix_run.output.exists()


def test_gpu_contention_before_export_is_recorded(matrix_run: SimpleNamespace) -> None:
    matrix_run.args.batches = [1, 2]
    matrix_run.args.discard_engines = True
    run.check_gpu.side_effect = [None, RuntimeError("Competing workload"), None]
    run.main()
    directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b1"
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == "failed" and result["stage"] == "build"
    assert result["error"] == "RuntimeError: Competing workload"
    assert result["diagnostics"] == ""
    assert not (directory / "build.log").exists()
    assert matrix_run.process.call_count == 3
    next_directory = matrix_run.output / "H100-zipformer_rnnt-fp16-b2"
    assert (
        json.loads((next_directory / "result.json").read_text())["status"] == "complete"
    )


@pytest.mark.parametrize("gpu", GPUS)
def test_default_batches_include_full_grid(cli: list[str], gpu: str) -> None:
    index = cli.index("--batches")
    del cli[index : index + 2]
    cli[cli.index("--gpu") + 1] = gpu
    assert run.parse_args().batches == list(BATCHES)


@pytest.mark.parametrize(
    "options,message",
    (
        (["--models", "parakeet_v3", "parakeet_v3"], "Duplicate"),
        (["--batches", "1", "1"], "Duplicate"),
        (["--precisions", "fp32", "fp32"], "Duplicate"),
        (["--device-id", "-1"], "non-negative"),
    ),
)
def test_invalid_matrix_is_rejected(
    cli: list[str],
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    message: str,
) -> None:
    cli.extend(options)
    with pytest.raises(SystemExit) as error:
        run.parse_args()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
