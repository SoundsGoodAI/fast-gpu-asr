#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU tests for full-corpus collection, timing, and evidence integrity."""

import json
import os
import subprocess
import sys
import wave
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call
from uuid import UUID

import numpy as np
import pytest

from benchmarks import collect
from benchmarks.common import file_hash, write_json

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def audio_campaign(
    tmp_path: Path,
) -> dict[str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]]:
    """Create real PCM16 audio with tied durations and dataset-local IDs.

    Parameters
    ----------
    tmp_path : Path
        Root containing two tiny materialized datasets.

    Returns
    -------
    dict[str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]]
        Two datasets with a partial batch, one resampled waveform, and reused IDs.
    """

    datasets: dict[str, list[dict[str, str | int]]] = {"first": [], "second": []}
    for name, identifier, frames, rate in (
        ("first", "later", 4800, 16000),
        ("first", "b", 1600, 16000),
        ("first", "a", 800, 8000),
        ("second", "a", 3200, 16000),
    ):
        path = tmp_path / name / f"{identifier}.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x20" * frames)
        datasets[name].append(
            {
                "id": identifier,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": file_hash(path),
                "frames": frames,
                "rate": rate,
                "text": "reference",
            }
        )
    return {
        "id": "campaign",
        "source": {"commit": "frozen"},
        "protocol": {**collect.PROTOCOL, "passes": 3},
        "datasets": datasets,
    }


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Provide deterministic ASR outputs and one-second measured intervals.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Supplies a deterministic clock and disables external GPU process queries.

    Returns
    -------
    Mock
        ASR stand-in; audio loading, timing, and serialization remain real.
    """

    model = Mock(
        encoder=SimpleNamespace(batch_size=2, sample_rate=16000),
        decoder=SimpleNamespace(beam=6),
        side_effect=lambda audios: (
            ["word" if audio.size == 1600 else "" for audio in audios],
            [[("word", 0.0, 0.1)] if audio.size == 1600 else [] for audio in audios],
        ),
    )
    monkeypatch.setattr(collect, "perf_counter", Mock(side_effect=count()))
    monkeypatch.setattr(collect, "check_gpu", Mock())
    return model


@pytest.fixture
def collection_run(
    tmp_path: Path,
    audio_campaign: dict[
        str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]
    ],
    model: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Prepare a run with real artifact hashes and mocked CUDA construction.

    Parameters
    ----------
    tmp_path : Path
        Root holding dataset files, fake build artifacts, and collection output.
    audio_campaign : dict[
        str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]
    ]
        Tiny dataset suite used by the real measurement loop.
    model : Mock
        Deterministic ASR stand-in.
    monkeypatch : pytest.MonkeyPatch
        Replaces CUDA, external queries, source validation, and CLI arguments.

    Returns
    -------
    SimpleNamespace
        Output paths, run metadata, and dependency mocks.
    """

    directory = tmp_path / "run"
    bundle = directory / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "encoder.trt").write_bytes(b"test engine")
    plugins = tmp_path / "src/fast_gpu_asr/tensorrt_plugins"
    plugins.mkdir(parents=True)
    (plugins / "feature.so").write_bytes(b"test plugin")
    for name, root in (("bundle", bundle), ("plugin", plugins)):
        write_json(
            directory / f"{name}-hashes.json",
            {path.name: file_hash(path) for path in root.iterdir()},
        )
    spec = {
        "campaign_id": audio_campaign["id"],
        "gpu_uuid": GPU,
        "batch_size": 2,
        "beam": 6,
    }
    write_json(directory / "run.json", spec)
    write_json(directory / "hardware.json", {"orchestrator": True})
    gpu_query = Mock(
        return_value=[
            ["NVIDIA H100", GPU, "81920", "580", "[N/A]", "Default", "Disabled"]
        ]
    )
    campaign = Mock(return_value=audio_campaign)
    factory = Mock(return_value=model)
    properties = Mock(return_value={"uuid": UUID(GPU[4:]).bytes})
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    monkeypatch.setattr(collect, "nvidia_query", gpu_query)
    monkeypatch.setattr(
        collect.importlib.metadata, "version", lambda name: f"{name}-version"
    )
    monkeypatch.setattr(collect, "python_version", lambda: "3.14.5")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 2})
    monkeypatch.setattr(subprocess, "check_output", Mock(return_value='{"cpus": []}'))
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000\n")
    monkeypatch.setattr(
        collect,
        "Path",
        lambda path: quota if path == "/sys/fs/cgroup/cpu.max" else Path(path),
    )
    monkeypatch.setattr(collect, "load_campaign", campaign)
    monkeypatch.setattr(
        collect,
        "fast_gpu_asr",
        SimpleNamespace(ASR=factory, __file__=str(plugins.parent / "__init__.py")),
    )
    monkeypatch.setattr(
        collect,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    getDeviceProperties=properties,
                    runtimeGetVersion=lambda: 13020,
                    driverGetVersion=lambda: 13030,
                )
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--campaign",
            str(tmp_path / "campaign.json"),
            "--datasets-root",
            str(tmp_path),
            "--run-dir",
            str(directory),
        ],
    )
    return SimpleNamespace(
        directory=directory,
        plugins=plugins,
        spec=spec,
        gpu_query=gpu_query,
        quota=quota,
        campaign=campaign,
        factory=factory,
        properties=properties,
    )


@pytest.mark.parametrize("output", ("", '123, "python, worker"\n456, process\n'))
def test_nvidia_query_parses_csv_and_limits_wait(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    query = Mock(return_value=output)
    monkeypatch.setattr(subprocess, "check_output", query)

    assert collect.nvidia_query("compute-apps", "pid,process_name", GPU) == (
        [["123", "python, worker"], ["456", "process"]] if output else []
    )
    query.assert_called_once_with(
        [
            "nvidia-smi",
            f"--id={GPU}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "error",
    (
        FileNotFoundError(),
        subprocess.CalledProcessError(1, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 30),
    ),
)
def test_nvidia_query_propagates_errors(monkeypatch: pytest.MonkeyPatch, error) -> None:
    monkeypatch.setattr(subprocess, "check_output", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        collect.nvidia_query("gpu", "name", GPU)
    assert caught.value is error


@pytest.mark.parametrize("case", ("empty", "self", "competitor", "mps", "mps-env"))
def test_check_gpu_requires_an_uncontended_device(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    for name in os.environ:
        if name.startswith("CUDA_MPS_"):
            monkeypatch.delenv(name)
    processes = {
        "empty": [],
        "self": [[str(os.getpid()), "python"]],
        "competitor": [[str(os.getpid()), "python"], [str(os.getpid() + 1), "python"]],
        "mps": [[str(os.getpid()), "nvidia-cuda-mps-server"]],
        "mps-env": [],
    }
    monkeypatch.setattr(collect, "nvidia_query", Mock(return_value=processes[case]))
    if case == "mps-env":
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "")
    if case in ("empty", "self"):
        collect.check_gpu(GPU)
    else:
        with pytest.raises(RuntimeError, match="MPS"):
            collect.check_gpu(GPU)
    collect.nvidia_query.assert_called_once_with(
        "compute-apps", "pid,process_name", GPU
    )


@pytest.mark.parametrize(
    ("has_quota", "mig_mode"), ((False, "Disabled"), (True, "Enabled"))
)
def test_main_records_gpu_and_host_metadata(
    collection_run: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    has_quota: bool,
    mig_mode: str,
) -> None:
    run = collection_run
    run.gpu_query.return_value[0][-1] = mig_mode
    if has_quota:
        run.quota.write_text("200000 100000\n")
    else:
        run.quota.unlink()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    collect.main()

    result = json.loads((run.directory / "hardware.json").read_text(encoding="utf-8"))
    assert result["gpu"] == {
        "name": "NVIDIA H100",
        "uuid": GPU,
        "memory.total": "81920",
        "driver_version": "580",
        "power.limit": "[N/A]",
        "compute_mode": "Default",
        "mig.mode.current": mig_mode,
    }
    run.gpu_query.assert_called_once_with(
        "gpu",
        "name,uuid,memory.total,driver_version,power.limit,compute_mode,mig.mode.current",
        GPU,
    )
    assert result["cuda_runtime"] == 13020
    assert result["cuda_driver"] == 13030
    assert result["python"] == "3.14.5"
    assert result["cpu_affinity"] == [2, 4]
    assert result["cpu_quota"] == ("200000 100000" if has_quota else None)
    assert result["thread_environment"] == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": None,
    }
    assert json.loads(result["lscpu"]) == {"cpus": []}
    assert result["packages"] == {
        name: f"{name}-version"
        for name in ("cupy-cuda13x", "tensorrt-cu13", "torch", "numpy", "cuda-bindings")
    }


@pytest.mark.parametrize("passes", (1, 2, 3, 5))
def test_main_excludes_pending_work_and_includes_completion(
    collection_run: SimpleNamespace,
    model: Mock,
    monkeypatch: pytest.MonkeyPatch,
    passes: int,
) -> None:
    collection_run.campaign.return_value["protocol"]["passes"] = passes
    events = Mock()
    events.clock.side_effect = count(step=5)
    events.attach_mock(model, "model")
    events.attach_mock(collect.check_gpu, "check_gpu")
    model.stream.synchronize = events.synchronize
    monkeypatch.setattr(collect, "perf_counter", events.clock)

    collect.main()

    timed_call = [
        call.synchronize(),
        call.clock(),
        call.model(ANY),
        call.synchronize(),
        call.clock(),
    ]
    expected = (
        [call.check_gpu(GPU)]
        + [call.model(ANY), call.synchronize()] * 5
        + [call.check_gpu(GPU)]
    )
    for _ in range(passes):
        expected += timed_call * 2 + [call.check_gpu(GPU)]
        expected += timed_call + [call.check_gpu(GPU)]
    assert events.mock_calls == expected
    assert len(list(collection_run.directory.glob("pass-*"))) == passes
    assert len(
        json.loads(
            (collection_run.directory / "collection.json").read_text(encoding="utf-8")
        )["evidence"]
    ) == (passes * 3)
    for index in range(1, passes + 1):
        records = [
            json.loads(line)
            for line in (collection_run.directory / f"pass-{index}/batches.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["inference_seconds"] for r in records] == [5.0, 5.0, 5.0]


@pytest.mark.parametrize("result", (([], [[], []]), (["", ""], [])))
def test_main_rejects_incomplete_outputs(
    collection_run: SimpleNamespace,
    model: Mock,
    result: tuple[list[str], list[list[tuple[str, float, float]]]],
) -> None:
    model.side_effect = None
    model.return_value = result
    with pytest.raises(ValueError, match="incomplete batch"):
        collect.main()
    assert not (collection_run.directory / "pass-1/batches.jsonl").read_text()
    assert not (collection_run.directory / "pass-1/totals.json").exists()
    assert not (collection_run.directory / "collection.json").exists()


@pytest.mark.parametrize("elapsed", (0.0, -1.0, float("inf"), float("nan")))
def test_main_rejects_invalid_elapsed_time(
    collection_run: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    elapsed: float,
) -> None:
    monkeypatch.setattr(collect, "perf_counter", Mock(side_effect=[0, elapsed]))
    with pytest.raises(ValueError, match="Invalid inference time"):
        collect.main()
    assert not (collection_run.directory / "pass-1/batches.jsonl").read_text()
    assert not (collection_run.directory / "pass-1/totals.json").exists()
    assert not (collection_run.directory / "collection.json").exists()


def test_main_loads_normalized_audio_in_batch_order(
    tmp_path: Path,
    audio_campaign: dict[
        str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]
    ],
    collection_run: SimpleNamespace,
    model: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = Mock(wraps=collect.read_audio)
    monkeypatch.setattr(collect, "read_audio", reader)

    collect.main()

    batches = [c.args[0] for c in model.call_args_list]
    assert [[len(audio) for audio in batch] for batch in batches] == [
        [1600],
        [1600],
        [3200],
        [4800, 1599],
        [4800, 1599],
    ] + [[4800, 1599], [1600], [3200]] * 3
    for batch in batches:
        for audio in batch:
            assert audio.ndim == 1 and audio.dtype == np.float32
            np.testing.assert_array_equal(audio, 0.25)
    hashes = {
        tmp_path / row["path"]: row["sha256"]
        for rows in audio_campaign["datasets"].values()
        for row in rows
    }
    for c in reader.call_args_list:
        path, expected_hash = c.args
        assert expected_hash == hashes[path]
    assert json.loads(
        (collection_run.directory / "warmup.json").read_text(encoding="utf-8")
    ) == [
        {"dataset": "first", "ids": ["b"]},
        {"dataset": "first", "ids": ["b"]},
        {"dataset": "second", "ids": ["a"]},
        {"dataset": "first", "ids": ["later", "a"]},
        {"dataset": "first", "ids": ["later", "a"]},
    ]


@pytest.mark.parametrize("after_warmup", (False, True))
def test_main_rejects_changed_audio_before_inference(
    tmp_path: Path,
    audio_campaign: dict[
        str, str | dict[str, str] | dict[str, list[dict[str, str | int]]]
    ],
    collection_run: SimpleNamespace,
    model: Mock,
    monkeypatch: pytest.MonkeyPatch,
    after_warmup: bool,
) -> None:
    row = audio_campaign["datasets"]["first"][-1]
    audio_path = tmp_path / row["path"]

    def persist_and_corrupt(
        path: Path,
        value: dict[str, str | int | float | None | list[int] | dict[str, str | None]]
        | list[dict[str, str | list[str]]],
    ) -> None:
        """Change a waveform after the real warmup record has been saved.

        Parameters
        ----------
        path : Path
            Collector output path forwarded to the real JSON writer.
        value : dict[
            str, str | int | float | None | list[int] | dict[str, str | None]
        ] | list[dict[str, str | list[str]]]
            Hardware metadata, totals, completion marker, or warmup records,
            written without modification.
        """

        write_json(path, value)
        if path.name == "warmup.json":
            audio_path.write_bytes(b"changed")

    if after_warmup:
        monkeypatch.setattr(collect, "write_json", persist_and_corrupt)
    else:
        audio_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Audio changed"):
        collect.main()
    assert model.call_count == (5 if after_warmup else 3)
    assert (collection_run.directory / "warmup.json").exists() == after_warmup
    assert not (collection_run.directory / "collection.json").exists()


@pytest.mark.parametrize("capacity", (1, 2, 256))
def test_main_records_batches_transcripts_and_totals(
    collection_run: SimpleNamespace, model: Mock, capacity: int
) -> None:
    output = collection_run.directory
    model.encoder.batch_size = capacity
    write_json(output / "run.json", {**collection_run.spec, "batch_size": capacity})
    expected = {
        1: [
            ("first", ["later"]),
            ("first", ["a"]),
            ("first", ["b"]),
            ("second", ["a"]),
        ],
        2: [("first", ["later", "a"]), ("first", ["b"]), ("second", ["a"])],
        256: [("first", ["later", "a", "b"]), ("second", ["a"])],
    }[capacity]
    collect.main()

    assert model.call_count == 5 + 3 * len(expected)
    for index in range(1, 4):
        directory = output / f"pass-{index}"
        records = [
            json.loads(line)
            for line in (directory / "batches.jsonl").read_text().splitlines()
        ]
        assert [(r["dataset"], r["ids"]) for r in records] == expected
        assert [r["audio_seconds"] for r in records] == pytest.approx(
            {1: [0.3, 0.1, 0.1, 0.2], 2: [0.4, 0.1, 0.2], 256: [0.5, 0.2]}[capacity]
        )
        assert [r["inference_seconds"] for r in records] == [1.0] * len(expected)
        assert json.loads(
            (directory / "totals.json").read_text(encoding="utf-8")
        ) == pytest.approx(
            {
                "audio_seconds": 0.7,
                "inference_seconds": len(expected),
                "rtfx": 0.7 / len(expected),
            }
        )
        for name, ids in (("first", ["later", "a", "b"]), ("second", ["a"])):
            assert [
                json.loads(line)
                for line in (directory / f"{name}.jsonl").read_text().splitlines()
            ] == [
                {
                    "id": identifier,
                    "pred_text": "word" if identifier == "b" else "",
                    "word_timestamps": [["word", 0.0, 0.1]]
                    if identifier == "b"
                    else [],
                }
                for identifier in ids
            ]
    assert (output / "collection.json").is_file()


def test_main_rejects_nonfinite_transcript_timestamps(
    collection_run: SimpleNamespace, model: Mock
) -> None:
    model.side_effect = lambda audios: (
        ["word"] * len(audios),
        [[("word", 0.0, float("nan"))] for _ in audios],
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        collect.main()
    assert not (collection_run.directory / "pass-1/totals.json").exists()
    assert not (collection_run.directory / "collection.json").exists()


@pytest.mark.parametrize("name", ("bundle", "plugin"))
@pytest.mark.parametrize("change", ("modified", "missing", "extra", "empty-manifest"))
def test_artifact_verification_rejects_changes(
    collection_run: SimpleNamespace, name: str, change: str
) -> None:
    run = collection_run
    hashes = collect.check_artifacts(run.directory)
    assert hashes == {
        name: json.loads(
            (run.directory / f"{name}-hashes.json").read_text(encoding="utf-8")
        )
        for name in ("bundle", "plugin")
    }
    root = run.directory / "bundle" if name == "bundle" else run.plugins
    path = next(root.iterdir())
    if change == "modified":
        path.write_bytes(b"replacement")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (root / "unexpected.so").write_bytes(b"extra")
    else:
        write_json(run.directory / f"{name}-hashes.json", {})
    with pytest.raises(ValueError, match="differ from the export hashes"):
        collect.check_artifacts(run.directory)


def test_main_publishes_completion_only_after_all_passes(
    collection_run: SimpleNamespace,
) -> None:
    run = collection_run
    collect.main()

    run.factory.assert_called_once_with(run.directory / "bundle", device_id=0)
    run.properties.assert_called_once_with(0)
    assert (
        run.campaign.call_args_list
        == [call(run.directory.parent / "campaign.json", check_source=True)] * 2
    )
    marker = json.loads((run.directory / "collection.json").read_text(encoding="utf-8"))
    assert marker["campaign_id"] == "campaign"
    assert marker["source"] == {"commit": "frozen"}
    expected_paths = {
        f"pass-{i}/{name}.jsonl"
        for i in range(1, 4)
        for name in ("first", "second", "batches")
    }
    assert marker["evidence"] == {
        name: file_hash(run.directory / name) for name in expected_paths
    }


@pytest.mark.parametrize("existing", ("warmup.json", "collection.json", "pass-1"))
def test_main_refuses_existing_collection_without_writes(
    collection_run: SimpleNamespace, existing: str
) -> None:
    run = collection_run
    path = run.directory / existing
    if existing == "pass-1":
        path.mkdir()
        path = path / "batches.jsonl"
    path.write_text("existing evidence")
    before = {p: p.read_bytes() for p in run.directory.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="use a new run"):
        collect.main()
    assert {
        p: p.read_bytes() for p in run.directory.rglob("*") if p.is_file()
    } == before
    run.factory.assert_not_called()
    run.gpu_query.assert_not_called()


@pytest.mark.parametrize("failure", ("campaign", "gpu", "checkout"))
def test_main_rejects_wrong_campaign_or_gpu_before_loading(
    collection_run: SimpleNamespace, failure: str
) -> None:
    run = collection_run
    if failure == "campaign":
        write_json(run.directory / "run.json", {**run.spec, "campaign_id": "other"})
    elif failure == "gpu":
        run.properties.return_value = {"uuid": UUID(int=0).bytes}
    else:
        collect.fast_gpu_asr.__file__ = str(run.directory / "__init__.py")
    with pytest.raises(
        ValueError,
        match={
            "campaign": "different campaign",
            "gpu": "CUDA_VISIBLE_DEVICES",
            "checkout": "checkout",
        }[failure],
    ):
        collect.main()
    run.factory.assert_not_called()
    assert not (run.directory / "collection.json").exists()


@pytest.mark.parametrize("attribute", ("batch_size", "sample_rate", "beam"))
def test_main_rejects_incompatible_runtime_metadata(
    collection_run: SimpleNamespace, model: Mock, attribute: str
) -> None:
    target = model.decoder if attribute == "beam" else model.encoder
    setattr(target, attribute, 1)
    with pytest.raises(ValueError, match="differs from the run"):
        collect.main()
    model.assert_not_called()
    assert not (collection_run.directory / "collection.json").exists()


def test_inference_failure_leaves_diagnostics_without_completion(
    collection_run: SimpleNamespace, model: Mock
) -> None:
    model.side_effect = [None] * 5 + [RuntimeError("inference failed")]
    with pytest.raises(RuntimeError, match="inference failed"):
        collect.main()
    assert (collection_run.directory / "pass-1/batches.jsonl").is_file()
    assert not (collection_run.directory / "collection.json").exists()


@pytest.mark.parametrize(
    ("completed_checks", "inference_calls"),
    [(0, 0), (1, 5), (2, 7), (7, 14)],
    ids=["before-loading", "after-warmup", "after-dataset", "after-last-dataset"],
)
def test_gpu_competition_never_publishes_completion(
    collection_run: SimpleNamespace,
    model: Mock,
    completed_checks: int,
    inference_calls: int,
) -> None:
    collect.check_gpu.side_effect = [None] * completed_checks + [
        RuntimeError("competing-gpu detected")
    ]
    with pytest.raises(RuntimeError, match="competing-gpu"):
        collect.main()
    assert model.call_count == inference_calls
    assert not (collection_run.directory / "collection.json").exists()


@pytest.mark.parametrize(
    "change",
    ("source", "metadata", "artifact", "rehash", "missing-evidence", "marker-write"),
)
def test_final_integrity_failure_never_publishes_completion(
    collection_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    run = collection_run
    if change == "source":
        run.campaign.side_effect = [
            run.campaign.return_value,
            ValueError("Sources changed"),
        ]

    def persist_and_change(
        path: Path,
        value: dict[str, str | int | float | None | list[int] | dict[str, str | None]]
        | list[dict[str, str | list[str]]],
    ) -> None:
        """Alter final-pass evidence or fail publication of the completion marker.

        Parameters
        ----------
        path : Path
            Destination of the collector's JSON output.
        value : dict[
            str, str | int | float | None | list[int] | dict[str, str | None]
        ] | list[dict[str, str | list[str]]]
            Hardware metadata, totals, completion marker, or warmup records
            forwarded unchanged to the real JSON writer.

        Raises
        ------
        OSError
            Completion-marker writing is the selected failure mode.
        """

        if change == "marker-write" and path.name == "collection.json":
            raise OSError("collection write failed")
        write_json(path, value)
        if path != run.directory / "pass-3/totals.json":
            return
        if change == "metadata":
            write_json(run.directory / "run.json", {**run.spec, "beam": 1})
        elif change in ("artifact", "rehash"):
            path = run.plugins / "feature.so"
            path.write_bytes(b"replacement")
            if change == "rehash":
                write_json(
                    run.directory / "plugin-hashes.json", {path.name: file_hash(path)}
                )
        elif change == "missing-evidence":
            (run.directory / "pass-3/first.jsonl").unlink()

    monkeypatch.setattr(collect, "write_json", persist_and_change)
    with pytest.raises(
        (ValueError, OSError),
        match={
            "source": "Sources changed",
            "metadata": "Run metadata changed",
            "artifact": "Plugin files differ",
            "rehash": "Export hashes changed",
            "missing-evidence": "first.jsonl",
            "marker-write": "collection write failed",
        }[change],
    ):
        collect.main()
    assert (run.directory / "pass-3/totals.json").is_file()
    assert not (run.directory / "collection.json").exists()
