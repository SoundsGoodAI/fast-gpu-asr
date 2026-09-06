#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Collect complete dataset passes through the public ASR API on one GPU.

Run as a fresh subprocess via ``python -m benchmarks.collect``. The orchestrator
maps the requested GPU UUID to CUDA device zero with ``CUDA_VISIBLE_DEVICES``;
standalone invocations must provide the same mapping. Only synchronized ASR calls
are timed. Input loading, hashing, resampling, and result serialization are not.
"""

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import subprocess
from io import StringIO
from math import isfinite
from pathlib import Path
from platform import python_version
from time import perf_counter
from uuid import UUID

import cupy as cp

import fast_gpu_asr

from .common import (
    PROTOCOL,
    ROOT,
    JSONValue,
    batches,
    file_hash,
    load_campaign,
    pass_totals,
    read_audio,
    write_json,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse full-corpus benchmark collection arguments.

    Returns
    -------
    argparse.Namespace
        Campaign, dataset root, and run directory paths consumed by ``main``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, required=True, help="Frozen campaign JSON file."
    )
    parser.add_argument(
        "--datasets-root", type=Path, required=True, help="Materialized dataset root."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Orchestrator-created run directory with an exported bundle.",
    )

    return parser.parse_args()


def nvidia_query(kind: str, fields: str, gpu: str) -> list[list[str]]:
    """Query one physical GPU through NVIDIA's CSV interface.

    Parameters
    ----------
    kind : str
        Query category, such as ``gpu`` or ``compute-apps``.
    fields : str
        Comma-separated NVIDIA query field names.
    gpu : str
        Physical GPU UUID or nvidia-smi index, unaffected by CUDA visibility.

    Returns
    -------
    list[list[str]]
        CSV records, with quoted commas preserved and leading spaces removed.
        An empty process query returns an empty list.

    Raises
    ------
    OSError
        nvidia-smi is unavailable or cannot be started.
    subprocess.SubprocessError
        The query fails or exceeds the 30-second timeout.
    """

    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            f"--query-{kind}={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )

    return list(csv.reader(StringIO(output), skipinitialspace=True))


def check_gpu(gpu: str) -> None:
    """Reject visible competing CUDA processes and MPS configuration.

    Parameters
    ----------
    gpu : str
        Physical GPU UUID to inspect.

    Raises
    ------
    RuntimeError
        Another compute process, an MPS process, or a CUDA_MPS_* variable exists.

    Notes
    -----
    This is a point-in-time check, not an exclusive GPU reservation. The calling
    process is allowed; its PID must be visible consistently to nvidia-smi and
    Python. Workloads that start and stop between checks cannot be detected.
    """

    processes = nvidia_query("compute-apps", "pid,process_name", gpu)
    if any(int(pid) != os.getpid() or "mps" in name.lower() for pid, name in processes):
        raise RuntimeError(f"Competing GPU workload or MPS detected: {processes}")
    if any(name.startswith("CUDA_MPS_") for name in os.environ):
        raise RuntimeError("Unset CUDA_MPS_* before running this benchmark.")


def check_artifacts(run_dir: Path) -> dict[str, dict[str, str]]:
    """Verify the exact bundle and native plugins recorded after export.

    Parameters
    ----------
    run_dir : Path
        Run directory containing the bundle and the orchestrator's hash manifests.

    Returns
    -------
    dict[str, dict[str, str]]
        Verified bundle and plugin hashes, suitable for comparing the snapshots
        before model loading and after collection.

    Raises
    ------
    ValueError
        An artifact inventory is empty or differs from its saved hashes.

    Notes
    -----
    Hashes include binary plugins, which the campaign's source fingerprint does
    not cover. Hashing runs outside inference timing and never changes artifacts.
    """

    verified = {}
    for name, root, pattern in (
        ("bundle", run_dir / "bundle", "**/*"),
        ("plugin", ROOT / "src/fast_gpu_asr/tensorrt_plugins", "*.so"),
    ):
        expected = json.loads(
            (run_dir / f"{name}-hashes.json").read_text(encoding="utf-8")
        )
        actual = {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in sorted(root.glob(pattern))
            if path.is_file()
        }
        if not expected or actual != expected:
            raise ValueError(f"{name.title()} files differ from the export hashes.")

        verified[name] = actual

    return verified


def main() -> None:
    """Validate and collect one orchestrator-created run in a fresh process.

    Existing collection output is refused. CUDA device zero must match the
    recorded physical GPU. Source, run metadata, and artifact snapshots must
    remain unchanged until every pass finishes. Only then is
    ``collection.json`` written, containing the complete transcript/timing hashes.
    Failures propagate to the orchestrator for logging; partial runs cannot be
    promoted to complete by the scorer.

    GPU identity, CUDA/package versions, CPU affinity and quota, lscpu output,
    and thread settings are recorded before model loading, outside timing.
    NVIDIA fields remain strings to preserve unsupported values such as ``[N/A]``;
    an unavailable cgroup-v2 quota is recorded as ``None``, not as unlimited time.

    Warmups span quantiles of the longest utterance in each real batch. Repeated
    selections are intentional when there are fewer batches than warmup calls.
    Datasets never share batches, and partial final batches count only real audio.
    Every load preserves batch order, verifies the frozen audio hash, and returns
    normalized mono waveforms resampled to the protocol's sample rate.

    Each measured call drains earlier stream work before the timer starts. The
    final synchronization is inside the interval, as are feature extraction,
    encoding, decoding, postprocessing, allocations, and graph capture. Loading
    audio, validating outputs, and serializing results remain outside the timer.
    Incomplete ASR outputs and nonpositive/nonfinite elapsed times are rejected.
    GPU process checks run before model loading, after warmup, and after each
    dataset. These boundary snapshots do not reserve the device exclusively.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    campaign = load_campaign(args.campaign, check_source=True)
    spec = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if spec["campaign_id"] != campaign["id"]:
        raise ValueError("Run belongs to a different campaign.")
    if any(args.run_dir.glob("pass-*")) or any(
        (args.run_dir / name).exists() for name in ("warmup.json", "collection.json")
    ):
        raise FileExistsError("Collection output exists; use a new run directory.")
    artifacts = check_artifacts(args.run_dir)
    check_gpu(spec["gpu_uuid"])

    # A separately installed wheel would bypass the source and plugin fingerprints.
    if Path(fast_gpu_asr.__file__).resolve().parent != ROOT / "src/fast_gpu_asr":
        raise ValueError("Collection must import ASR from the campaign's checkout.")

    actual_gpu = "GPU-" + str(
        UUID(bytes=cp.cuda.runtime.getDeviceProperties(0)["uuid"])
    )
    if actual_gpu != spec["gpu_uuid"]:
        raise ValueError(
            f"CUDA device zero is {actual_gpu}, not {spec['gpu_uuid']}. "
            f"Set CUDA_VISIBLE_DEVICES={spec['gpu_uuid']} before starting collection."
        )

    fields = (
        "name,uuid,memory.total,driver_version,power.limit,"
        "compute_mode,mig.mode.current"
    )
    values = nvidia_query("gpu", fields, spec["gpu_uuid"])[0]
    packages = ("cupy-cuda13x", "tensorrt-cu13", "torch", "numpy", "cuda-bindings")
    quota = Path("/sys/fs/cgroup/cpu.max")
    machine: dict[str, JSONValue] = {
        "gpu": dict(zip(fields.split(","), values, strict=True)),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "python": python_version(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": quota.read_text().strip() if quota.exists() else None,
        "lscpu": subprocess.check_output(["lscpu", "--json"], text=True),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }

    write_json(args.run_dir / "hardware.json", machine)

    model = fast_gpu_asr.ASR(args.run_dir / "bundle", device_id=0)
    if (
        model.encoder.batch_size != spec["batch_size"]
        or model.encoder.sample_rate != PROTOCOL["sample_rate"]
        or model.decoder.beam != spec["beam"]
    ):
        raise ValueError(
            "Engine batch capacity, sample rate, or beam differs from the run."
        )

    capacity = model.encoder.batch_size
    suite = {
        name: batches(rows, capacity) for name, rows in campaign["datasets"].items()
    }
    # Quantiles of real batch duration cover short, typical, and long shapes.
    warmups = sorted(
        ((name, batch) for name, dataset in suite.items() for batch in dataset),
        key=lambda item: max(r["frames"] / r["rate"] for r in item[1]),
    )
    indexes = [
        round(i * (len(warmups) - 1) / max(PROTOCOL["warmups"] - 1, 1))
        for i in range(PROTOCOL["warmups"])
    ]
    for i in indexes:
        audios = [
            read_audio(args.datasets_root / r["path"], r["sha256"])[0]
            for r in warmups[i][1]
        ]
        model(audios)
        model.stream.synchronize()

    write_json(
        args.run_dir / "warmup.json",
        [
            {"dataset": warmups[i][0], "ids": [r["id"] for r in warmups[i][1]]}
            for i in indexes
        ],
    )
    check_gpu(spec["gpu_uuid"])

    for pass_index in range(1, campaign["protocol"]["passes"] + 1):
        directory = args.run_dir / f"pass-{pass_index}"
        directory.mkdir()
        records = []
        with (directory / "batches.jsonl").open("w") as timing_file:
            for name, dataset_batches in suite.items():
                with (directory / f"{name}.jsonl").open("w") as transcripts:
                    for batch in dataset_batches:
                        audios = [
                            read_audio(args.datasets_root / r["path"], r["sha256"])[0]
                            for r in batch
                        ]
                        model.stream.synchronize()

                        start = perf_counter()
                        texts, timestamps = model(audios)
                        model.stream.synchronize()
                        elapsed = perf_counter() - start

                        if len(texts) != len(audios) or len(timestamps) != len(audios):
                            raise ValueError("ASR returned an incomplete batch.")
                        if not isfinite(elapsed) or elapsed <= 0:
                            raise ValueError("Invalid inference time.")

                        seconds = sum(r["frames"] / r["rate"] for r in batch)
                        record = {
                            "dataset": name,
                            "ids": [r["id"] for r in batch],
                            "audio_seconds": seconds,
                            "inference_seconds": elapsed,
                        }
                        records.append(record)
                        timing_file.write(json.dumps(record, allow_nan=False) + "\n")
                        timing_file.flush()

                        for row, text, words in zip(
                            batch, texts, timestamps, strict=True
                        ):
                            result = {
                                "id": row["id"],
                                "pred_text": text,
                                "word_timestamps": words,
                            }
                            transcripts.write(
                                json.dumps(result, allow_nan=False) + "\n"
                            )

                # This boundary check also precedes the next dataset.
                check_gpu(spec["gpu_uuid"])
                logger.info("Pass %d: %s complete", pass_index, name)

        write_json(directory / "totals.json", pass_totals(campaign, records, capacity))

    load_campaign(args.campaign, check_source=True)

    if json.loads((args.run_dir / "run.json").read_text(encoding="utf-8")) != spec:
        raise ValueError("Run metadata changed during collection.")
    if check_artifacts(args.run_dir) != artifacts:
        raise ValueError("Export hashes changed during collection.")

    write_json(
        args.run_dir / "collection.json",
        {
            "campaign_id": campaign["id"],
            "source": campaign["source"],
            "evidence": {
                f"pass-{index}/{name}.jsonl": file_hash(
                    args.run_dir / f"pass-{index}/{name}.jsonl"
                )
                for index in range(1, campaign["protocol"]["passes"] + 1)
                for name in (*campaign["datasets"], "batches")
            },
        },
    )


if __name__ == "__main__":
    main()
