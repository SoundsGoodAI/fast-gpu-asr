#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Build, collect, and score a sequential matrix on an already available GPU.

Run via ``python -m benchmarks.run`` with a frozen campaign and a new output
directory. Each stage runs in a fresh subprocess on the selected physical GPU.
"""

import argparse
import importlib.metadata
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from platform import python_version
from re import search
from shutil import rmtree
from sys import executable
from traceback import format_exc

import cupy as cp

from .collect import check_gpu, nvidia_query
from .common import (
    BATCHES,
    BEAMS,
    GPUS,
    MODELS,
    PRECISIONS,
    PROTOCOL,
    ROOT,
    file_hash,
    load_campaign,
    write_json,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the GPU, matrix subset, verified inputs, and output location.

    Returns
    -------
    argparse.Namespace
        Validated matrix selection, defaulting to all configured batch capacities
        on every GPU. Use --batches to restrict the sweep explicitly.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=GPUS, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument(
        "--batches",
        choices=BATCHES,
        type=int,
        nargs="+",
        help="Batch capacities; defaults to all configured values on every GPU.",
    )
    parser.add_argument(
        "--precisions", choices=PRECISIONS, nargs="+", default=PRECISIONS
    )
    parser.add_argument(
        "--discard-engines",
        action="store_true",
        help="Remove each run's bundle after its attempt; retain logs and any hashes.",
    )

    for model in MODELS:
        parser.add_argument(f"--{model}", type=Path, required=True)

    args = parser.parse_args()

    if args.batches is None:
        args.batches = list(BATCHES)
    if args.device_id < 0:
        parser.error("--device-id must be non-negative.")
    for option in (args.models, args.batches, args.precisions):
        if len(option) != len(set(option)):
            parser.error("Duplicate models, batches, or precisions are not allowed.")

    return args


def check_checkpoint(path: Path, files: dict[str, str]) -> None:
    """Verify the selected checkpoint and its frozen companion files.

    Parameters
    ----------
    path : Path
        Checkpoint passed to the exporter, retaining its original filename.
    files : dict[str, str]
        Filename-to-SHA-256 mapping saved by preparation for this model.

    Raises
    ------
    ValueError
        The selected filename or any file's digest differs from the campaign.
        Missing files raise OSError.
    """

    if files.keys() - {"config.yaml", "bpe.model"} != {path.name}:
        raise ValueError(f"Checkpoint is not the frozen model file: {path}")

    for filename, expected in files.items():
        if file_hash(path.parent / filename) != expected:
            raise ValueError(f"Checkpoint changed: {path.parent / filename}")


def main() -> None:
    """Run export, collection, and scoring sequentially for each configuration.

    Existing output is refused; retries require a new run root. The matrix file
    records even configurations not reached after interruption. Campaign/source
    changes abort the matrix. Checkpoints and sidecars are hashed before launch
    and again after each export, outside inference timing.
    Export arguments use the shared protocol, model-specific beam width, and the
    same requested precision for both engines, retaining exporter math defaults.

    Host/GPU metadata is captured before export so failed builds retain it too.
    Collection refreshes that snapshot in its own process before measurement;
    failures retain the latest saved snapshot. NVIDIA fields remain strings;
    an unavailable cgroup-v2 quota is ``None``.

    Stage failures retain diagnostics and allow the next configuration to run.
    KeyboardInterrupt is recorded and re-raised; abrupt termination can leave a
    running record. Only the scorer marks a run complete. Optional engine cleanup
    removes this run's bundle on success or failure, preserving logs and hashes
    when available. Filesystem failures during recording or cleanup propagate.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    campaign = load_campaign(args.campaign, check_source=True)
    for model in args.models:
        check_checkpoint(getattr(args, model), campaign["models"][model]["files"])

    gpu = nvidia_query("gpu", "name,uuid", str(args.device_id))[0]
    if search(rf"\b{args.gpu}\b", gpu[0]) is None:
        raise ValueError(f"Requested {args.gpu}, found {gpu[0]}.")
    check_gpu(gpu[1])

    fields = (
        "name,uuid,memory.total,driver_version,power.limit,"
        "compute_mode,mig.mode.current"
    )
    values = nvidia_query("gpu", fields, gpu[1])[0]
    packages = ("cupy-cuda13x", "tensorrt-cu13", "torch", "numpy", "cuda-bindings")
    quota = Path("/sys/fs/cgroup/cpu.max")
    machine = {
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

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "matrix.json",
        [
            f"{args.gpu}-{model}-{precision}-b{batch}"
            for model in args.models
            for precision in args.precisions
            for batch in args.batches
        ],
    )

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu[1]}
    for model in args.models:
        for precision in args.precisions:
            for batch in args.batches:
                if load_campaign(args.campaign, check_source=True) != campaign:
                    raise ValueError("Campaign changed during the matrix run.")

                directory = args.output / f"{args.gpu}-{model}-{precision}-b{batch}"
                directory.mkdir()
                spec = {
                    "campaign_id": campaign["id"],
                    "model": model,
                    "gpu": args.gpu,
                    "gpu_uuid": gpu[1],
                    "batch_size": batch,
                    "precision": precision,
                    "beam": BEAMS[model],
                    "started_at": datetime.now(UTC).isoformat(),
                }
                write_json(directory / "run.json", spec)
                write_json(directory / "hardware.json", machine)
                result = {
                    "status": "running",
                    "campaign_id": campaign["id"],
                    "spec": spec,
                    "hardware": machine,
                }
                write_json(directory / "result.json", result)

                stage = "build"
                try:
                    check_gpu(gpu[1])
                    family = (
                        "zipformer"
                        if model in ("zipformer_rnnt", "zipformer_cr_ctc_rnnt")
                        else "parakeet"
                    )
                    decoder = (
                        "transducer_modified_beam_search"
                        if BEAMS[model] > 1
                        else "transducer_greedy_search"
                    )
                    command = [
                        executable,
                        "-m",
                        f"fast_gpu_asr.export.export_{family}",
                        "--model-path",
                        str(getattr(args, model).absolute()),
                        "--output-dir",
                        str(directory.resolve() / "bundle"),
                        "--batch-size",
                        str(batch),
                        "--beam",
                        str(BEAMS[model]),
                        "--decoder-type",
                        decoder,
                        "--encoder-precision",
                        precision,
                        "--decoder-precision",
                        precision,
                        "--min-audio-seconds",
                        str(PROTOCOL["audio_seconds"][0]),
                        "--opt-audio-seconds",
                        str(PROTOCOL["audio_seconds"][1]),
                        "--max-audio-seconds",
                        str(PROTOCOL["audio_seconds"][2]),
                        "--optimization-level",
                        str(PROTOCOL["optimization_level"]),
                    ]
                    write_json(directory / "export-command.json", command)
                    with open(directory / "build.log", "w", encoding="utf-8") as log:
                        subprocess.run(
                            command,
                            env=env,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )

                    check_checkpoint(
                        getattr(args, model), campaign["models"][model]["files"]
                    )
                    write_json(
                        directory / "bundle-hashes.json",
                        {
                            str(p.relative_to(directory / "bundle")): file_hash(p)
                            for p in sorted((directory / "bundle").rglob("*"))
                            if p.is_file()
                        },
                    )
                    tensorrt_plugins = sorted(
                        (ROOT / "src" / "fast_gpu_asr" / "tensorrt_plugins").glob(
                            "*.so"
                        )
                    )
                    write_json(
                        directory / "plugin-hashes.json",
                        {p.name: file_hash(p) for p in tensorrt_plugins},
                    )

                    for stage in ("collect", "score"):
                        command = [
                            executable,
                            "-m",
                            f"benchmarks.{stage}",
                            "--campaign",
                            str(args.campaign.resolve()),
                            "--run-dir",
                            str(directory.resolve()),
                        ]
                        option = "datasets_root" if stage == "collect" else "scorer"
                        command += [
                            f"--{option.replace('_', '-')}",
                            str(getattr(args, option).resolve()),
                        ]
                        with open(
                            directory / f"{stage}.log", "w", encoding="utf-8"
                        ) as log:
                            subprocess.run(
                                command,
                                env=env,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )

                except (Exception, KeyboardInterrupt) as error:
                    log_path = directory / f"{stage}.log"
                    result.update(
                        status="failed",
                        hardware=json.loads(
                            (directory / "hardware.json").read_text(encoding="utf-8")
                        ),
                        stage=stage,
                        error=f"{type(error).__name__}: {error}",
                        traceback=format_exc(),
                        diagnostics=log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )[-8000:]
                        if log_path.exists()
                        else "",
                    )

                    write_json(directory / "result.json", result)
                    if isinstance(error, KeyboardInterrupt):
                        raise

                finally:
                    if args.discard_engines and (directory / "bundle").exists():
                        rmtree(directory / "bundle")

                logger.info("Finished %s; see result.json", directory.name)


if __name__ == "__main__":
    main()
