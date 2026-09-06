#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Shared protocol, provenance, audio loading, and complete-pass validation.

Preparation freezes source identities and original audio metadata. Collection,
scoring, and reporting reuse that evidence and the same duration/batching rules.
These helpers do not import the ASR runtime or initialize CUDA. The public
scorer and its optional dependencies are imported only by ``upstream``.
"""

import importlib
import json
import subprocess
import sys
import wave
from audioop import ratecv
from hashlib import file_digest, sha256
from io import BytesIO
from math import isclose, isfinite
from pathlib import Path
from types import ModuleType

import numpy as np

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "ami_cleaned_test",
    "earnings22_cleaned_aa_chunked_test",
    "gigaspeech_cleaned_test",
    "librispeech_test.clean",
    "librispeech_test.other",
    "spgispeech_test",
    "voxpopuli_cleaned_aa_test",
)
MODELS = {
    "zipformer_rnnt": "soundsgoodai/Zipformer-transducer-XL-290M",
    "zipformer_cr_ctc_rnnt": "soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M",
    "parakeet_v2": "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet_v3": "nvidia/parakeet-tdt-0.6b-v3",
}
BEAMS = {
    "zipformer_rnnt": 6,
    "zipformer_cr_ctc_rnnt": 6,
    "parakeet_v2": 6,
    "parakeet_v3": 6,
}
GPUS = ("A100", "H100", "H200", "B200", "B300")
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PRECISIONS = ("fp32", "fp16", "bf16")
PROTOCOL = {
    "version": 1,
    "sample_rate": 16000,
    "audio_seconds": [0.1, 8, 40],
    "optimization_level": 5,
    "warmups": 5,
    "passes": 1,
    "timing": "synchronized full ASR call",
    "sort": "duration, utterance ID; within dataset",
    "math": "exporter defaults, including TF32 and eligible reduced-math tactics",
}


def file_hash(path: Path) -> str:
    """Hash a file incrementally without loading a whole checkpoint or engine.

    Parameters
    ----------
    path : Path
        File whose exact bytes should be fingerprinted.

    Returns
    -------
    str
        Lowercase SHA-256 hex digest; empty files are supported.
    """

    with open(path, "rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: JSONValue) -> None:
    """Write finite JSON to a sibling temporary file, then replace the target.

    Parameters
    ----------
    path : Path
        Destination in an existing directory. An existing file is replaced.
    value : JSONValue
        JSON-serializable data; NaN and infinity are rejected, including nested values.

    Notes
    -----
    Each target must have one writer. Its sibling ``<filename>.tmp`` may remain
    after a failure and is overwritten on retry. Readers see the old or new target,
    never a partial write. This is not a multi-file transaction or an fsync-based
    guarantee against power loss.
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def source_identity() -> dict[str, JSONValue]:
    """Fingerprint the current checkout, including uncommitted source changes.

    Returns
    -------
    dict[str, JSONValue]
        ``commit`` is Git HEAD; ``files`` maps repository-relative paths to SHA-256
        digests for pyproject.toml, uv.lock, and all .py/.cu/.h files under src and
        benchmarks. Tests, documentation, model files, and compiled binaries are
        excluded. Exported artifacts and plugin binaries are verified separately.

    Notes
    -----
    This is a point-in-time observation, not a source lock. Preparation and
    collection compare snapshots at their boundaries to detect intervening edits.
    """

    files = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("src", "benchmarks"):
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.suffix in (".py", ".cu", ".h") and p.is_file()
        )

    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)},
    }


def load_campaign(path: Path, check_source: bool = False) -> dict[str, JSONValue]:
    """Load a frozen campaign and verify its fingerprint and protocol envelope.

    Parameters
    ----------
    path : Path
        Private campaign JSON produced by preparation, not a redacted public report.
    check_source : bool
        Compare against the current source checkout for export/collection. Leave
        false for historical rescoring/reporting, which use the saved provenance.

    Returns
    -------
    dict[str, JSONValue]
        Unmodified campaign with the exact ordered dataset suite and a positive
        measured pass count. Preparation validates individual rows/checkpoints;
        this helper does not reload their audio or verify remote checkpoint files.

    Raises
    ------
    ValueError
        The campaign fingerprint, protocol, dataset mapping/order, or requested
        source identity check does not match.
    """

    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(campaign, dict)
        or campaign.get("id")
        != sha256(
            json.dumps(
                {k: v for k, v in campaign.items() if k != "id"}, sort_keys=True
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("Campaign fingerprint mismatch.")

    protocol = campaign.get("protocol")
    datasets = campaign.get("datasets")
    if not isinstance(protocol, dict) or not isinstance(datasets, dict):
        raise ValueError("Campaign uses a different protocol or dataset suite.")

    passes = protocol.get("passes")
    if (
        not isinstance(passes, int)
        or passes < 1
        or protocol != {**PROTOCOL, "passes": passes}
        or tuple(datasets) != DATASETS
    ):
        raise ValueError("Campaign uses a different protocol or dataset suite.")

    if check_source and campaign["source"] != source_identity():
        raise ValueError("Sources or lockfile changed; freeze a new campaign.")

    return campaign


def upstream(path: Path) -> ModuleType:
    """Import the public scorer from a trusted checkout of its main branch.

    Parameters
    ----------
    path : Path
        Root of the leaderboard Git checkout; relative paths are resolved first.

    Returns
    -------
    ModuleType
        ``normalizer.eval_utils`` module exposing the public ``score_results`` API.

    Raises
    ------
    ValueError
        The checkout is not on main, Git reports normalizer edits/untracked files,
        or the imported scorer originates outside the requested checkout.

    Notes
    -----
    The temporary search-path entry is removed even when importing fails.
    No fetching or branch switching is performed; update main before running the
    CLI. Normalizer modules remain cached in Python, so use a fresh process after
    updating the checkout. This does not sandbox imported code.
    """

    path = path.resolve()
    branch = subprocess.check_output(
        ["git", "-C", str(path), "branch", "--show-current"], text=True
    ).strip()
    if branch != "main":
        raise ValueError("Scorer must be checked out on main.")

    if subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "normalizer",
        ],
        text=True,
    ).strip():
        raise ValueError("Scorer normalizer directory must be clean.")

    sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("normalizer.eval_utils")
        if not Path(module.__file__).resolve().is_relative_to(path):
            raise ValueError("Another normalizer package was already imported.")
        return module
    finally:
        sys.path.pop(0)


def read_audio(
    path: Path, expected_hash: str | None = None
) -> tuple[np.typing.NDArray[np.float32], dict[str, str | int]]:
    """Read and hash mono PCM16 WAV, resampling outside inference timing.

    Parameters
    ----------
    path : Path
        Nonempty mono 16-bit PCM WAV file.
    expected_hash : str or None
        Frozen SHA-256 digest to verify before decoding. None records the hash
        without comparing it, as needed during campaign preparation.

    Returns
    -------
    tuple[np.typing.NDArray[np.float32], dict[str, str | int]]
        One-dimensional audio at the protocol sample rate, scaled by 1/32768,
        and original-file ``sha256``, ``frames``, and ``rate`` metadata. Only
        differing sample rates invoke ``audioop.ratecv`` with a fresh state.
        Resampling rounding does not change the saved source duration.

    Raises
    ------
    ValueError
        The hash differs, audio is empty/not mono PCM16, or its sample payload
        is truncated or has an invalid sample rate.
    """

    data = path.read_bytes()

    checksum = sha256(data).hexdigest()
    if expected_hash is not None and checksum != expected_hash:
        raise ValueError(f"Audio changed: {path}")

    with wave.open(BytesIO(data)) as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or frames <= 0:
            raise ValueError(f"Expected nonempty mono PCM16 WAV: {path}")

        pcm = wav.readframes(frames)
        if len(pcm) != frames * 2 or rate <= 0:
            raise ValueError(f"Truncated or invalid WAV: {path}")

    if rate != PROTOCOL["sample_rate"]:
        pcm, _ = ratecv(pcm, 2, 1, rate, PROTOCOL["sample_rate"], None)

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    return audio, {"sha256": checksum, "frames": frames, "rate": rate}


def batches(
    rows: list[dict[str, JSONValue]], capacity: int
) -> list[list[dict[str, JSONValue]]]:
    """Sort one dataset by duration then ID, retaining the final partial batch.

    Parameters
    ----------
    rows : list[dict[str, JSONValue]]
        Prepared rows with unique string IDs and positive source frames/rates.
    capacity : int
        Fixed engine batch capacity from ``BATCHES``.

    Returns
    -------
    list[list[dict[str, JSONValue]]]
        New batch lists referencing the original row dictionaries. The input order
        is unchanged; empty input returns no batches. Datasets are batched separately.

    Raises
    ------
    ValueError
        The capacity is outside the supported benchmark matrix.
    """

    if capacity not in BATCHES:
        raise ValueError(f"Unsupported batch capacity: {capacity}")

    ordered = sorted(rows, key=lambda r: (r["frames"] / r["rate"], r["id"]))
    return [ordered[i : i + capacity] for i in range(0, len(ordered), capacity)]


def pass_totals(
    campaign: dict[str, JSONValue], records: list[dict[str, JSONValue]], capacity: int
) -> dict[str, float]:
    """Verify one complete pass and pool real audio over measured compute time.

    Parameters
    ----------
    campaign : dict[str, JSONValue]
        Prepared, nonempty dataset mapping in collection order. Source sample
        counts and rates have already been validated during preparation.
    records : list[dict[str, JSONValue]]
        Batch timings with ``dataset``, ordered ``ids``, and numeric ``audio_seconds``
        and ``inference_seconds`` fields, in the collector's emission order.
    capacity : int
        Fixed engine batch capacity used during collection.

    Returns
    -------
    dict[str, float]
        Summed source ``audio_seconds``, summed ``inference_seconds``, and their
        ``rtfx`` ratio. Padding/unused batch rows and dataset-level ratio averaging
        never contribute. Saved durations are checked with relative tolerance 1e-12;
        the returned audio total is recomputed from original frame counts/rates.

    Raises
    ------
    ValueError
        Capacity, batch counts, dataset/utterance order, or saved durations differ,
        or an elapsed time is nonpositive or nonfinite.
    """

    expected = [
        (name, batch)
        for name, rows in campaign["datasets"].items()
        for batch in batches(rows, capacity)
    ]
    if len(records) != len(expected):
        raise ValueError("Incomplete suite: batch count differs.")

    audio_seconds = inference_seconds = 0.0
    for record, (name, batch) in zip(records, expected, strict=True):
        seconds = sum(r["frames"] / r["rate"] for r in batch)
        if record["dataset"] != name or record["ids"] != [r["id"] for r in batch]:
            raise ValueError("Incomplete suite: dataset, order, or utterance mismatch.")
        if not isclose(record["audio_seconds"], seconds, rel_tol=1e-12):
            raise ValueError("Audio duration includes padding or is inconsistent.")

        elapsed = record["inference_seconds"]
        if not isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid inference time.")

        audio_seconds += seconds
        inference_seconds += elapsed

    return {
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "rtfx": audio_seconds / inference_seconds,
    }
