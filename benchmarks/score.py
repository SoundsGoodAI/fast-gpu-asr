#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Validate collected passes and score with the public leaderboard toolset on main.

Run via ``python -m benchmarks.score`` after collection completes. Scoring needs
neither audio nor a GPU: transcripts are joined to frozen raw references, while
RTFx is recomputed from the collector's synchronized batch timings.
"""

import argparse
import contextlib
import json
import logging
import statistics
import tempfile
from math import isfinite
from pathlib import Path

from .common import (
    BATCHES,
    BEAMS,
    DATASETS,
    GPUS,
    MODELS,
    PRECISIONS,
    JSONValue,
    batches,
    file_hash,
    load_campaign,
    pass_totals,
    upstream,
    write_json,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths needed to score a completed benchmark collection.

    Returns
    -------
    argparse.Namespace
        Frozen campaign, collected run directory, and scorer checkout on main.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, required=True, help="Frozen campaign JSON file."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Completed collection directory."
    )
    parser.add_argument(
        "--scorer", type=Path, required=True, help="Upstream scorer checkout on main."
    )

    return parser.parse_args()


def check_collection(
    campaign: dict[str, JSONValue], run_dir: Path
) -> dict[str, JSONValue]:
    """Verify the collection marker and every expected transcript/timing hash.

    Parameters
    ----------
    campaign : dict[str, JSONValue]
        Campaign already validated by ``load_campaign``.
    run_dir : Path
        Directory containing the completed collection.

    Returns
    -------
    dict[str, JSONValue]
        Verified marker, suitable for comparing snapshots before/after scoring.

    Raises
    ------
    ValueError
        The marker is absent, its campaign/source differs, or its evidence
        inventory or hashes do not match. Missing evidence files raise OSError.
    """

    marker = run_dir / "collection.json"
    if not marker.exists():
        raise ValueError("Collection did not finish with a valid source snapshot.")

    collection = json.loads(marker.read_text(encoding="utf-8"))
    if (
        collection["campaign_id"] != campaign["id"]
        or collection["source"] != campaign["source"]
    ):
        raise ValueError("Collection source snapshot differs from the campaign.")

    expected_files = {
        f"pass-{index}/{name}.jsonl"
        for index in range(1, campaign["protocol"]["passes"] + 1)
        for name in (*DATASETS, "batches")
    }
    if set(collection["evidence"]) != expected_files:
        raise ValueError("Missing or unexpected transcript/timing evidence hashes.")

    for name, expected in collection["evidence"].items():
        if file_hash(run_dir / name) != expected:
            raise ValueError(f"Collected evidence changed: {name}")

    return collection


def main() -> None:
    """Validate collected evidence, score all passes, and publish run summaries.

    The frozen campaign's source snapshot must match the collection marker;
    rescoring need not use that old checkout. Exported engines may already have
    been discarded. The scorer must use main with a clean normalizer directory.
    Rescoring uses the current checkout, regardless of any older campaign's scorer
    revision; upstream normalization and WER changes can therefore affect results.

    Reference and hypothesis text is normalized only by upstream. Earnings22
    chunks are joined by parent and chunk index by the upstream scorer.
    Empty predictions remain scored; WER above 100 percent is valid.
    Prediction IDs must exactly match collection order. Manifest durations use
    original sample counts; individual transcription times remain unset because
    collection measures whole batches, not individual utterances.

    WER is the unweighted mean of upstream-rounded dataset WERs. RTFx pools
    original audio seconds over measured batch time, never dataset ratios.
    Logs are retained on failure. All passes and final evidence/metadata checks
    must succeed before summaries are written. Each JSON replacement is atomic,
    but a write failure does not roll back other summary files already written.
    Errors propagate to the caller. Existing result.json is replaced only after
    scoring and per-pass writes succeed; failed rescoring retains the previous
    final result rather than labeling it as a new successful evaluation.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    campaign = load_campaign(args.campaign)
    collection = check_collection(campaign, args.run_dir)

    metadata = {
        name: json.loads((args.run_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("run", "hardware", "bundle-hashes", "plugin-hashes")
    }
    spec = metadata["run"]
    if spec["campaign_id"] != campaign["id"]:
        raise ValueError("Run belongs to a different campaign.")
    if (
        spec["model"] not in MODELS
        or spec["gpu"] not in GPUS
        or spec["precision"] not in PRECISIONS
        or not isinstance(spec["batch_size"], int)
        or spec["batch_size"] not in BATCHES
        or not isinstance(spec["beam"], int)
        or spec["beam"] != BEAMS[spec["model"]]
    ):
        raise ValueError("Configuration is outside the benchmark matrix.")
    if metadata["hardware"]["gpu"]["uuid"] != spec["gpu_uuid"]:
        raise ValueError("Recorded GPU differs from the run configuration.")
    if not metadata["bundle-hashes"] or not metadata["plugin-hashes"]:
        raise ValueError("Missing bundle or plugin hashes.")

    implementation = upstream(args.scorer)

    passes = []
    for index in range(1, campaign["protocol"]["passes"] + 1):
        directory = args.run_dir / f"pass-{index}"
        with open(directory / "batches.jsonl", encoding="utf-8") as stream:
            timings = [json.loads(line) for line in stream]

        if any(
            not isinstance(record[key], (int, float))
            for record in timings
            for key in ("audio_seconds", "inference_seconds")
        ):
            raise ValueError("Batch timings must be numbers.")
        totals = pass_totals(campaign, timings, spec["batch_size"])

        with tempfile.TemporaryDirectory() as temporary:
            for name, rows in campaign["datasets"].items():
                with open(directory / f"{name}.jsonl", encoding="utf-8") as stream:
                    predictions = [json.loads(line) for line in stream]

                ordered = [
                    row for batch in batches(rows, spec["batch_size"]) for row in batch
                ]
                if [p["id"] for p in predictions] != [row["id"] for row in ordered]:
                    raise ValueError("Incomplete or reordered transcripts.")

                manifest = []
                for row, prediction in zip(ordered, predictions, strict=True):
                    if not isinstance(prediction["pred_text"], str):
                        raise ValueError(
                            "Hypotheses must be strings, including empty predictions."
                        )

                    manifest.append(
                        {
                            **row,
                            "pred_text": prediction["pred_text"],
                            "duration": row["frames"] / row["rate"],
                            "time": None,
                        }
                    )

                filename = f"MODEL_benchmark-model_DATASET_{name}.jsonl"
                with open(Path(temporary) / filename, "w", encoding="utf-8") as stream:
                    for row in manifest:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")

            with (
                open(directory / "scorer.log", "w", encoding="utf-8") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                _, scores = implementation.score_results(temporary, language="en")

        if set(scores) != {f"benchmark/model | {name}" for name in DATASETS}:
            raise ValueError(
                "Upstream scorer did not return exactly the seven datasets."
            )

        by_dataset = {
            name: dict(scores[f"benchmark/model | {name}"]) for name in DATASETS
        }
        for name in DATASETS:
            metrics = by_dataset[name]
            wer = metrics["wer"]
            if (
                not isinstance(wer, (int, float))
                or not isfinite(wer)
                or wer < 0
                or any(
                    not isinstance(metrics[k], int) or metrics[k] < 0
                    for k in ("ins", "del", "sub")
                )
            ):
                raise ValueError(f"Invalid upstream WER or error counts: {name}")

            records = [r for r in timings if r["dataset"] == name]
            seconds = sum(r["frames"] / r["rate"] for r in campaign["datasets"][name])
            elapsed = sum(r["inference_seconds"] for r in records)
            metrics.update(
                samples=len(campaign["datasets"][name]),
                audio_length=seconds,
                inference_time=elapsed,
                rtfx=seconds / elapsed,
            )

        result = {
            **totals,
            "datasets": by_dataset,
            "mean_wer_percent": statistics.mean(s["wer"] for s in by_dataset.values()),
        }
        passes.append(result)
        logger.info(
            "Pass %d: mean dataset WER %.2f%%, RTFx %.2f",
            index,
            result["mean_wer_percent"],
            result["rtfx"],
        )

    if check_collection(campaign, args.run_dir) != collection:
        raise ValueError("Collection marker changed during scoring.")
    if any(
        json.loads((args.run_dir / f"{name}.json").read_text(encoding="utf-8")) != value
        for name, value in metadata.items()
    ):
        raise ValueError("Run metadata changed during scoring.")

    for index, result in enumerate(passes, 1):
        write_json(args.run_dir / f"pass-{index}" / "scores.json", result)

    rtfx = [p["rtfx"] for p in passes]
    result = {
        "status": "complete",
        "campaign_id": campaign["id"],
        "spec": spec,
        "hardware": metadata["hardware"],
        "bundle_files": metadata["bundle-hashes"],
        "plugins": metadata["plugin-hashes"],
        "passes": passes,
        "rtfx_median": statistics.median(rtfx),
        "rtfx_range": [min(rtfx), max(rtfx)],
        "inference_seconds_median": statistics.median(
            p["inference_seconds"] for p in passes
        ),
        "evidence": collection["evidence"],
    }
    write_json(args.run_dir / "result.json", result)
    logger.info("Scored %s", args.run_dir)


if __name__ == "__main__":
    main()
