#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Publish benchmark plots and tables from complete, consistent local evidence.

Run via ``python -m benchmarks.report`` on a CPU reporting host with the benchmark
dependencies and Chrome/Chromium installed. Raw transcripts, references, and
failure logs stay local. Reporting checks saved scorer output; it does not rerun
normalization or WER scoring. Rescore with ``benchmarks.score`` when needed.

The private campaign and run artifacts are read-only inputs. Public output
contains complete measurements, a reference-redacted manifest, Markdown tables,
and one SVG per model/precision pair. An optional README update replaces only its
benchmark marker block. See ``benchmarks/README.md`` for the command-line workflow.
"""

import argparse
import json
import logging
from contextlib import ExitStack
from hashlib import sha256
from math import isclose, isfinite, log10
from pathlib import Path
from re import fullmatch, search
from statistics import mean, median
from tempfile import TemporaryDirectory
from urllib.parse import quote

import plotly.graph_objects as go
import plotly.io as pio

from .common import (
    BATCHES,
    BEAMS,
    DATASETS,
    GPUS,
    MODELS,
    PRECISIONS,
    JSONValue,
    batches,
    load_campaign,
    pass_totals,
    write_json,
)

logger = logging.getLogger(__name__)
COLORS = dict(
    zip(
        GPUS,
        ("blueviolet", "deeppink", "dodgerblue", "darkorange", "limegreen"),
        strict=True,
    )
)
START = "<!-- benchmark-results:start -->"
END = "<!-- benchmark-results:end -->"


def parse_args() -> argparse.Namespace:
    """Parse paths needed to validate runs and publish a benchmark report.

    Returns
    -------
    argparse.Namespace
        Private campaign path, one or more run roots, public output directory,
        and optional README path. Filesystem and evidence checks happen later.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, required=True, help="Frozen private campaign JSON."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Matrix roots or individual run directories; include failed attempts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Public report directory; generated filenames are replaced.",
    )
    parser.add_argument(
        "--readme", type=Path, help="README containing one benchmark marker pair."
    )

    return parser.parse_args()


def validate_result(
    campaign: dict[str, JSONValue], directory: Path
) -> dict[str, JSONValue]:
    """Cross-check run metadata, completion provenance, scores, and raw evidence.

    Parameters
    ----------
    campaign : dict[str, JSONValue]
        Private campaign already validated by ``load_campaign``, with the dataset
        rows, protocol, and source identities frozen during preparation.
    directory : Path
        Individual run directory containing ``result.json``, ``run.json``, and
        ``hardware.json``. Complete runs additionally require collection and
        export hash manifests, plus each pass's scores and transcript/timing JSONLs.

    Returns
    -------
    dict[str, JSONValue]
        Public status, run specification, and hardware metadata. Complete runs
        include export/evidence hashes, validated per-pass dataset scores, and
        recomputed throughput and inference-time summaries. Failed runs add their
        failing stage and a generic message. Neither failed nor running results
        contain measurements. Raw predictions, scorer extras, and diagnostics
        are omitted.

    Raises
    ------
    ValueError
        JSON is invalid, metadata or evidence is inconsistent, a configuration is
        unsupported, or numeric values fail validation.
    OSError
        A required artifact is missing or cannot be read.
    KeyError, TypeError
        Records omit required fields or have incompatible structures.

    Notes
    -----
    Complete results require every configured pass and every dataset. Prediction
    IDs must follow each dataset's duration-then-ID batch order. Audio seconds come
    from frozen frame counts and sample rates; inference seconds come from batches.
    Saved timing totals and ratios must match within relative tolerance 1e-12.
    RTFx pools audio seconds over inference seconds, never dataset ratios.

    Saved WERs and error counts must agree with each pass's scorer output, but
    normalization and WER are not rerun. Empty hypotheses and WER above 100 percent
    are valid. Per-pass mean WER is the unweighted mean of dataset WERs. Summary
    medians and pass ranges are derived again, not trusted from ``result.json``.

    Hash inventories are compared with the saved export manifests, not current
    binaries; the collection source snapshot is compared with the campaign, not
    the current checkout. Engines may have been discarded and no GPU is needed.
    These checks detect inconsistent artifacts, not coordinated changes to all
    copies. Public metadata must be finite JSON. Transcript JSONLs are hashed and
    checked in one read, without retaining entire files' hypotheses or timestamps.
    The campaign and run artifacts are not modified.
    """

    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result["campaign_id"] != campaign["id"]:
        raise ValueError("Different campaign.")
    if result["status"] not in ("complete", "failed", "running"):
        raise ValueError("Unknown run status.")

    spec = result["spec"]
    if (
        spec != json.loads((directory / "run.json").read_text(encoding="utf-8"))
        or spec["campaign_id"] != campaign["id"]
    ):
        raise ValueError("Run metadata differs from the result or campaign.")
    if (
        spec["model"] not in MODELS
        or spec["gpu"] not in GPUS
        or spec["precision"] not in PRECISIONS
        or not isinstance(spec["beam"], int)
        or spec["beam"] != BEAMS[spec["model"]]
        or not isinstance(spec["batch_size"], int)
        or spec["batch_size"] not in BATCHES
    ):
        raise ValueError("Configuration is outside the benchmark matrix.")

    machine = json.loads((directory / "hardware.json").read_text(encoding="utf-8"))
    if (
        result["hardware"] != machine
        or machine["gpu"]["uuid"] != spec["gpu_uuid"]
        or search(rf"\b{spec['gpu']}\b", machine["gpu"]["name"]) is None
    ):
        raise ValueError("Hardware metadata differs from the recorded GPU.")

    spec_keys = (
        "campaign_id",
        "model",
        "gpu",
        "gpu_uuid",
        "batch_size",
        "precision",
        "beam",
        "started_at",
    )
    public = {
        "status": result["status"],
        "campaign_id": campaign["id"],
        "spec": {key: spec[key] for key in spec_keys},
        "hardware": machine,
    }
    json.dumps(public, allow_nan=False)

    if result["status"] != "complete":
        if result["status"] == "failed":
            if result["stage"] not in ("build", "collect", "score"):
                raise ValueError("Unknown failing stage.")
            public.update(stage=result["stage"], error="See local run logs.")

        return public

    num_passes = campaign["protocol"]["passes"]
    if len(result["passes"]) != num_passes:
        raise ValueError(f"Expected {num_passes} complete passes.")

    collection = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    if (
        collection["campaign_id"] != campaign["id"]
        or collection["source"] != campaign["source"]
        or collection["evidence"] != result["evidence"]
    ):
        raise ValueError("Collection provenance differs from the campaign or result.")

    expected_files = {
        f"pass-{index}/{name}.jsonl"
        for index in range(1, num_passes + 1)
        for name in (*DATASETS, "batches")
    }
    if set(result["evidence"]) != expected_files:
        raise ValueError("Missing transcript or batch evidence hashes.")

    for key, filename in (
        ("bundle_files", "bundle-hashes.json"),
        ("plugins", "plugin-hashes.json"),
    ):
        hashes = json.loads((directory / filename).read_text(encoding="utf-8"))
        if (
            not isinstance(hashes, dict)
            or not hashes
            or hashes != result[key]
            or any(
                fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()
            )
        ):
            raise ValueError("Export hash inventory differs from the result.")

        public[key] = hashes

    capacity = spec["batch_size"]
    passes: list[dict[str, JSONValue]] = []
    for index, saved in enumerate(result["passes"], 1):
        folder = directory / f"pass-{index}"
        if saved != json.loads((folder / "scores.json").read_text(encoding="utf-8")):
            raise ValueError("Result differs from the saved pass scores.")

        path = folder / "batches.jsonl"
        checksum = sha256()
        timings: list[dict[str, JSONValue]] = []
        with open(path, "rb") as stream:
            for line in stream:
                checksum.update(line)
                timings.append(json.loads(line))

        if checksum.hexdigest() != result["evidence"][f"pass-{index}/batches.jsonl"]:
            raise ValueError(f"Evidence changed: {path.name}")
        if any(
            not isinstance(record[key], (int, float))
            for record in timings
            for key in ("audio_seconds", "inference_seconds")
        ):
            raise ValueError("Batch timings must be numbers.")

        totals = pass_totals(campaign, timings, capacity)
        if any(
            not isinstance(saved[k], (int, float))
            or not isfinite(saved[k])
            or not isclose(saved[k], value, rel_tol=1e-12)
            for k, value in totals.items()
        ):
            raise ValueError("Saved summary does not match batch timings.")
        if any(not isfinite(value) or value <= 0 for value in totals.values()):
            raise ValueError("Invalid pooled timing.")
        if set(saved["datasets"]) != set(DATASETS):
            raise ValueError("Missing WER results.")

        datasets: dict[str, JSONValue] = {}
        for name, rows in campaign["datasets"].items():
            path = folder / f"{name}.jsonl"
            checksum = sha256()
            ordered = (row for batch in batches(rows, capacity) for row in batch)
            with open(path, "rb") as stream:
                for line in stream:
                    checksum.update(line)
                    prediction = json.loads(line)
                    row = next(ordered, None)
                    if row is None or prediction["id"] != row["id"]:
                        raise ValueError("Incomplete or reordered transcripts.")
                    if not isinstance(prediction["pred_text"], str):
                        raise ValueError(
                            "Hypotheses must be strings, including empty predictions."
                        )

            if checksum.hexdigest() != result["evidence"][f"pass-{index}/{name}.jsonl"]:
                raise ValueError(f"Evidence changed: {path.name}")
            if next(ordered, None) is not None:
                raise ValueError("Incomplete or reordered transcripts.")

            scores = saved["datasets"][name]
            wer = scores["wer"]
            if not isinstance(wer, (int, float)) or not isfinite(wer) or wer < 0:
                raise ValueError("Invalid WER result.")

            errors = {key: scores[key] for key in ("ins", "del", "sub")}
            if any(not isinstance(v, int) or v < 0 for v in errors.values()):
                raise ValueError("Invalid WER error counts.")

            seconds = sum(r["frames"] / r["rate"] for r in rows)
            elapsed = sum(
                r["inference_seconds"] for r in timings if r["dataset"] == name
            )
            metrics = {
                "audio_length": seconds,
                "inference_time": elapsed,
                "rtfx": seconds / elapsed,
            }
            if (
                not isinstance(scores["samples"], int)
                or scores["samples"] != len(rows)
                or any(
                    not isinstance(scores[k], (int, float))
                    or not isfinite(scores[k])
                    or not isclose(scores[k], value, rel_tol=1e-12)
                    for k, value in metrics.items()
                )
            ):
                raise ValueError(
                    "Dataset scores do not match batch timings or samples."
                )

            datasets[name] = {"wer": wer, **errors, "samples": len(rows), **metrics}

        passes.append(
            {
                **totals,
                "datasets": datasets,
                "mean_wer_percent": mean(s["wer"] for s in datasets.values()),
            }
        )

    rtfx = [p["rtfx"] for p in passes]
    public.update(
        passes=passes,
        evidence=result["evidence"],
        rtfx_median=median(rtfx),
        rtfx_range=[min(rtfx), max(rtfx)],
        inference_seconds_median=median(p["inference_seconds"] for p in passes),
    )

    return public


def render_report(
    campaign: dict[str, JSONValue],
    records: list[dict[str, JSONValue]],
    output: Path,
    prefix: str,
) -> str:
    """Write staged public artifacts and return the Markdown README block.

    Parameters
    ----------
    campaign : dict[str, JSONValue]
        Validated private campaign, including raw references used to derive the
        public reference hashes. It is not modified.
    records : list[dict[str, JSONValue]]
        Validated public results and attempt statuses assembled by ``main``.
        Complete entries must already exclude raw diagnostics and scorer text;
        validation, deduplication, and hardware comparisons are not repeated here.
    output : Path
        Existing staging directory receiving ``results.json``, ``manifest.json``,
        ``results.md``, and one 560-by-340 SVG per model/precision pair.
    prefix : str
        URL-escaped relative path from the README to the final report directory,
        independent of the staging path. Use ``.`` when the README is inside the
        report directory. ``main`` passes an empty string when the returned
        block is unused.

    Returns
    -------
    str
        Markdown containing figures, fastest configurations per model/GPU, timing
        notes, findings, and result links. Marker comments are not included and
        the README itself is not written by this function.

    Notes
    -----
    Only complete runs are published, sorted by model, GPU, precision, and batch.
    Highlights select maximum median throughput, not minimum WER. Single-pass
    reports omit run-to-run ranges; other reports retain every pass's mean WER
    and show the median throughput with its minimum/maximum pass range.

    Every figure shares the throughput bounds, GPU colors, and logarithmic batch
    axis. Error bars span minimum/maximum pass throughput; a single pass has
    zero-width bars, not a repeatability estimate. Lines connect measured points
    without inventing data at unmeasured capacities. Empty figures are labeled.
    Static export requires Kaleido and Chrome/Chromium. All figures are exported
    in one call so they can share a browser session.

    Findings identify the best measured median throughput, breaking ties by the
    sorted configuration order. Scaling requires a batch-one baseline with the
    same GPU, model, and precision. A three-precision WER range requires matching
    GPU, model, and batch and includes all passes. Neither finding establishes
    saturation or accuracy parity; reports without complete runs say pending.

    The public manifest retains protocol, source/model identities, and selected
    row metadata, replacing reference text with SHA-256 of
    ``json.dumps(text, sort_keys=True).encode()`` using default ASCII escaping.
    It is a redacted view, not a loadable replacement for the private campaign.
    File and renderer errors propagate; staging cleanup and publication belong
    to ``main``; this function does not roll back partial output.
    """

    complete = sorted(
        (r for r in records if r["status"] == "complete"),
        key=lambda r: (
            tuple(MODELS).index(r["spec"]["model"]),
            GPUS.index(r["spec"]["gpu"]),
            PRECISIONS.index(r["spec"]["precision"]),
            r["spec"]["batch_size"],
        ),
    )
    write_json(
        output / "results.json", {"campaign_id": campaign["id"], "runs": complete}
    )

    public_row_fields = (
        "id",
        "path",
        "sha256",
        "frames",
        "rate",
        "parent_id",
        "chunk_index",
    )
    public = {
        **{key: campaign[key] for key in ("id", "protocol", "source", "models")},
        "datasets": {
            name: [
                {
                    **{k: r[k] for k in public_row_fields if k in r},
                    "reference_sha256": sha256(
                        json.dumps(r["text"], sort_keys=True).encode()
                    ).hexdigest(),
                }
                for r in rows
            ]
            for name, rows in campaign["datasets"].items()
        },
    }
    write_json(output / "manifest.json", public)

    figures, paths = [], []
    maximum = max((r["rtfx_range"][1] for r in complete), default=1) * 1.12
    for precision in PRECISIONS:
        for model in MODELS:
            fig = go.Figure()
            selected = [
                r
                for r in complete
                if r["spec"]["model"] == model and r["spec"]["precision"] == precision
            ]
            for gpu in GPUS:
                points = [r for r in selected if r["spec"]["gpu"] == gpu]

                fig.add_trace(
                    go.Scatter(
                        x=[r["spec"]["batch_size"] for r in points] or [None],
                        y=[r["rtfx_median"] for r in points] or [None],
                        mode="lines+markers",
                        name=gpu,
                        line={"color": COLORS[gpu], "width": 3},
                        marker={
                            "color": COLORS[gpu],
                            "size": 8,
                            "line": {"color": "white", "width": 1.5},
                        },
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                r["rtfx_range"][1] - r["rtfx_median"] for r in points
                            ],
                            "arrayminus": [
                                r["rtfx_median"] - r["rtfx_range"][0] for r in points
                            ],
                            "color": COLORS[gpu],
                            "thickness": 1,
                            "width": 3,
                        },
                        hovertemplate=(
                            "Batch %{x}<br>RTFx %{y:,.1f}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )

            fig.update_layout(
                template="plotly_white",
                font={"family": "Arial, sans-serif", "size": 12, "color": "dimgray"},
                paper_bgcolor="white",
                plot_bgcolor="white",
                title={
                    "text": f"{model.title()} / {precision.upper()}",
                    "font": {"size": 16, "color": "black", "weight": 600},
                    "x": 0.5,
                    "y": 0.97,
                    "yanchor": "top",
                },
                margin={"l": 70, "r": 20, "t": 80, "b": 60},
                legend={
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.03,
                    "yanchor": "bottom",
                    "font": {"size": 12},
                },
                xaxis={
                    "type": "log",
                    # Plotly expresses explicit log-axis limits as base-10 exponents.
                    "range": [log10(0.85), log10(300)],
                    "tickmode": "array",
                    "tickvals": list(BATCHES),
                    "ticktext": [str(batch) for batch in BATCHES],
                    "tickangle": 0,
                    "title": {"text": "Batch capacity"},
                    "showgrid": False,
                    "showline": True,
                    "linecolor": "lightgray",
                    "ticks": "outside",
                    "tickcolor": "lightgray",
                },
                yaxis={
                    "range": [0, maximum],
                    "title": {"text": "Pooled RTFx (higher is better)"},
                    "gridcolor": "gainsboro",
                    "griddash": "dot",
                    "zeroline": False,
                },
            )
            if not selected:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text="No complete suite measurements yet",
                )
            figures.append(fig)
            paths.append(output / f"{model}-{precision}.svg")

    pio.write_images(fig=figures, file=paths, format="svg", width=560, height=340)

    single_pass = campaign["protocol"]["passes"] == 1
    headers = [
        "| Model | GPU | Batch | Precision | "
        + (
            "RTFx | Suite time | Mean WER |"
            if single_pass
            else "RTFx (range) | Suite time | Mean WER, all passes |"
        ),
        "|---|---|---:|---|---:|---:|---|",
    ]
    table, highlighted = [], {}
    for r in complete:
        s = r["spec"]
        wers = ", ".join(f"{p['mean_wer_percent']:.2f}%" for p in r["passes"])
        lo, hi = r["rtfx_range"]
        throughput = f"{r['rtfx_median']:,.1f}"
        if not single_pass:
            throughput += f" ({lo:,.1f}-{hi:,.1f})"

        details = (
            f" | {s['gpu']} | {s['batch_size']} | {s['precision']} | {throughput} | "
            f"{r['inference_seconds_median']:.2f} s | {wers} |"
        )
        table.append(f"| {s['model']}{details}")
        key = (s["gpu"], s["model"])
        if key not in highlighted or r["rtfx_median"] > highlighted[key][0]:
            highlighted[key] = (
                r["rtfx_median"],
                f"| [{s['model']}]({prefix}/results.md){details}",
            )

    notes = ["Measurements pending. No speedup or accuracy-parity claims yet."]
    if complete:
        best = max(complete, key=lambda r: r["rtfx_median"])
        spec = best["spec"]
        notes = [
            f"Best measured point: {spec['model']} on {spec['gpu']}, "
            f"batch {spec['batch_size']}, {spec['precision'].upper()}: "
            f"{best['rtfx_median']:,.1f} RTFx. This is the best tested capacity, "
            "not proof that larger batches have saturated."
        ]
        group = [
            r
            for r in complete
            if all(r["spec"][k] == spec[k] for k in ("gpu", "model", "precision"))
        ]
        single = next((r for r in group if r["spec"]["batch_size"] == 1), None)
        if single and spec["batch_size"] != 1:
            scaling = best["rtfx_median"] / single["rtfx_median"]
            notes.append(
                f"Scaling: that point delivers {scaling:.2f}x "
                "the batch-one throughput under the same protocol."
            )

        matched = [
            r
            for r in complete
            if all(r["spec"][k] == spec[k] for k in ("gpu", "model", "batch_size"))
        ]
        if len(matched) == 3:
            wers = [p["mean_wer_percent"] for r in matched for p in r["passes"]]
            notes.append(
                "Precision: at that GPU/model/batch, the three requested precisions "
                f"span {min(wers):.2f}-{max(wers):.2f}% mean dataset WER "
                "across all passes. "
                "See the complete per-dataset results before choosing a precision."
            )

    timing_note = (
        "Single measured pass per configuration; no run-to-run range is estimated."
        if single_pass
        else "Median pooled throughput and median full-suite inference time; range is "
        f"the minimum and maximum of {campaign['protocol']['passes']} complete passes."
    )

    report = [
        "# Complete Benchmark Results",
        "",
        *headers,
        *table,
        "",
        timing_note + " Mean WER is an "
        "unweighted mean of the seven upstream-rounded dataset WERs.",
        "",
        *[f"- {note}" for note in notes],
        "",
        f"Complete: {len(complete)}.",
        "Per-pass WERs and host/GPU metadata: [results.json](results.json).",
    ]

    (output / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    body = [
        "| " + " | ".join(f"{model} (beam {BEAMS[model]})" for model in MODELS) + " |",
        "|" + "---|" * len(MODELS),
    ]
    for precision in PRECISIONS:
        body.append(
            "| "
            + " | ".join(
                f"![{model} {precision}]({prefix}/{model}-{precision}.svg)"
                for model in MODELS
            )
            + " |"
        )

    if complete:
        body += [
            "",
            "Fastest measured configuration per model/GPU "
            "(selected by throughput, not WER):",
            "",
            *headers,
            *[row for _, row in highlighted.values()],
        ]

    body += [
        "",
        timing_note,
        "",
        *[f"- {note}" for note in notes],
        "",
        f"[Full results]({prefix}/results.md) | "
        f"[Machine-readable results]({prefix}/results.json)",
    ]

    return "\n".join(body)


def main() -> None:
    """Load a private campaign, discover attempts, and publish a CPU-only report.

    Raises
    ------
    ValueError
        The campaign is invalid or conflicts with output/README paths; a run root
        is missing, is not a directory, or contains neither attempts nor matrix
        entries; matrix JSON or names are invalid or duplicated; or a child path
        resolves outside its root or to the root itself. Also raised for output
        overlapping evidence, invalid README markers or destinations, duplicate
        complete configurations, incompatible hardware/software snapshots, or
        a README changed during rendering or publication.
    OSError
        An input cannot be read or resolved, or report publication fails.

    Notes
    -----
    The campaign's fingerprint and protocol are validated without requiring its
    source snapshot to match the current checkout, so historical runs can be
    reported without ASR initialization, a GPU, audio, or retained engines.
    The private campaign file must stay outside public output and cannot be the
    README destination.

    Run roots may be matrices or individual attempts. A directory containing
    ``run.json`` or ``result.json`` is an attempt; otherwise its immediate children
    and optional ``matrix.json`` are checked. Discovered paths are resolved,
    sorted, and deduplicated. Unstarted matrix entries are retained even when
    their directories do not exist yet.

    Matrix entries must be nonempty single relative directory names, without
    separators or dot/parent components. Symlink aliases of the same path are
    deduplicated, not treated as retries. Distinct attempts with the same model
    configuration remain distinct and duplicate complete results abort reporting.
    Individual malformed runs are logged and excluded. Only complete results
    enter public artifacts; incomplete/failed attempts and raw diagnostics remain
    local. Fatal setup, validation, renderer, and publication errors propagate.
    Command-line arguments come from ``parse_args`` and logs go to standard error.
    Campaign and run evidence files are not modified.

    Cross-run hardware comparisons parse ``lscpu`` JSON and ignore the live
    ``CPU MHz:`` and ``CPU(s) scaling MHz:`` values, including nested entries.
    All other fields must match. Raw snapshots remain unchanged in public output,
    and each run's hardware/result snapshot comparison is still exact.

    Output cannot overlap run directories. An optional UTF-8 README must contain
    exactly one ordered benchmark marker pair and cannot overwrite evidence or a
    generated filename. Only its marked block is replaced, using URL-escaped
    relative links and preserving surrounding content, newlines, and permissions.

    Report files and the optional README replacement are staged before existing
    artifacts are replaced, so validation, rendering, and staging failures leave
    them intact. Replacements are atomic per file, not a multi-file transaction:
    failure after promotion starts can leave a partially updated report. The
    README is rechecked after rendering and immediately before replacement, but
    there is no locking; use one publisher and avoid concurrent README edits.
    Temporary directories are cleaned up on normal completion or exceptions.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    if args.readme is not None and args.readme.resolve() == args.campaign.resolve():
        raise ValueError("README must not overwrite the private campaign.")
    if args.campaign.resolve().is_relative_to(args.output.resolve()):
        raise ValueError(
            "Private campaign must be outside the public report directory."
        )

    campaign = load_campaign(args.campaign)

    directories = set()
    for root in sorted({p.resolve() for p in args.runs}):
        if not root.is_dir():
            raise ValueError(f"Run root is not a directory: {root}")

        if any((root / name).is_file() for name in ("run.json", "result.json")):
            directories.add(root)
            continue

        found = {
            p.parent
            for pattern in ("*/run.json", "*/result.json")
            for p in root.glob(pattern)
        }
        matrix = root / "matrix.json"
        if matrix.exists():
            names = json.loads(matrix.read_text(encoding="utf-8"))
            if (
                not isinstance(names, list)
                or not names
                or any(
                    not isinstance(name, str)
                    or not name
                    or name in (".", "..")
                    or Path(name).name != name
                    or "\\" in name
                    for name in names
                )
            ):
                raise ValueError(
                    f"Matrix must contain nonempty run directory names: {matrix}"
                )
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate matrix entries: {matrix}")

            found.update(root / name for name in names)

        if not found:
            raise ValueError(f"No run records or matrix entries: {root}")

        for path in found:
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or resolved == root:
                raise ValueError(f"Run path escapes its root: {path}")

            directories.add(resolved)

    output = args.output.resolve()
    readme = args.readme
    directories = sorted(directories)
    if any(p.is_relative_to(output) or output.is_relative_to(p) for p in directories):
        raise ValueError("Report output must not overlap run directories.")
    if output.exists() and not output.is_dir():
        raise ValueError("Report output must be a directory.")

    generated = {"results.json", "manifest.json", "results.md"} | {
        f"{model}-{precision}.svg" for model in MODELS for precision in PRECISIONS
    }
    for name in sorted(generated):
        if (output / name).is_dir():
            raise ValueError(f"Generated report path is a directory: {output / name}")

    prefix = ""
    if readme is not None:
        readme = readme.resolve()
        if any(readme.is_relative_to(p) for p in directories):
            raise ValueError("README must not overwrite run evidence.")

        original = readme.read_bytes()
        text = original.decode("utf-8")
        if (
            text.count(START) != 1
            or text.count(END) != 1
            or text.index(START) >= text.index(END)
        ):
            raise ValueError(
                "README must contain exactly one ordered benchmark marker pair."
            )
        if readme.parent == output and readme.name in generated:
            raise ValueError("README conflicts with a generated report filename.")

        prefix = quote(
            output.relative_to(readme.parent, walk_up=True).as_posix(), safe="/"
        )

    records, seen, machines = [], set(), {}
    base = directories[0].parent if directories else output
    while any(not path.parent.is_relative_to(base) for path in directories):
        base = base.parent

    for directory in directories:
        name = directory.relative_to(base).as_posix()
        if not directory.exists():
            records.append({"status": "not_run", "run": name})
            continue

        if directory.is_dir() and not (directory / "result.json").exists():
            records.append({"status": "incomplete", "run": name})
            continue

        try:
            result = validate_result(campaign, directory)
            machine = dict(result["hardware"])
            if "lscpu" in machine:
                # Normalize only the comparison copy, preserving the raw snapshot.
                machine["lscpu"] = json.loads(machine["lscpu"])
                pending = [machine["lscpu"]["lscpu"]]
                while pending:
                    fields = pending.pop()
                    if not isinstance(fields, list):
                        raise ValueError("Invalid lscpu metadata.")
                    for field in fields:
                        if field["field"] in ("CPU MHz:", "CPU(s) scaling MHz:"):
                            field["data"] = None
                        if "children" in field:
                            pending.append(field["children"])
        except (
            ValueError,
            KeyError,
            OSError,
            TypeError,
            IndexError,
            OverflowError,
        ) as error:
            logger.warning("Rejected %s: %s", directory, error)
            result = {"status": "rejected"}

        result["run"] = name
        if result["status"] == "complete":
            spec = result["spec"]
            key = tuple(spec[k] for k in ("gpu", "model", "precision", "batch_size"))
            if key in seen:
                raise ValueError(
                    f"Duplicate configuration: {key}; do not cherry-pick runs."
                )

            seen.add(key)
            gpu = spec["gpu"]
            if gpu in machines and machines[gpu] != machine:
                raise ValueError(f"Incompatible hardware/software snapshots for {gpu}.")

            machines[gpu] = machine

        records.append(result)
        logger.info("%s: %s", name, result["status"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        temporary = stack.enter_context(
            TemporaryDirectory(prefix=".benchmark-report-", dir=output.parent)
        )
        staged = Path(temporary)
        body = render_report(campaign, records, staged, prefix)
        if readme is not None:
            if readme.read_bytes() != original:
                raise ValueError("README changed during report generation.")

            before, _, middle = text.partition(START)
            _, _, after = middle.partition(END)
            newline = "\r\n" if "\r\n" in text else "\n"
            block = START + "\n\n" + body + "\n\n" + END
            temporary = stack.enter_context(TemporaryDirectory(dir=readme.parent))
            replacement = Path(temporary) / "README.md"
            replacement.write_bytes(
                (before + block.replace("\n", newline) + after).encode("utf-8")
            )
            replacement.chmod(readme.stat().st_mode & 0o777)

        output.mkdir(exist_ok=True)
        for path in sorted(staged.iterdir()):
            path.replace(output / path.name)
        if readme is not None:
            if readme.read_bytes() != original:
                raise ValueError("README changed during report publication.")
            replacement.replace(readme)

    logger.info("Report: %s (%d complete, %d total)", output, len(seen), len(records))


if __name__ == "__main__":
    main()
