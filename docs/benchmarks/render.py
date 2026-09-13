#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Regenerate benchmark tables, SVG plots, and the README performance block.

Run ``python docs/benchmarks/render.py`` from the repository root. Defaults read
``measurements.csv`` and write ``results.md`` and two SVGs beside this script,
updating the README benchmark block. Override with ``--csv``, ``--output``, and
``--readme``.

The CSV contains one complete pass per configuration: real audio and inference
seconds, plus seven WER percentages. RTFx and mean WER are derived during rendering.
Plotly is required; SVG export additionally needs Kaleido and Chrome
(set ``BROWSER_PATH`` if needed). No GPU is used. See ``methodology.md`` for provenance:
rendering checks summary consistency, not the original transcripts or dataset audio.
"""

import argparse
from csv import DictReader
from dataclasses import dataclass
from html import escape
from math import ceil, isclose, isfinite, log10
from pathlib import Path
from re import sub
from statistics import mean
from xml.etree import ElementTree as ET

import plotly.graph_objects as go

# Insertion order controls model plots, GPU legends, and dataset WER columns.
MODELS = {
    "zipformer_cr_ctc_rnnt": "Zipformer CR-CTC-transducer",
    "parakeet_v3": "Parakeet V3",
}
COLORS = {"A100": "dodgerblue", "H200": "limegreen", "B300": "red"}
PRECISIONS = ("fp16", "bf16", "fp32")
PLOTS = (PRECISIONS,)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DATASETS = {
    "ami_cleaned_test": "AMI",
    "earnings22_cleaned_aa_chunked_test": "Earnings22",
    "gigaspeech_cleaned_test": "GigaSpeech",
    "librispeech_test.clean": "LS Clean",
    "librispeech_test.other": "LS Other",
    "spgispeech_test": "SPGISpeech",
    "voxpopuli_cleaned_aa_test": "VoxPopuli",
}
FIELDS = (
    "gpu",
    "model",
    "precision",
    "batch_size",
    "beam",
    "audio_seconds",
    "inference_seconds",
    *(f"wer_{name}" for name in DATASETS),
)
START = "<!-- benchmark-results:start -->"
END = "<!-- benchmark-results:end -->"
BENCHMARK_URL = (
    "https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/"
)
RAW_BENCHMARK_URL = (
    "https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/"
)


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for benchmark rendering.

    Returns
    -------
    argparse.Namespace
        CSV, output-directory, and README paths. Defaults are relative to this
        script, not the working directory; file validation happens in ``main``.
    """

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "measurements.csv",
        help="Input CSV; defaults to measurements.csv beside this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root,
        help="Output directory for results.md and SVGs; defaults beside this script.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=root.parents[1] / "README.md",
        help="README with benchmark markers; defaults to the repository README.",
    )

    return parser.parse_args()


@dataclass(frozen=True)
class Measurement:
    """Immutable summary of one complete pass; validated by ``load_measurements``.

    Attributes
    ----------
    gpu, model, precision : str
        Keys from COLORS, MODELS, and PRECISIONS. Precision describes the
        requested type, not a strict arithmetic guarantee.
    batch_size, beam : int
        Exported batch capacity and decoder beam width, respectively.
    audio_seconds, inference_seconds : float
        Real audio duration excluding padding, and total synchronized full-ASR
        time, respectively, both in seconds.
    wers : tuple[float, ...]
        Seven dataset WER percentages in DATASETS order.
    """

    gpu: str
    model: str
    precision: str
    batch_size: int
    beam: int
    audio_seconds: float
    inference_seconds: float
    wers: tuple[float, ...]

    @property
    def rtfx(self) -> float:
        """Pooled audio-to-inference ratio, not the mean of dataset RTFx values."""
        return self.audio_seconds / self.inference_seconds

    @property
    def mean_wer(self) -> float:
        """Unweighted mean of the seven dataset WERs, expressed as a percentage."""
        return mean(self.wers)


def load_measurements(path: Path) -> list[Measurement]:
    """Read complete batch series from CSV, allowing pending model/precision series.

    Parameters
    ----------
    path : Path
        UTF-8 CSV with the exact FIELDS header and column order.

    Returns
    -------
    list[Measurement]
        Records sorted by COLORS, MODELS, PRECISIONS, then batch size.

    Raises
    ------
    ValueError
        Invalid columns, values, duplicates, incomplete batch series, or differing
        audio totals. Row errors include the record number, starting at two.
    """

    with open(path, newline="", encoding="utf-8") as stream:
        reader = DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected CSV columns or column order.")
        records, seen = [], set()
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Wrong column count.")

                record = Measurement(
                    row["gpu"],
                    row["model"],
                    row["precision"],
                    int(row["batch_size"]),
                    int(row["beam"]),
                    float(row["audio_seconds"]),
                    float(row["inference_seconds"]),
                    tuple(float(row[f"wer_{name}"]) for name in DATASETS),
                )
                if (
                    record.gpu not in COLORS
                    or record.model not in MODELS
                    or record.precision not in PRECISIONS
                    or record.batch_size not in BATCHES
                    or record.beam != 6
                    or (record.precision == "fp32" and record.batch_size == 256)
                ):
                    raise ValueError("Configuration is outside the published matrix.")
                if not (
                    0 < record.audio_seconds < float("inf")
                    and 0 < record.inference_seconds < float("inf")
                    and 0 < record.rtfx < float("inf")
                ):
                    raise ValueError("Timing totals must be finite and positive.")
                if not all(isfinite(v) and v >= 0 for v in record.wers):
                    raise ValueError("WERs must be finite, nonnegative percentages.")

                key = (record.gpu, record.model, record.precision, record.batch_size)
                if key in seen:
                    raise ValueError("Duplicate configuration.")

                seen.add(key)
                records.append(record)
            except ValueError as error:
                raise ValueError(f"CSV line {line}: {error}") from error

    if not records:
        raise ValueError("No measurements.")
    if any(
        not isclose(r.audio_seconds, records[0].audio_seconds, rel_tol=1e-12)
        for r in records
    ):
        raise ValueError("All configurations must cover the same real audio total.")
    for gpu, model, precision in {key[:3] for key in seen}:
        batches = BATCHES[:8] if precision == "fp32" else BATCHES
        if any((gpu, model, precision, batch) not in seen for batch in batches):
            raise ValueError(f"Incomplete {gpu} {model} {precision} batch series.")

    return sorted(
        records,
        key=lambda r: (
            tuple(COLORS).index(r.gpu),
            tuple(MODELS).index(r.model),
            PRECISIONS.index(r.precision),
            r.batch_size,
        ),
    )


def pending_results(records: list[Measurement]) -> str:
    """Describe missing model/precision series on partially measured GPUs.

    Parameters
    ----------
    records : list[Measurement]
        Validated, complete batch series from ``load_measurements``.

    Returns
    -------
    str
        Markdown paragraphs with each incomplete GPU's configuration count and
        missing series, in COLORS/MODELS/PRECISIONS order. GPUs with no
        measurements are omitted. Empty when all represented GPUs have their
        complete 52-configuration matrix.
    """

    present = {(r.gpu, r.model, r.precision) for r in records}
    notes = []
    for gpu in COLORS:
        count = sum(r.gpu == gpu for r in records)
        if 0 < count < 52:
            pending = [
                f"{name} {precision.upper()} "
                f"(batches 1-{128 if precision == 'fp32' else 256})"
                for model, name in MODELS.items()
                for precision in PRECISIONS
                if (gpu, model, precision) not in present
            ]
            notes.append(
                f"**{gpu}: {count}/52 configurations.** "
                f"Pending: {', '.join(pending)}. No values are filled in."
            )
    return "\n\n".join(notes)


def plot_grid() -> str:
    """Stack full-width model plots with all three precisions overlaid.

    Returns
    -------
    str
        Single-column HTML table with headings and absolute, PyPI-compatible URLs.
    """

    lines = ["<table>", "  <tbody>"]
    for model, name in MODELS.items():
        lines += [
            "    <tr>",
            '      <th width="100%"><div align="center">'
            f"<big>{escape(name)} (beam 6)</big></div></th>",
            "    </tr>",
        ]
        for precisions in PLOTS:
            suffix = "-".join(precisions)
            label = " / ".join(p.upper() for p in precisions)
            path = escape(f"{RAW_BENCHMARK_URL}{model.split('_')[0]}-{suffix}.svg")
            lines += [
                "    <tr>",
                f'      <td width="100%"><a href="{path}"><img src="{path}" '
                f'width="100%" alt="{escape(name)}, {label}" /></a></td>',
                "    </tr>",
            ]
    lines += ["  </tbody>", "</table>"]
    return "\n".join(lines)


def highlights(records: list[Measurement]) -> str:
    """Render FP16 metrics with batches 1 and 256 paired for comparison.

    Parameters
    ----------
    records : list[Measurement]
        Sorted records from ``load_measurements``.

    Returns
    -------
    str
        One row per GPU/model with batch pairs for each metric, then evaluation links.
    """

    lines = [
        "**FP16, batch sizes 1 and 256, beam 6:**",
        "",
        "| GPU | Model | Batch 1<br>RTFx | Batch 256<br>RTFx"
        " | Batch 1<br>Suite time | Batch 256<br>Suite time"
        " | Batch 1<br>Mean WER | Batch 256<br>Mean WER |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    large_batches = {
        (r.gpu, r.model): r
        for r in records
        if r.precision == "fp16" and r.batch_size == 256
    }
    for r in records:
        if r.precision == "fp16" and r.batch_size == 1:
            large = large_batches[r.gpu, r.model]
            lines.append(
                f"| {r.gpu} | [{MODELS[r.model]}]"
                f"({BENCHMARK_URL}results.md#{r.gpu.lower()})"
                f" | {r.rtfx:,.1f} | {large.rtfx:,.1f}"
                f" | {r.inference_seconds:.2f} s | {large.inference_seconds:.2f} s"
                f" | {r.mean_wer:.3f}% | {large.mean_wer:.3f}% |"
            )

    lines += [
        "",
        "We reproduced the [Open ASR Leaderboard]"
        "(https://github.com/huggingface/open_asr_leaderboard) English evaluation "
        "with fast-gpu-asr, using its datasets and WER scorer and reporting both "
        "**RTFx and WER**. "
        f"[Methodology]({BENCHMARK_URL}methodology.md#timing-and-scoring).",
    ]

    return "\n".join(lines)


def overview(records: list[Measurement]) -> str:
    """Build the README performance block.

    The fastest configuration supplies the qualified headline and the GPU for
    both models' scaling/precision findings. Highlights still include every GPU.

    Parameters
    ----------
    records : list[Measurement]
        Sorted, validated batch series from ``load_measurements``.

    Returns
    -------
    str
        Markdown and HTML with absolute publication URLs, without README markers.
    """

    fastest = max(records, key=lambda r: r.rtfx)
    gpu = fastest.gpu
    planned = "/".join(g for g in COLORS if not any(r.gpu == g for r in records))
    by_key = {(r.model, r.precision, r.batch_size): r for r in records if r.gpu == gpu}
    scaling, gains = [], []
    bf16 = {}
    precision_gains, wer_spans = [], []
    gains_taper = True
    for model, name in MODELS.items():
        if (model, "fp16", 256) not in by_key:
            continue
        large = by_key[model, "fp16", 256].rtfx
        scaling.append(f"{large / by_key[model, 'fp16', 1].rtfx:.1f}x for {name}")
        gain = (large / by_key[model, "fp16", 128].rtfx - 1) * 100
        gains_taper = gains_taper and 0 < gain < 100
        gains.append(f"{gain:+.1f}% for {name}")
        if (model, "bf16", 256) in by_key:
            change = (by_key[model, "bf16", 256].rtfx / large - 1) * 100
            difference = "slightly" if 0 < abs(change) < 1 else f"{abs(change):.1f}%"
            comparison = f"{difference} {'lower' if change < 0 else 'higher'}"
            bf16.setdefault(comparison, []).append(name)
        if (model, "fp32", 128) in by_key:
            change = (
                by_key[model, "fp16", 128].rtfx / by_key[model, "fp32", 128].rtfx - 1
            ) * 100
            precision_gains.append(f"{change:+.1f}% for {name}")
        wers = [r.mean_wer for r in by_key.values() if r.model == model]
        wer_spans.append(f"{max(wers) - min(wers):.3f} percentage points for {name}")

    findings = []
    if scaling:
        findings += [
            f"- **Batching matters.** On {gpu} with FP16, batch 256 versus batch 1 "
            f"delivers {' and '.join(scaling)}.",
            f"- **{'Gains taper' if gains_taper else 'Larger batches'}.** On that "
            f"same GPU with FP16, doubling capacity from 128 to 256 changes "
            f"throughput by {' and '.join(gains)}.",
        ]
    if bf16:
        comparisons = [
            f"{comparison} for {' and '.join(names)}"
            for comparison, names in bf16.items()
        ]
        findings.append(
            f"- **BF16 is supported too.** At {gpu}/batch 256, BF16 throughput is "
            f"{' and '.join(comparisons)} than FP16. WER is recorded for every "
            "configuration; precision can marginally change outputs."
        )
    if precision_gains:
        findings.append(
            f"- **FP16 vs. FP32.** At {gpu}/batch 128, FP16 changes throughput "
            f"by {' and '.join(precision_gains)} relative to FP32."
        )
    if wer_spans:
        findings.append(
            "- **Mean-WER is consistent across precisions and batches.** "
            f"On {gpu}, the recorded mean-WER span across all measured precisions "
            f"and batches is {' and '.join(wer_spans)}."
        )
    pending = pending_results(records)

    return "\n\n".join(
        [
            f"## Batched speech recognition at up to {fastest.rtfx:,.0f} RTFx on {gpu}",
            *([pending] if pending else []),
            plot_grid(),
            "**RTFx** = total audio duration / total inference time: "
            "throughput, not request latency.",
            highlights(records),
            *(["### Key Observations", "\n".join(findings)] if findings else []),
            *([f"{planned} measurements are planned."] if planned else []),
            f"[All results]({BENCHMARK_URL}results.md) | "
            f"[Download CSV]({RAW_BENCHMARK_URL}measurements.csv)",
        ]
    )


def results_page(records: list[Measurement]) -> str:
    """Build detailed per-GPU timing and WER tables without the README overview.

    Parameters
    ----------
    records : list[Measurement]
        Sorted, validated batch series from ``load_measurements``.

    Returns
    -------
    str
        Markdown grouped by GPU, ending with a newline. Links assume the CSV
        and methodology are beside ``results.md``.
    """

    gpus = [gpu for gpu in COLORS if any(r.gpu == gpu for r in records)]
    planned = "/".join(g for g in COLORS if g not in gpus)
    pending = pending_results(records)
    sections = [
        "# Benchmark Results",
        f"{len(records)} complete configurations on {', '.join(gpus)}. Both models "
        "use decoder beam 6, with one measured pass per configuration over seven "
        "English datasets. FP32 covers batches 1-128; FP16 and BF16 also cover 256."
        + (f" {planned} measurements are planned, not plotted." if planned else ""),
        *([pending] if pending else []),
        "[Methodology and hardware](methodology.md) | "
        "[CSV and plot reproduction](methodology.md#rebuild-plots) | "
        "[Measurements CSV](measurements.csv)",
        "We reproduced the [Open ASR Leaderboard]"
        "(https://github.com/huggingface/open_asr_leaderboard) English evaluation "
        "with fast-gpu-asr, using its datasets and WER scorer and reporting both "
        "**RTFx and WER**.",
        "RTFx uses pooled real audio seconds divided by synchronized full-ASR "
        "seconds; padding, loading, file I/O, resampling, and scoring are excluded. "
        "Mean WER is the unweighted mean of seven upstream-rounded WERs. "
        "Precision labels retain exporter math defaults, including TF32 for FP32. "
        "Single-pass results do not estimate repeatability or establish that "
        "small differences are significant.",
        "Dataset columns below use the cleaned AMI, Earnings22, GigaSpeech, and "
        "VoxPopuli sets. Earnings22 chunks are reassembled before scoring. "
        "All WER values are percentages; timing and WER come from the same pass.",
    ]
    for gpu in gpus:
        group = [r for r in records if r.gpu == gpu]
        lines = [
            f"## {gpu}",
            "",
            "### Throughput",
            "",
            "| Model | Precision | Batch | RTFx | Suite time | Mean WER |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in group:
            lines.append(
                f"| {MODELS[r.model]} | {r.precision.upper()} | {r.batch_size}"
                f" | {r.rtfx:,.1f} | {r.inference_seconds:.2f} s | {r.mean_wer:.3f}% |"
            )
        lines += [
            "",
            "<details>",
            "<summary>All per-dataset WERs</summary>",
            "",
            "| Model | Precision | Batch | " + " | ".join(DATASETS.values()) + " |",
            "|---|---|---:|" + "---:|" * len(DATASETS),
        ]
        for r in group:
            lines.append(
                f"| {MODELS[r.model]} | {r.precision.upper()} | {r.batch_size} | "
                + " | ".join(f"{wer:.2f}" for wer in r.wers)
                + " |"
            )
        lines += ["", "</details>"]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def get_figure(
    records: list[Measurement], model: str, precisions: tuple[str, ...]
) -> go.Figure:
    """Create a Plotly figure with GPU colors and precision-specific line styles.

    Parameters
    ----------
    records : list[Measurement]
        All validated records, not just the plotted subset; their maximum RTFx
        determines the shared vertical limit. An FP32-only plot uses 20,000 RTFx.
    model : str
        Model key from MODELS.
    precisions : tuple[str, ...]
        Precision keys to overlay, in legend-row order.

    Returns
    -------
    plotly.graph_objects.Figure
        Throughput curves with logarithmic batch ticks and no error bars.
        FP16 uses solid lines, BF16 dashed lines, and FP32 dotted lines.
        All curves share axes; FP32 data stops at batch 128, without extending
        to the batch-256 points measured for FP16/BF16.
        GPU/precision pairs without measurements are omitted.

    Notes
    -----
    Only SVG export requires Kaleido and a browser.
    """

    extra_legend_rows = len(precisions) - 1
    batches, max_rtfx = BATCHES[:8], 20000
    if precisions != ("fp32",):
        max_rtfx = ceil(max(r.rtfx for r in records) * 1.05 / 5000) * 5000
        batches = BATCHES

    fig = go.Figure()
    for precision in precisions:
        for gpu, color in COLORS.items():
            group = sorted(
                (
                    r
                    for r in records
                    if (r.gpu, r.model, r.precision) == (gpu, model, precision)
                ),
                key=lambda r: r.batch_size,
            )
            if group:
                fig.add_scatter(
                    x=[r.batch_size for r in group],
                    y=[r.rtfx for r in group],
                    mode="lines+markers",
                    name=f"{gpu} {precision.upper()}",
                    legendgroup=gpu,
                    line={
                        "color": color,
                        "width": 5,
                        "dash": {"fp16": "solid", "bf16": "dash", "fp32": "dot"}[
                            precision
                        ],
                    },
                    marker={"size": 13},
                )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=700 + 60 * extra_legend_rows,
        font={"family": "Arial", "size": 24, "weight": 700},
        margin={"l": 115, "r": 40, "t": 95 + 60 * extra_legend_rows, "b": 100},
        legend={
            "orientation": "h",
            "y": 1.14 + 0.12 * extra_legend_rows,
            "x": 0.5,
            "xanchor": "center",
            "entrywidth": 230,
            "entrywidthmode": "pixels",
            "itemwidth": 60,
            "font": {"size": 34, "weight": 700},
            "traceorder": "normal",
        },
        xaxis={
            "title": {
                "text": "Batch size",
                "font": {"size": 32, "weight": 700},
                "standoff": 20,
            },
            "tickfont": {"size": 32, "weight": 700},
            "type": "log",
            "range": [-0.05, round(log10(max(batches)) + 0.05, 2)],
            "tickvals": batches,
            "ticktext": [str(batch) for batch in batches],
        },
        yaxis={
            "title": {"text": "RTFx", "font": {"size": 32, "weight": 700}},
            "tickfont": {"size": 32, "weight": 700},
            "rangemode": "tozero",
            "gridcolor": "gainsboro",
            "range": [0, max_rtfx],
            "tickformat": ",d",
        },
    )

    return fig


def main() -> None:
    """Render everything before writing, preserving files on rendering failures.

    Raises
    ------
    ValueError
        CSV validation fails or README markers are missing, repeated, or unordered.
    RuntimeError
        The README changes during rendering; generated files are not replaced.
    xml.etree.ElementTree.ParseError, KeyError
        An exported SVG has invalid XML or an unresolved local reference.

    Notes
    -----
    Recheck the README for concurrent edits before replacing generated files.
    Writes are not a multi-file transaction or lock; an I/O failure can leave
    partial updates. The README uses absolute publication URLs for PyPI. Links in
    ``results.md`` remain relative: its CSV and methodology must be in ``--output``;
    they are not copied. Unrelated files are untouched.
    Normalized SVG identifiers keep output stable with the same plotting stack.
    Plots must remain separate assets; their IDs would collide if combined inline.
    """

    args = parse_args()
    records = load_measurements(args.csv)
    original = args.readme.read_text(encoding="utf-8")
    if (
        original.count(START) != 1
        or original.count(END) != 1
        or original.index(START) >= original.index(END)
    ):
        raise ValueError("README must contain one ordered benchmark marker pair.")

    before, rest = original.split(START)
    _, after = rest.split(END)
    updated = f"{before}{START}\n\n{overview(records)}\n\n{END}{after}"
    outputs = {"results.md": results_page(records)}
    for model in MODELS:
        for precisions in PLOTS:
            svg = get_figure(records, model, precisions).to_image(format="svg").decode()
            # Normalize random identifiers without changing geometry or local links.
            root = ET.fromstring(svg)
            identifiers = [
                node.attrib["id"] for node in root.iter() if "id" in node.attrib
            ]
            replacements = {name: f"plot-{i}" for i, name in enumerate(identifiers)}
            for node in root.iter():
                for key, value in node.attrib.items():
                    if key == "id":
                        node.set(key, replacements[value])
                    elif value.startswith("url(#") and value.endswith(")"):
                        node.set(key, f"url(#{replacements[value[5:-1]]})")
                    elif key.endswith("href") and value.startswith("#"):
                        node.set(key, "#" + replacements[value[1:]])
                    elif key == "class":
                        node.set(key, sub(r"\btrace[a-f0-9]{6}\b", "trace", value))
            ET.register_namespace("", "http://www.w3.org/2000/svg")
            ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
            suffix = "-".join(precisions)
            outputs[f"{model.split('_')[0]}-{suffix}.svg"] = ET.tostring(
                root, encoding="unicode"
            )

    if args.readme.read_text(encoding="utf-8") != original:
        raise RuntimeError("README changed during rendering; no files replaced.")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output / name).write_text(content, encoding="utf-8")
    args.readme.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
