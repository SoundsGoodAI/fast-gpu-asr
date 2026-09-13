#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU coverage for the compact publication renderer, without private evidence."""

import csv
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import plotly.graph_objects as go
import pytest

from docs.benchmarks import render


def write_csv(path: Path, rows: list[dict]) -> Path:
    """Write test rows with the public CSV header."""
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=render.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def rows() -> list[dict]:
    """Create full matrices with distinct speeds and known, unequal dataset WERs."""
    return [
        {
            "gpu": gpu,
            "model": model,
            "precision": precision,
            "batch_size": batch,
            "beam": 6,
            "audio_seconds": 12345.0,
            "inference_seconds": 1000 / (gpu_index * model_index * batch * factor),
            **{f"wer_{name}": i + 0.25 for i, name in enumerate(render.DATASETS)},
        }
        for gpu_index, gpu in enumerate(("A100", "H200"), 1)
        for model_index, model in zip((2, 1), render.MODELS, strict=True)
        for precision, factor in (("fp16", 1), ("bf16", 0.9), ("fp32", 0.8))
        for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if precision != "fp32" or batch != 256
    ]


def test_load_sorts_and_derives_metrics(tmp_path, rows):
    path = write_csv(tmp_path / "measurements.csv", list(reversed(rows)))
    records = render.load_measurements(path)
    assert len(records) == 104
    for record, row in zip(records, rows, strict=True):
        assert (record.gpu, record.model, record.precision, record.batch_size) == (
            row["gpu"],
            row["model"],
            row["precision"],
            row["batch_size"],
        )
        assert record.rtfx == 12345.0 / row["inference_seconds"]
        assert record.wers == (0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25)
        assert record.mean_wer == 3.25


@pytest.mark.parametrize(
    "column,value",
    [
        ("gpu", "unknown"),
        ("gpu", "H100"),
        ("gpu", "B200"),
        ("model", "unknown"),
        ("precision", "int8"),
        ("beam", 1),
        ("batch_size", 3),
        ("batch_size", "1.5"),
        ("audio_seconds", 0),
        ("audio_seconds", "nan"),
        ("inference_seconds", 0),
        ("inference_seconds", -1),
        ("inference_seconds", "inf"),
        ("inference_seconds", "nan"),
        ("wer_ami_cleaned_test", ""),
        ("wer_ami_cleaned_test", -1),
        ("wer_ami_cleaned_test", "inf"),
        ("wer_ami_cleaned_test", "nan"),
    ],
)
def test_rejects_invalid_values(tmp_path, rows, column, value):
    rows[0][column] = value
    with pytest.raises(ValueError, match="CSV line 2"):
        render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))


@pytest.mark.parametrize("audio,inference", [(1e308, 1e-308), (1e-308, 1e308)])
def test_rejects_invalid_derived_rtfx(tmp_path, rows, audio, inference):
    rows[0].update(audio_seconds=audio, inference_seconds=inference)
    with pytest.raises(ValueError, match="CSV line 2: Timing totals"):
        render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))


def test_rejects_fp32_batch_256(tmp_path, rows):
    rows[8]["precision"] = "fp32"
    with pytest.raises(ValueError, match="outside the published matrix"):
        render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))


@pytest.mark.parametrize(
    "change,message",
    [
        ("empty", "No measurements"),
        ("missing", "Incomplete A100"),
        ("duplicate", "Duplicate configuration"),
    ],
)
def test_rejects_incomplete_or_duplicate_matrices(tmp_path, rows, change, message):
    if change == "empty":
        rows.clear()
    elif change == "missing":
        rows.pop(0)
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError, match=message):
        render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))


def test_requires_shared_audio_but_allows_roundoff_and_large_wer(tmp_path, rows):
    rows[0]["audio_seconds"] = math.nextafter(12345.0, float("inf"))
    rows[0]["wer_ami_cleaned_test"] = 101.0
    rows[1]["wer_ami_cleaned_test"] = 0.0
    path = write_csv(tmp_path / "measurements.csv", rows)
    assert render.load_measurements(path)[0].wers[0] == 101.0
    rows[0]["audio_seconds"] += 1
    with pytest.raises(ValueError, match="same real audio total"):
        render.load_measurements(write_csv(path, rows))


@pytest.mark.parametrize("malformed", ["missing_header", "short_row", "long_row"])
def test_rejects_bad_columns(tmp_path, rows, malformed):
    path = write_csv(tmp_path / "measurements.csv", rows)
    with open(path, newline="") as stream:
        table = list(csv.reader(stream))
    if malformed == "missing_header":
        table[0].pop()
    elif malformed == "short_row":
        table[1].pop()
    else:
        table[1].append("unexpected")
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerows(table)
    with pytest.raises(ValueError, match="column"):
        render.load_measurements(path)


def test_all_rows_and_scoped_highlights_are_rendered(tmp_path, rows):
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    page = render.results_page(records)
    overview = render.overview(records)
    assert overview.splitlines()[0] == (
        "## Batched speech recognition at up to 12,000 RTFx on H200"
    )
    assert (
        "</table>\n\n**RTFx** = total audio duration / total inference time: "
        "throughput, not request latency.\n"
        "Each configuration processes **3.4 hours of audio across seven English "
        "datasets**.\n\n**FP16, decoder beam 6:**"
    ) in overview
    assert "256.0x for Zipformer CR-CTC Transducer" in overview
    assert "10.0% lower for Zipformer CR-CTC Transducer and Parakeet V3 TDT" in overview
    assert "### Key Observations" in overview
    assert overview.endswith(
        f"[All results]({render.BENCHMARK_URL}results.md) | "
        f"[Download CSV]({render.RAW_BENCHMARK_URL}measurements.csv)"
    )
    highlights = render.highlights(records)
    table_rows = [line for line in highlights.splitlines() if "results.md#" in line]
    expected = [
        r for r in records if r.precision == "fp16" and r.batch_size in (1, 256)
    ]
    assert len(table_rows) == 4
    for line, small, large in zip(
        table_rows, expected[::2], expected[1::2], strict=True
    ):
        assert line == (
            f"| {small.gpu} | [{render.MODELS[small.model]}]"
            f"({render.BENCHMARK_URL}results.md#{small.gpu.lower()})"
            f" | {small.rtfx:,.1f} | {large.rtfx:,.1f}"
            f" | {small.inference_seconds:.2f} s | {large.inference_seconds:.2f} s"
            f" | {small.mean_wer:.3f}% | {large.mean_wer:.3f}% |"
        )
    assert "Batch size" not in highlights
    assert next(line for line in highlights.splitlines() if line.startswith("|")) == (
        "| GPU | Model | Batch&nbsp;1<br>RTFx | Batch&nbsp;256<br>RTFx"
        " | Batch&nbsp;1<br>Suite&nbsp;time | Batch&nbsp;256<br>Suite&nbsp;time"
        " | Batch&nbsp;1<br>Mean&nbsp;WER | Batch&nbsp;256<br>Mean&nbsp;WER |"
    )
    assert overview.count("<img ") == 2
    for family in ("zipformer", "parakeet"):
        assert f"{family}-fp16-bf16-fp32.svg" in overview
    assert "<img " not in page
    assert "| GPU | Model |" not in page
    assert "**FP16, decoder beam 6:**" not in page
    assert "Both models use decoder beam 6" in page
    assert "[Methodology](methodology.md#timing-and-scoring)" not in page
    for text in (page, overview):
        assert "B300 measurements are planned" in text
        assert "H100" not in text and "B200" not in text
        assert (
            "[Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)"
            in text
        )
        assert "reporting both **RTFx and WER**" in text
    assert "methodology.md#timing-and-scoring" in overview
    assert "[Methodology and hardware](methodology.md)" in page
    assert "[CSV and plot reproduction](methodology.md#rebuild-plots)" in page
    for gpu in ("A100", "H200"):
        section = page.split(f"## {gpu}\n", 1)[1].split("\n## ", 1)[0]
        assert section.count("| Zipformer CR-CTC Transducer |") == 52
        assert section.count("| Parakeet V3 TDT |") == 52
        assert section.count("0.25 | 1.25 | 2.25 | 3.25 | 4.25 | 5.25 | 6.25") == 52


def test_highlights_keep_batch_wers_separate(tmp_path, rows):
    for row in rows:
        if row["batch_size"] == 256:
            for dataset in render.DATASETS:
                row[f"wer_{dataset}"] += 7
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    table_rows = [
        line
        for line in render.highlights(records).splitlines()
        if "results.md#" in line
    ]
    assert len(table_rows) == 4
    for line in table_rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[6] == "3.250%"
        assert cells[7] == "10.250%"


def test_observations_use_matched_precision_batches_and_per_gpu_wer_spans(
    tmp_path, rows
):
    for row in rows:
        if row["gpu"] == "A100":
            row["wer_ami_cleaned_test"] += row["batch_size"]
            continue
        if row["batch_size"] == 128:
            if row["model"] == "zipformer_cr_ctc_rnnt":
                row["inference_seconds"] = 10 if row["precision"] == "fp16" else 20
            else:
                row["inference_seconds"] = 20 if row["precision"] == "fp16" else 10
        if row["precision"] == "fp16" and row["batch_size"] == 16:
            row["wer_ami_cleaned_test"] += (
                1.4 if row["model"] == "zipformer_cr_ctc_rnnt" else 0.7
            )
        if row["batch_size"] == 8:
            if (row["model"], row["precision"]) == ("zipformer_cr_ctc_rnnt", "bf16"):
                row["wer_ami_cleaned_test"] += 2.8
            if (row["model"], row["precision"]) == ("parakeet_v3", "fp32"):
                row["wer_ami_cleaned_test"] += 2.1
        if (row["model"], row["precision"], row["batch_size"]) == (
            "zipformer_cr_ctc_rnnt",
            "bf16",
            256,
        ):
            row["inference_seconds"] = 1000 / (2 * 2 * 256 * 1.005)
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    text = render.overview(records)
    assert "slightly higher for Zipformer CR-CTC Transducer" in text
    assert (
        "At H200/batch 128, FP16 changes throughput by +100.0% for "
        "Zipformer CR-CTC Transducer and -50.0% for Parakeet V3 TDT relative to FP32."
    ) in text
    assert (
        "On H200, the recorded mean-WER span across all measured precisions "
        "and batches is 0.400 percentage points for Zipformer CR-CTC Transducer and "
        "0.300 percentage points for Parakeet V3 TDT."
    ) in text


def test_bf16_observation_groups_matching_comparisons(tmp_path, rows):
    for row in rows:
        if row["precision"] == "bf16":
            row["inference_seconds"] *= 0.9 / 1.005
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    text = render.overview(records)
    assert (
        "BF16 throughput is slightly higher for Zipformer CR-CTC Transducer "
        "and Parakeet V3 TDT than FP16."
    ) in text


def test_plot_grid_places_models_in_equal_width_columns():
    table = ET.fromstring(render.plot_grid())
    rows = table.findall("./tbody/tr")
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)
    for header, cell, family, name in zip(
        rows[0].findall("th"),
        rows[1].findall("td"),
        ("zipformer", "parakeet"),
        render.MODELS.values(),
        strict=True,
    ):
        assert header.get("width") == "50%"
        assert header.find("div").get("align") == "center"
        assert header.find("div/big").text == f"{name} beam 6"
        link = cell.find("a")
        image = link.find("img")
        path = f"{render.RAW_BENCHMARK_URL}{family}-fp16-bf16-fp32.svg"
        assert cell.get("width") == "50%"
        assert link.get("href") == image.get("src") == path
        assert image.get("width") == "100%"
        assert image.get("alt") == f"{name}, FP16 / BF16 / FP32"


def test_figures_use_precision_axes_and_real_measurements(tmp_path, rows):
    for row in rows:
        row["inference_seconds"] /= 2
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    for model in render.MODELS:
        for precisions in (
            ("fp16",),
            ("bf16",),
            ("fp16", "bf16"),
            ("fp16", "bf16", "fp32"),
            ("fp32",),
        ):
            chart = render.get_figure(records, model, precisions)
            fp32_only = precisions == ("fp32",)
            assert chart.layout.xaxis.type == "log"
            assert chart.layout.xaxis.title.text == "Batch size"
            assert chart.layout.yaxis.title.text == "RTFx"
            assert chart.layout.legend.orientation == "h"
            assert chart.layout.legend.y > 1
            batches = render.BATCHES[:8] if fp32_only else render.BATCHES
            assert tuple(chart.layout.xaxis.tickvals) == batches
            assert tuple(chart.layout.xaxis.ticktext) == tuple(map(str, batches))
            assert chart.layout.xaxis.range == pytest.approx(
                (-0.05, 2.16 if fp32_only else 2.46)
            )
            assert chart.layout.yaxis.range == (
                0,
                20_000 if fp32_only else 27_000,
            )
            assert chart.layout.yaxis.dtick == 5_000
            series = [(gpu, p) for p in precisions for gpu in ("A100", "H200")]
            for trace, (gpu, precision) in zip(chart.data, series, strict=True):
                assert trace.name == f"{gpu} {precision.upper()}"
                assert trace.legendgroup == gpu
                expected = [
                    r
                    for r in records
                    if (r.gpu, r.model, r.precision) == (gpu, model, precision)
                ]
                assert tuple(trace.x) == tuple(r.batch_size for r in expected)
                assert tuple(trace.y) == tuple(r.rtfx for r in expected)
                assert max(trace.y) < chart.layout.yaxis.range[1]
                assert trace.line.color == {"A100": "deepskyblue", "H200": "lime"}[gpu]
                assert (
                    trace.line.dash
                    == {"fp16": "solid", "bf16": "dash", "fp32": "dot"}[precision]
                )
                assert len(trace.x) == (8 if precision == "fp32" else 9)
                assert trace.x[-1] == (128 if precision == "fp32" else 256)


def test_b300_adds_curves_without_changing_existing_data(tmp_path, rows):
    rows += [{**row, "gpu": "B300"} for row in rows if row["gpu"] == "H200"]
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    assert len(records) == 156
    chart = render.get_figure(records, "parakeet_v3", ("fp32",))
    assert [trace.name for trace in chart.data] == [
        "A100 FP32",
        "H200 FP32",
        "B300 FP32",
    ]
    assert chart.data[2].y == chart.data[1].y
    assert chart.data[2].line.color == "red"
    for text in (render.overview(records), render.results_page(records)):
        assert "measurements are planned" not in text
        assert "H100" not in text and "B200" not in text
    assert "results.md#b300" in render.overview(records)
    assert "## B300\n" in render.results_page(records)


def test_pending_b300_bf16_is_explicit_without_inventing_measurements(tmp_path, rows):
    b300 = [
        {**row, "gpu": "B300", "inference_seconds": row["inference_seconds"] / 2}
        for row in rows
        if row["gpu"] == "H200"
        and (row["model"], row["precision"]) != ("parakeet_v3", "bf16")
    ]
    path = write_csv(tmp_path / "measurements.csv", rows + b300)
    records = render.load_measurements(path)
    assert len(records) == 147
    for model in render.MODELS:
        for precisions in (("fp16", "bf16", "fp32"), ("fp32",)):
            chart = render.get_figure(records, model, precisions)
            expected = [
                f"{gpu} {precision.upper()}"
                for precision in precisions
                for gpu in ("A100", "H200", "B300")
                if (gpu, model, precision) != ("B300", "parakeet_v3", "bf16")
            ]
            assert [trace.name for trace in chart.data] == expected
    for text in (render.overview(records), render.results_page(records)):
        assert "B300: 43/52 configurations" in text
        assert "Pending: Parakeet V3 TDT BF16 (batches 1-256)" in text
        assert "B300 measurements are planned" not in text
    overview = render.overview(records)
    assert overview.splitlines()[0].endswith("25,000 RTFx on B300")
    bf16_finding = next(
        line for line in overview.splitlines() if line.startswith("- **BF16")
    )
    assert "Zipformer CR-CTC Transducer" in bf16_finding
    assert "Parakeet" not in bf16_finding

    completed = [
        {**row, "gpu": "B300"}
        for row in rows
        if row["gpu"] == "H200"
        and (row["model"], row["precision"]) == ("parakeet_v3", "bf16")
    ]
    records = render.load_measurements(write_csv(path, rows + b300 + completed))
    assert len(records) == 156
    assert render.pending_results(records) == ""


def test_absent_series_do_not_require_unmeasured_comparisons(tmp_path, rows):
    rows = [
        row
        for row in rows
        if (row["gpu"], row["model"], row["precision"])
        == ("A100", "zipformer_cr_ctc_rnnt", "bf16")
    ]
    records = render.load_measurements(write_csv(tmp_path / "measurements.csv", rows))
    text = render.overview(records)
    assert "A100: 9/52 configurations" in text
    assert "Key Observations" not in text
    chart = render.get_figure(records, "parakeet_v3", ("fp32",))
    assert len(chart.data) == 0
    assert chart.layout.xaxis.tickvals[-1] == 128


def test_parse_args_defaults_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["render"])
    root = Path(render.__file__).resolve().parent
    assert vars(render.parse_args()) == {
        "csv": root / "measurements.csv",
        "output": root,
        "readme": root.parents[1] / "README.md",
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        render.START,
        render.END + render.START,
        render.START + render.START + render.END,
        render.START + render.END + render.END,
    ],
)
def test_rejects_bad_readme_markers(tmp_path, rows, monkeypatch, text):
    path = write_csv(tmp_path / "measurements.csv", rows)
    readme = tmp_path / "README.md"
    readme.write_text(text)
    output = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--csv",
            str(path),
            "--readme",
            str(readme),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        go.Figure,
        "to_image",
        lambda self, **kwargs: pytest.fail("Invalid markers must fail before export"),
    )
    with pytest.raises(ValueError, match="one ordered benchmark marker pair"):
        render.main()
    assert readme.read_text() == text
    assert not output.exists()


@pytest.mark.parametrize("failure", ["export", "malformed_svg", "readme_edit"])
def test_render_failure_preserves_existing_publication(
    tmp_path, rows, monkeypatch, failure
):
    path = write_csv(tmp_path / "measurements.csv", rows)
    readme = tmp_path / "README.md"
    original = f"intro\n{render.START}\nold\n{render.END}\n"
    readme.write_text(original)
    output = tmp_path / "plots"
    output.mkdir()
    (output / "results.md").write_text("old results")
    (output / "zipformer-fp16-bf16-fp32.svg").write_text("old plot")
    before = {path: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--csv",
            str(path),
            "--output",
            str(output),
            "--readme",
            str(readme),
        ],
    )
    calls = 0

    def export(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "export":
                raise RuntimeError("Browser unavailable")
            if failure == "malformed_svg":
                return b"<broken>"
            readme.write_text("concurrent edit")
        return b"<svg />"

    monkeypatch.setattr(go.Figure, "to_image", export)
    error = ET.ParseError if failure == "malformed_svg" else RuntimeError
    with pytest.raises(error):
        render.main()
    assert calls == 2
    assert readme.read_text() == (
        "concurrent edit" if failure == "readme_edit" else original
    )
    assert before == {path: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize(
    "readme_name,output_name",
    [
        ("pages/README.md", "plots & charts"),
        ("pages/README.md", "."),
        ("README.md", "."),
    ],
)
def test_main_keeps_publication_urls_independent_of_output_paths(
    tmp_path, rows, monkeypatch, readme_name, output_name
):
    monkeypatch.chdir(tmp_path)
    path = write_csv(tmp_path / "measurements.csv", rows)
    readme = Path(readme_name)
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(f"{render.START}\nold\n{render.END}\n")
    output = tmp_path / output_name
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--csv",
            str(path),
            "--output",
            str(output),
            "--readme",
            str(readme),
        ],
    )
    monkeypatch.setattr(go.Figure, "to_image", lambda self, **kwargs: b"<svg />")
    render.main()
    text = readme.read_text()
    assert (
        f"[Methodology]({render.BENCHMARK_URL}methodology.md#timing-and-scoring)"
        in text
    )
    chart = render.RAW_BENCHMARK_URL + "zipformer-fp16-bf16-fp32.svg"
    assert f'href="{chart}"' in text
    assert f'src="{chart}"' in text
    assert (output / "zipformer-fp16-bf16-fp32.svg").is_file()


def test_main_writes_only_generated_files_and_is_repeatable(
    tmp_path, rows, monkeypatch
):
    path = write_csv(tmp_path / "measurements.csv", rows)
    readme = tmp_path / "README.md"
    readme.write_text(f"manual intro\n{render.START}\nold\n{render.END}\nmanual end\n")
    output = tmp_path / "plots"
    output.mkdir()
    (output / "manual.md").write_text("keep")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--csv",
            str(path),
            "--output",
            str(output),
            "--readme",
            str(readme),
        ],
    )
    svg = """<svg xmlns="http://www.w3.org/2000/svg"
        xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs><clipPath id="random" /></defs>
        <g clip-path="url(#random)" class="trace tracea1b2c3 scatter">
        <use xlink:href="#random" /></g></svg>"""
    monkeypatch.setattr(go.Figure, "to_image", lambda self, **kwargs: svg.encode())
    render.main()
    expected = {"manual.md", "results.md"} | {
        f"{model}-fp16-bf16-fp32.svg" for model in ("zipformer", "parakeet")
    }
    assert {p.name for p in output.iterdir()} == expected
    assert (output / "manual.md").read_text() == "keep"
    records = render.load_measurements(path)
    assert (output / "results.md").read_text() == render.results_page(records)
    text = readme.read_text()
    assert text == (
        f"manual intro\n{render.START}\n\n"
        f"{render.overview(records)}\n\n{render.END}\nmanual end\n"
    )
    assert (
        f"[Methodology]({render.BENCHMARK_URL}methodology.md#timing-and-scoring)"
        in text
    )
    for path in output.glob("*.svg"):
        root = ET.parse(path).getroot()
        group = root.find("{http://www.w3.org/2000/svg}g")
        assert group.attrib["clip-path"] == "url(#plot-0)"
        assert group[0].attrib["{http://www.w3.org/1999/xlink}href"] == "#plot-0"
        assert group.attrib["class"] == "trace trace scatter"
    before = {p: p.read_bytes() for p in [readme, *output.iterdir()]}
    for replacement in (
        svg.replace("random", "different").replace("tracea1b2c3", "traced4e5f6"),
        (output / "zipformer-fp16-bf16-fp32.svg").read_text(),
    ):
        svg = replacement
        render.main()
        assert before == {p: p.read_bytes() for p in before}
