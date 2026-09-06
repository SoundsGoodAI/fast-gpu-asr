#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU report validation and publication tests; static browser export is mocked."""

import copy
import hashlib
import json
import math
import runpy
import shutil
import statistics
import sys
from pathlib import Path
from unittest.mock import Mock

import plotly.graph_objects as go
import pytest

from benchmarks import report
from benchmarks.common import (
    BEAMS,
    DATASETS,
    MODELS,
    PROTOCOL,
    JSONValue,
    file_hash,
    write_json,
)


@pytest.fixture
def campaign() -> dict[str, JSONValue]:
    """Supply seven datasets with unequal durations and valid parent references.

    Returns
    -------
    dict[str, JSONValue]
        Frozen protocol, model identities, and rows with private reference text.
    """

    value = {
        "protocol": {**PROTOCOL, "passes": 3},
        "source": {"commit": "frozen", "files": {"benchmark.py": "a" * 64}},
        "models": {
            name: {"repo": repo, "revision": "b" * 40} for name, repo in MODELS.items()
        },
        "datasets": {},
    }
    for i, name in enumerate(DATASETS, 1):
        value["datasets"][name] = [
            {
                "id": str(j),
                "path": f"{name}/{j}.wav",
                "sha256": "c" * 64,
                "frames": duration * i * 16000,
                "rate": 16000,
                "text": "private reference",
                **(
                    {"parent_id": "session", "chunk_index": j}
                    if name == DATASETS[1]
                    else {}
                ),
            }
            for j, duration in enumerate((4, 1, 2))
        ]
    value["id"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def write_run(
    campaign: dict[str, JSONValue],
    directory: Path,
    batch: int = 2,
    precision: str = "fp16",
    speed: float = 1,
    model: str = "zipformer_cr_ctc_rnnt",
) -> dict[str, JSONValue]:
    """Write consistent evidence using independent batch and timing formulas.

    Parameters
    ----------
    campaign : dict[str, JSONValue]
        Frozen dataset rows, protocol, and campaign identity.
    directory : Path
        New run directory receiving metadata and scored passes.
    batch : int
        Capacity for longest-first rows, including partial batches.
    precision : str
        Precision recorded in the run specification.
    speed : float
        Divisor applied to asymmetric pass timings.
    model : str
        Model variant whose configured beam is recorded in the run.

    Returns
    -------
    dict[str, JSONValue]
        Saved result, allowing tests to change scores and their matching copies.
    """

    directory.mkdir(parents=True)
    spec = {
        "campaign_id": campaign["id"],
        "model": model,
        "gpu": "H100",
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "batch_size": batch,
        "precision": precision,
        "beam": BEAMS[model],
        "started_at": "2026-09-05T00:00:00+00:00",
    }
    machine = {
        "gpu": {
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": spec["gpu_uuid"],
            "memory.total": "81920",
        },
        "cpu_affinity": [0, 1],
        "cpu_quota": None,
        "packages": {"numpy": "test"},
    }
    result = {
        "status": "complete",
        "campaign_id": campaign["id"],
        "spec": spec,
        "hardware": machine,
        "bundle_files": {"encoder.trt": "d" * 64},
        "plugins": {"feature.so": "e" * 64},
        "passes": [],
        "evidence": {},
    }
    for name, value in (
        ("run.json", spec),
        ("hardware.json", machine),
        ("bundle-hashes.json", result["bundle_files"]),
        ("plugin-hashes.json", result["plugins"]),
    ):
        write_json(directory / name, value)
    for index in range(1, campaign["protocol"]["passes"] + 1):
        factor = (2, 4, 1)[(index - 1) % 3]
        folder = directory / f"pass-{index}"
        folder.mkdir()
        records, datasets = [], {}
        for i, (name, rows) in enumerate(campaign["datasets"].items(), 1):
            rows = sorted(rows, key=lambda r: (-r["frames"] / r["rate"], r["id"]))
            seconds = elapsed = 0
            for j, offset in enumerate(range(0, len(rows), batch), 1):
                group = rows[offset : offset + batch]
                duration = sum(r["frames"] / r["rate"] for r in group)
                timing = i * i * j * factor / speed
                records.append(
                    {
                        "dataset": name,
                        "ids": [r["id"] for r in group],
                        "audio_seconds": duration,
                        "inference_seconds": timing,
                    }
                )
                seconds += duration
                elapsed += timing
            predictions = [
                {
                    "id": row["id"],
                    "pred_text": "" if j == 0 else "private hypothesis",
                    "word_timestamps": [],
                }
                for j, row in enumerate(rows)
            ]
            (folder / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
            datasets[name] = {
                "wer": i + index / 10,
                "ins": 1,
                "del": 2,
                "sub": 3,
                "samples": len(rows),
                "audio_length": seconds,
                "inference_time": elapsed,
                "rtfx": seconds / elapsed,
            }
        (folder / "batches.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        seconds = sum(r["audio_seconds"] for r in records)
        elapsed = sum(r["inference_seconds"] for r in records)
        saved = {
            "audio_seconds": seconds,
            "inference_seconds": elapsed,
            "rtfx": seconds / elapsed,
            "datasets": datasets,
            "mean_wer_percent": statistics.mean(s["wer"] for s in datasets.values()),
        }
        result["passes"].append(saved)
        write_json(folder / "scores.json", saved)
        for path in folder.glob("*.jsonl"):
            result["evidence"][path.relative_to(directory).as_posix()] = file_hash(path)
    write_json(
        directory / "collection.json",
        {
            "campaign_id": campaign["id"],
            "source": campaign["source"],
            "evidence": result["evidence"],
        },
    )
    write_json(directory / "result.json", result)
    return result


@pytest.fixture
def run_dir(tmp_path: Path, campaign: dict[str, JSONValue]) -> Path:
    """Create complete passes with partial batches and asymmetric timings.

    Parameters
    ----------
    tmp_path : Path
        Isolated root for run evidence.
    campaign : dict[str, JSONValue]
        Frozen campaign consumed by ``write_run``.

    Returns
    -------
    Path
        Complete Zipformer FP16 run with capacity two.
    """

    directory = tmp_path / "runs" / "H100-zipformer_cr_ctc_rnnt-fp16-b2"
    write_run(campaign, directory)
    return directory


@pytest.fixture
def export_images(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record real figures and write placeholder SVGs without Chrome or Kaleido.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores the static-export function after the test.

    Returns
    -------
    Mock
        Records figures and output paths, and can simulate export failures.
        Successful calls leave SVGs to exercise staging and image publication.
    """

    def write_images(file: list[Path], **kwargs: list[go.Figure] | str | int) -> None:
        """Write an SVG placeholder to each path passed to Plotly's exporter."""
        for path in file:
            path.write_text("<svg/>")

    export = Mock(side_effect=write_images)
    monkeypatch.setattr(report.pio, "write_images", export)
    return export


@pytest.fixture
def run_report(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """Invoke the real CLI with private campaign and matrix files outside output."""
    private = tmp_path_factory.mktemp("report-inputs")

    def invoke(
        campaign: dict[str, JSONValue],
        directories: list[Path],
        output: Path,
        readme: Path | None = None,
    ) -> None:
        """List attempts in matrices so missing and interrupted runs are discovered."""
        path = private / "campaign.json"
        write_json(path, campaign)
        directories = directories or [private / "unstarted"]
        roots = sorted({directory.parent for directory in directories})
        for root in roots:
            write_json(
                root / "matrix.json",
                sorted({p.name for p in directories if p.parent == root}),
            )
        argv = [
            "report",
            "--campaign",
            str(path),
            "--runs",
            *map(str, roots),
            "--output",
            str(output),
        ]
        if readme is not None:
            argv.extend(("--readme", str(readme)))
        monkeypatch.setattr(sys, "argv", argv)
        report.main()

    return invoke


def rewrite_evidence(
    directory: Path, name: str, rows: list[dict[str, JSONValue]]
) -> None:
    """Replace JSONL rows and refresh both hashes to test content validation.

    ``name`` is relative to the run directory. Result and collection hashes stay
    consistent, so rejection must come from the evidence contents, not provenance.
    """
    path = directory / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for filename in ("result.json", "collection.json"):
        value = json.loads((directory / filename).read_text())
        value["evidence"][name] = file_hash(path)
        write_json(directory / filename, value)


@pytest.mark.parametrize("empty", (False, True))
def test_render_preserves_plot_axes_ranges_and_missing_measurements(
    tmp_path: Path, export_images: Mock, campaign, empty: bool
) -> None:
    results = [
        {
            "status": "complete",
            "spec": {
                "model": model,
                "precision": precision,
                "gpu": gpu,
                "batch_size": batch,
            },
            "rtfx_median": median,
            "rtfx_range": [low, high],
            "inference_seconds_median": 1.0,
            "passes": [{"mean_wer_percent": 1.0} for _ in range(3)],
        }
        for model, precision, gpu, batch, median, low, high in (
            ("zipformer_cr_ctc_rnnt", "fp16", "A100", 64, 40, 30, 45),
            ("parakeet_v3", "bf16", "H200", 128, 80, 60, 100),
            ("zipformer_cr_ctc_rnnt", "fp16", "A100", 1, 10, 8, 12),
            ("zipformer_rnnt", "fp32", "H100", 2, 20, 15, 25),
            ("parakeet_v2", "fp16", "B200", 16, 60, 50, 70),
        )
        if not empty
    ]
    original = copy.deepcopy(results)

    report.render_report(campaign, results, tmp_path, ".")

    assert results == original
    export_images.assert_called_once()
    figures = export_images.call_args.kwargs["fig"]
    paths = export_images.call_args.kwargs["file"]
    assert export_images.call_args.kwargs["format"] == "svg"
    assert [path.name for path in paths] == [
        f"{model}-{precision}.svg"
        for precision in ("fp32", "fp16", "bf16")
        for model in (
            "zipformer_rnnt",
            "zipformer_cr_ctc_rnnt",
            "parakeet_v2",
            "parakeet_v3",
        )
    ]
    assert all(path.parent == tmp_path for path in paths)
    expected = (
        {}
        if empty
        else {
            ("zipformer_cr_ctc_rnnt", "fp16", "A100"): (
                (1, 64),
                (10, 40),
                (2, 5),
                (2, 10),
            ),
            ("parakeet_v3", "bf16", "H200"): ((128,), (80,), (20,), (20,)),
            ("zipformer_rnnt", "fp32", "H100"): ((2,), (20,), (5,), (5,)),
            ("parakeet_v2", "fp16", "B200"): ((16,), (60,), (10,), (10,)),
        }
    )
    for figure, path in zip(figures, paths, strict=True):
        model, precision = path.stem.split("-")
        assert figure.layout.title.text == f"{model.title()} / {precision.upper()}"
        assert figure.layout.xaxis.type == "log"
        assert figure.layout.xaxis.range == pytest.approx(
            [math.log10(0.85), math.log10(300)]
        )
        assert figure.layout.xaxis.tickvals == (1, 2, 4, 8, 16, 32, 64, 128, 256)
        assert figure.layout.xaxis.ticktext == tuple(
            map(str, figure.layout.xaxis.tickvals)
        )
        assert figure.layout.yaxis.range == pytest.approx([0, 1.12 if empty else 112])
        assert [trace.name for trace in figure.data] == [
            "A100",
            "H100",
            "H200",
            "B200",
            "B300",
        ]
        for trace in figure.data:
            assert (
                trace.line.color
                == trace.marker.color
                == trace.error_y.color
                == report.COLORS[trace.name]
            )
            assert (
                trace.x,
                trace.y,
                trace.error_y.array,
                trace.error_y.arrayminus,
            ) == expected.get(
                (model, precision, trace.name), ((None,), (None,), (), ())
            )
            assert trace.error_y.symmetric is False
        has_data = any((model, precision, gpu) in expected for gpu in report.GPUS)
        assert [a.text for a in figure.layout.annotations] == (
            [] if has_data else ["No complete suite measurements yet"]
        )


@pytest.mark.parametrize(
    ("readme_name", "output_name", "prefix", "newline"),
    (
        ("README.md", "docs/benchmarks", "docs/benchmarks", "\n"),
        ("docs/README.md", "plots (new)", "../plots%20%28new%29", "\r\n"),
        ("README.md", ".", ".", "\n"),
    ),
)
def test_main_links_svgs_and_preserves_readme(
    tmp_path: Path,
    export_images: Mock,
    run_report,
    campaign,
    readme_name: str,
    output_name: str,
    prefix: str,
    newline: str,
) -> None:
    readme = tmp_path / readme_name
    readme.parent.mkdir(parents=True, exist_ok=True)
    start, end = "<!-- benchmark-results:start -->", "<!-- benchmark-results:end -->"
    before = b"\xef\xbb\xbfBefore" + newline.encode()
    after = newline.encode() + b"After" + newline.encode()
    readme.write_bytes(
        before + f"{start}{newline}old figures{newline}{end}".encode() + after
    )
    readme.chmod(0o640)
    output = tmp_path / output_name

    run_report(campaign, [], output, readme)

    updated = readme.read_bytes()
    assert updated.startswith(before + start.encode())
    assert updated.endswith(end.encode() + after)
    if newline == "\r\n":
        assert b"\n" not in updated.replace(b"\r\n", b"")
    updated = updated.decode()
    assert readme.stat().st_mode & 0o777 == 0o640
    export_images.assert_called_once()
    for path in export_images.call_args.kwargs["file"]:
        assert f"({prefix}/{path.name})" in updated
        assert (output / path.name).read_text() == "<svg/>"
    for model, beam in BEAMS.items():
        assert f"{model} (beam {beam})" in updated
    assert "No speedup or accuracy-parity claims yet" in updated
    assert json.loads((output / "results.json").read_text())["runs"] == []
    assert "private reference" not in (output / "manifest.json").read_text()
    assert "Complete: 0" in (output / "results.md").read_text()


@pytest.mark.parametrize(
    "model,passes,batch,elapsed",
    (
        ("zipformer_rnnt", 1, 1, 840),
        ("zipformer_cr_ctc_rnnt", 2, 2, 420),
        ("parakeet_v2", 3, 64, 140),
        ("parakeet_v3", 4, 256, 140),
    ),
)
def test_report_supports_models_batches_and_pass_counts(
    tmp_path: Path,
    export_images: Mock,
    run_report,
    campaign,
    model,
    passes,
    batch,
    elapsed,
) -> None:
    campaign["protocol"]["passes"] = passes
    campaign["id"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in campaign.items() if k != "id"}, sort_keys=True
        ).encode()
    ).hexdigest()
    directory = tmp_path / "run"
    write_run(campaign, directory, model=model, batch=batch)
    output = tmp_path / "public"

    run_report(campaign, [directory], output)

    result = json.loads((output / "results.json").read_text())["runs"][0]
    assert len(result["passes"]) == passes
    assert result["spec"]["model"] == model
    assert result["spec"]["batch_size"] == batch
    rtfx = [196 / (elapsed * factor) for factor in (2, 4, 1, 2)[:passes]]
    assert result["rtfx_median"] == pytest.approx(statistics.median(rtfx))
    assert result["rtfx_range"] == pytest.approx([min(rtfx), max(rtfx)])
    markdown = (output / "results.md").read_text()
    if passes == 1:
        assert "Single measured pass per configuration" in markdown
        assert "RTFx (range)" not in markdown
    else:
        assert f"minimum and maximum of {passes} complete passes" in markdown
        assert "RTFx (range)" in markdown


def test_validate_recomputes_summaries_without_changing_evidence(campaign, run_dir):
    original = json.loads((run_dir / "result.json").read_text())
    original.update(
        rtfx_median=999, rtfx_range=[999, 999], inference_seconds_median=999
    )
    original["passes"][0]["datasets"][DATASETS[0]]["wer"] = 125
    for i, saved in enumerate(original["passes"], 1):
        saved["mean_wer_percent"] = 999
        write_json(run_dir / f"pass-{i}/scores.json", saved)
    write_json(run_dir / "result.json", original)
    before = {p: file_hash(p) for p in run_dir.rglob("*") if p.is_file()}

    result = report.validate_result(campaign, run_dir)

    assert result["rtfx_median"] == pytest.approx(196 / 840)
    assert result["rtfx_range"] == pytest.approx([196 / 1680, 196 / 420])
    assert result["inference_seconds_median"] == 840
    assert result["passes"][0]["mean_wer_percent"] == pytest.approx((125 + 27.6) / 7)
    assert result["passes"][0]["datasets"][DATASETS[0]]["wer"] == 125
    assert result["rtfx_median"] != pytest.approx(
        statistics.mean(d["rtfx"] for d in result["passes"][0]["datasets"].values())
    )
    assert {p: file_hash(p) for p in before} == before
    assert result["evidence"] == original["evidence"]


def test_validate_uses_frozen_dataset_durations_despite_timing_roundoff(
    campaign: dict[str, JSONValue], run_dir: Path
) -> None:
    name = "pass-1/batches.jsonl"
    path = run_dir / name
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["audio_seconds"] += 1e-12
    rewrite_evidence(run_dir, name, records)

    result = report.validate_result(campaign, run_dir)

    dataset = result["passes"][0]["datasets"][DATASETS[0]]
    assert dataset["audio_length"] == 7
    assert dataset["rtfx"] == 7 / 6
    assert result["passes"][0]["audio_seconds"] == 196


def test_validate_rejects_zero_total_audio(tmp_path: Path, campaign) -> None:
    for rows in campaign["datasets"].values():
        for row in rows:
            row["frames"] = 0
    campaign["id"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in campaign.items() if k != "id"}, sort_keys=True
        ).encode()
    ).hexdigest()
    directory = tmp_path / "run"
    write_run(campaign, directory)

    with pytest.raises(ValueError, match="Invalid pooled timing"):
        report.validate_result(campaign, directory)


@pytest.mark.parametrize(
    "filename",
    (
        "collection.json",
        "run.json",
        "hardware.json",
        "bundle-hashes.json",
        "plugin-hashes.json",
        "pass-2/scores.json",
        f"pass-3/{DATASETS[-1]}.jsonl",
    ),
)
def test_validate_requires_supporting_artifacts(campaign, run_dir, filename):
    (run_dir / filename).unlink()
    with pytest.raises(FileNotFoundError):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize(
    "filename,field,value,message",
    (
        ("collection.json", "source", {}, "provenance"),
        ("collection.json", "campaign_id", "other", "provenance"),
        ("collection.json", "evidence", {}, "provenance"),
        ("run.json", "precision", "fp32", "Run metadata"),
        ("hardware.json", "cpu_affinity", [2], "Hardware metadata"),
        ("bundle-hashes.json", "encoder.trt", "f" * 64, "hash inventory"),
        ("plugin-hashes.json", "feature.so", "f" * 64, "hash inventory"),
        ("pass-1/scores.json", "rtfx", 999, "saved pass scores"),
    ),
)
def test_validate_rejects_inconsistent_snapshots(
    campaign, run_dir, filename, field, value, message
):
    data = json.loads((run_dir / filename).read_text())
    data[field] = value
    write_json(run_dir / filename, data)
    with pytest.raises(ValueError, match=message):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize(
    "keys,value,message",
    (
        (("campaign_id",), "other", "Different campaign"),
        (("status",), "success", "Unknown run status"),
        (("spec", "campaign_id"), "other", "Run metadata"),
        (("spec", "model"), "unknown", "outside the benchmark matrix"),
        (("spec", "gpu"), "T4", "outside the benchmark matrix"),
        (("spec", "precision"), "int8", "outside the benchmark matrix"),
        (("spec", "beam"), 1, "outside the benchmark matrix"),
        (("spec", "beam"), 6.0, "outside the benchmark matrix"),
        (("spec", "batch_size"), 3, "outside the benchmark matrix"),
        (("spec", "batch_size"), 2.0, "outside the benchmark matrix"),
        (("hardware", "gpu", "uuid"), "other", "Hardware metadata"),
        (("hardware", "gpu", "name"), "NVIDIA H200", "Hardware metadata"),
        (("passes",), [], "complete passes"),
        (("passes", 0, "rtfx"), 999, "batch timings"),
        (("passes", 0, "rtfx"), "1", "batch timings"),
        (("passes", 0, "rtfx"), float("inf"), "batch timings"),
        (("passes", 0, "datasets"), {}, "Missing WER"),
        (("passes", 0, "datasets", DATASETS[0], "wer"), -1, "Invalid WER"),
        (("passes", 0, "datasets", DATASETS[0], "wer"), "1.0", "Invalid WER"),
        (("passes", 0, "datasets", DATASETS[0], "wer"), float("inf"), "Invalid WER"),
        (("passes", 0, "datasets", DATASETS[0], "wer"), float("nan"), "Invalid WER"),
        (("passes", 0, "datasets", DATASETS[0], "ins"), -1, "error counts"),
        (("passes", 0, "datasets", DATASETS[0], "sub"), 1.5, "error counts"),
        (("passes", 0, "datasets", DATASETS[0], "samples"), 2, "Dataset scores"),
        (("passes", 0, "datasets", DATASETS[0], "audio_length"), 999, "Dataset scores"),
        (
            ("passes", 0, "datasets", DATASETS[0], "inference_time"),
            999,
            "Dataset scores",
        ),
        (("passes", 0, "datasets", DATASETS[0], "rtfx"), 999, "Dataset scores"),
    ),
)
def test_validate_rejects_invalid_values_even_with_matching_snapshots(
    campaign, run_dir, keys, value, message
):
    result = json.loads((run_dir / "result.json").read_text())
    target = result
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    (run_dir / "result.json").write_text(json.dumps(result))
    # Keep copies consistent so value validation, not snapshot mismatch, must fail.
    write_json(run_dir / "run.json", result["spec"])
    write_json(run_dir / "hardware.json", result["hardware"])
    for i, saved in enumerate(result["passes"], 1):
        (run_dir / f"pass-{i}/scores.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=message):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize("name", (DATASETS[0], "batches"))
def test_validate_rejects_changed_evidence(campaign, run_dir: Path, name: str) -> None:
    path = run_dir / f"pass-1/{name}.jsonl"
    path.write_text(" " + path.read_text())
    with pytest.raises(ValueError, match="Evidence changed"):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize("extra", (False, True))
def test_validate_requires_exact_evidence_inventory(campaign, run_dir, extra):
    for filename in ("result.json", "collection.json"):
        result = json.loads((run_dir / filename).read_text())
        if extra:
            result["evidence"]["../private.jsonl"] = "a" * 64
        else:
            del result["evidence"][f"pass-1/{DATASETS[0]}.jsonl"]
        write_json(run_dir / filename, result)
    with pytest.raises(ValueError, match="Missing transcript or batch evidence hashes"):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize(
    "ids,hypothesis,message",
    (
        ([], "", "Incomplete or reordered transcripts"),
        (["0", "2"], "", "Incomplete or reordered transcripts"),
        (["0", "2", "1", "extra"], "", "Incomplete or reordered transcripts"),
        (["1", "2", "0"], "", "Incomplete or reordered transcripts"),
        (["0", "2", "1"], None, "Hypotheses must be strings"),
    ),
)
def test_validate_rejects_invalid_transcripts(
    campaign, run_dir, ids, hypothesis, message
):
    rewrite_evidence(
        run_dir,
        f"pass-1/{DATASETS[0]}.jsonl",
        [{"id": identifier, "pred_text": hypothesis} for identifier in ids],
    )
    with pytest.raises(ValueError, match=message):
        report.validate_result(campaign, run_dir)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("dataset", "unknown", "dataset, order, or utterance mismatch"),
        ("ids", ["2", "1"], "dataset, order, or utterance mismatch"),
        ("audio_seconds", 999, "Audio duration includes padding"),
        ("inference_seconds", 0, "Invalid inference time"),
        ("inference_seconds", float("inf"), "Invalid inference time"),
        ("inference_seconds", "1", "Batch timings must be numbers"),
    ),
)
def test_validate_rejects_invalid_batch_timings(
    campaign, run_dir, field, value, message
):
    name = "pass-1/batches.jsonl"
    rows = [json.loads(line) for line in (run_dir / name).read_text().splitlines()]
    rows[0][field] = value
    rewrite_evidence(run_dir, name, rows)
    with pytest.raises(ValueError, match=message):
        report.validate_result(campaign, run_dir)


def test_main_publishes_only_complete_runs_and_redacts_private_text(
    tmp_path: Path, export_images: Mock, run_report, campaign
) -> None:
    reference = "private reference caf\u00e9"
    campaign["datasets"][DATASETS[0]][0]["text"] = reference
    campaign["datasets"][DATASETS[0]][0]["original_text"] = "private reference"
    campaign["notes"] = "private reference"
    campaign["id"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in campaign.items() if k != "id"}, sort_keys=True
        ).encode()
    ).hexdigest()
    run_dir = tmp_path / "runs" / "H100-zipformer_cr_ctc_rnnt-fp16-b2"
    result = write_run(campaign, run_dir)
    result["diagnostics"] = "private hypothesis"
    result["spec"]["private_note"] = "private reference"
    for index, saved in enumerate(result["passes"], 1):
        saved["datasets"][DATASETS[0]]["reference_text"] = reference
        write_json(run_dir / f"pass-{index}/scores.json", saved)
    write_json(run_dir / "run.json", result["spec"])
    write_json(run_dir / "result.json", result)
    failed = tmp_path / "retry" / run_dir.name
    shutil.copytree(run_dir, failed)
    result = json.loads((failed / "result.json").read_text())
    result.update(
        status="failed",
        stage="score",
        error="private reference",
        traceback="private hypothesis",
        diagnostics="private reference",
    )
    write_json(failed / "result.json", result)
    (failed / "collection.json").unlink()
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    output = tmp_path / "public"
    before = {p: file_hash(p) for p in run_dir.rglob("*") if p.is_file()}

    run_report(campaign, [run_dir, failed, incomplete, tmp_path / "missing"], output)

    assert {p: file_hash(p) for p in before} == before
    runs = json.loads((output / "results.json").read_text())["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"
    assert runs[0]["run"] == f"runs/{run_dir.name}"
    for path in output.iterdir():
        text = path.read_text()
        assert "private reference" not in text
        assert "private hypothesis" not in text
        assert "diagnostics" not in text
    manifest = json.loads((output / "manifest.json").read_text())
    public_row = manifest["datasets"][DATASETS[0]][0]
    expected_hash = hashlib.sha256(json.dumps(reference).encode()).hexdigest()
    assert public_row["reference_sha256"] == expected_hash
    assert set(public_row) == {
        "id",
        "path",
        "sha256",
        "frames",
        "rate",
        "reference_sha256",
    }
    parented = manifest["datasets"][DATASETS[1]][0]
    assert (parented["parent_id"], parented["chunk_index"]) == ("session", 0)
    markdown = (output / "results.md").read_text()
    assert "Complete: 1." in markdown
    assert all(word not in markdown.lower() for word in ("failed", "rejected"))
    assert "0.2 (0.1-0.5)" in markdown
    assert "840.00 s" in markdown
    assert "4.10%, 4.20%, 4.30%" in markdown


@pytest.mark.parametrize(
    "raw", ("[]", "null", '"text"', '{"status": []}', '{"campaign_id":')
)
def test_main_omits_malformed_results_without_aborting(
    tmp_path: Path, export_images: Mock, run_report, campaign, run_dir: Path, raw: str
) -> None:
    (run_dir / "result.json").write_text(raw)
    run_report(campaign, [run_dir], tmp_path / "public")
    assert json.loads((tmp_path / "public/results.json").read_text())["runs"] == []


@pytest.mark.parametrize("field", ("hardware", "spec"))
def test_main_rejects_nonfinite_public_metadata_without_losing_valid_runs(
    tmp_path: Path,
    export_images: Mock,
    run_report,
    campaign: dict[str, JSONValue],
    run_dir: Path,
    field: str,
) -> None:
    other = tmp_path / "bad-metadata"
    result = write_run(campaign, other, batch=1)
    key = "cpu_quota" if field == "hardware" else "started_at"
    result[field][key] = float("nan")
    filename = "hardware.json" if field == "hardware" else "run.json"
    (other / filename).write_text(json.dumps(result[field]))
    (other / "result.json").write_text(json.dumps(result))

    output = tmp_path / "public"
    run_report(campaign, [run_dir, other], output)

    runs = json.loads((output / "results.json").read_text())["runs"]
    assert len(runs) == 1
    assert runs[0]["spec"]["batch_size"] == 2


@pytest.mark.parametrize("status", ("failed", "running"))
def test_noncomplete_runs_never_require_or_publish_measurements(
    campaign, run_dir, status
):
    value = json.loads((run_dir / "result.json").read_text())
    value.update(status=status, stage="build", diagnostics="private reference")
    write_json(run_dir / "result.json", value)
    (run_dir / "collection.json").unlink()
    result = report.validate_result(campaign, run_dir)
    assert result["status"] == status
    assert result["spec"] == value["spec"]
    assert not {"passes", "rtfx_median", "diagnostics"} & result.keys()


@pytest.mark.parametrize("different_hardware", (False, True))
def test_main_refuses_duplicate_points_or_incomparable_hardware(
    tmp_path: Path,
    export_images: Mock,
    run_report,
    campaign,
    run_dir: Path,
    different_hardware: bool,
) -> None:
    other = tmp_path / "other" / run_dir.name
    result = write_run(campaign, other, batch=1 if different_hardware else 2)
    if different_hardware:
        result["hardware"]["cpu_affinity"] = [2, 3]
        write_json(other / "result.json", result)
        write_json(other / "hardware.json", result["hardware"])
    output = tmp_path / "public"
    with pytest.raises(
        ValueError,
        match="Incompatible hardware"
        if different_hardware
        else "Duplicate configuration",
    ):
        run_report(campaign, [run_dir, other], output)
    export_images.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("changed_field", (None, "Model name:", "CPU max MHz:"))
def test_main_compares_stable_cpu_metadata_and_preserves_raw_snapshots(
    tmp_path, export_images, run_report, campaign, nested, changed_field
):
    directories, hardware = [], {}
    for batch in (1, 2):
        directory = tmp_path / "runs" / f"b{batch}"
        result = write_run(campaign, directory, batch=batch)
        fields = {
            "Model name:": "Example CPU",
            "CPU max MHz:": "4000.0000",
            "CPU MHz:": str(2000 + batch * 100),
            "CPU(s) scaling MHz:": f"{50 + batch}%",
        }
        if batch == 2 and changed_field:
            fields[changed_field] = "different"
        rows = [{"field": key, "data": value} for key, value in fields.items()]
        if nested:
            rows = [{"field": "CPU(s):", "data": "32", "children": rows}]
        result["hardware"]["lscpu"] = json.dumps({"lscpu": rows}, indent=batch)
        hardware[batch] = result["hardware"]
        write_json(directory / "hardware.json", result["hardware"])
        write_json(directory / "result.json", result)
        directories.append(directory)
    before = {p: file_hash(p) for p in (tmp_path / "runs").rglob("*.json")}
    output = tmp_path / "public"

    if changed_field:
        with pytest.raises(ValueError, match="Incompatible hardware"):
            run_report(campaign, directories, output)
        export_images.assert_not_called()
        assert not output.exists()
    else:
        run_report(campaign, directories, output)
        runs = json.loads((output / "results.json").read_text())["runs"]
        assert {r["spec"]["batch_size"]: r["hardware"] for r in runs} == hardware
    assert {p: file_hash(p) for p in before} == before


@pytest.mark.parametrize(
    "raw",
    (
        "not-json",
        '{"lscpu": null}',
        '{"lscpu": [null]}',
        '{"lscpu": {}}',
        '{"lscpu": [{"field": "CPU(s):", "data": "32", "children": {}}]}',
    ),
)
def test_main_rejects_malformed_cpu_metadata_without_losing_valid_runs(
    tmp_path, export_images, run_report, campaign, run_dir, raw
):
    other = tmp_path / "bad-cpu"
    result = write_run(campaign, other, batch=1)
    result["hardware"]["lscpu"] = raw
    write_json(other / "hardware.json", result["hardware"])
    write_json(other / "result.json", result)
    output = tmp_path / "public"

    run_report(campaign, [run_dir, other], output)

    runs = json.loads((output / "results.json").read_text())["runs"]
    assert [r["spec"]["batch_size"] for r in runs] == [2]


@pytest.mark.parametrize("speed", (1, 2))
def test_main_deduplicates_aliases_and_highlights_speed_not_accuracy(
    tmp_path: Path, export_images: Mock, run_report, campaign, run_dir: Path, speed: int
) -> None:
    fp32 = tmp_path / "runs/fp32"
    result = write_run(campaign, fp32, precision="fp32", speed=speed)
    for i, saved in enumerate(result["passes"], 1):
        for scores in saved["datasets"].values():
            scores["wer"] = 50
        write_json(fp32 / f"pass-{i}/scores.json", saved)
    write_json(fp32 / "result.json", result)
    alias = tmp_path / "alias"
    alias.symlink_to(run_dir, target_is_directory=True)
    readme = tmp_path / "README.md"
    readme.write_text(f"Before\n{report.START}\nold\n{report.END}\nAfter\n")
    output = tmp_path / "public"

    run_report(campaign, [run_dir, alias, fp32], output, readme)

    assert len(json.loads((output / "results.json").read_text())["runs"]) == 2
    text = readme.read_text()
    assert "| fp32 |" in text
    assert "| fp16 |" not in text
    assert "50.00%" in text
    assert "not proof that larger batches have saturated" in text


@pytest.mark.parametrize(
    "text",
    (
        "no markers",
        report.START,
        report.END + report.START,
        report.START + report.START + report.END,
        report.START + report.END + report.END,
    ),
)
def test_bad_readme_markers_fail_before_writing(
    tmp_path: Path, export_images: Mock, run_report, campaign, text: str
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(text)
    with pytest.raises(ValueError, match="marker pair"):
        run_report(campaign, [], tmp_path / "public", readme)
    assert readme.read_text() == text
    assert not (tmp_path / "public").exists()
    export_images.assert_not_called()


@pytest.mark.parametrize(
    "failure,error,message",
    (
        ("browser", RuntimeError, "browser unavailable"),
        ("concurrent-readme", ValueError, "README changed"),
        ("staging", OSError, "disk full"),
    ),
)
def test_generation_failures_preserve_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_images: Mock,
    run_report,
    campaign,
    failure: str,
    error: type[Exception],
    message: str,
) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "results.json").write_text("old results")
    (output / "zipformer_rnnt-fp32.svg").write_text("old figure")
    readme = tmp_path / "README.md"
    text = f"{report.START}\nold\n{report.END}"
    readme.write_text(text)
    if failure == "staging":
        monkeypatch.setattr(Path, "write_bytes", Mock(side_effect=OSError("disk full")))

    def interrupt_export(
        file: list[Path], **kwargs: list[go.Figure] | str | int
    ) -> None:
        """Write one staged image, then fail or simulate a concurrent README edit.

        Parameters
        ----------
        file : list[Path]
            Staging paths; only the first image is written.
        **kwargs : list[go.Figure] | str | int
            Unused figures, format, and dimensions from Plotly's export call.

        Raises
        ------
        RuntimeError
            The selected failure simulates a browser error.
        """

        file[0].write_text("new figure")
        if failure == "browser":
            raise RuntimeError("browser unavailable")
        if failure == "concurrent-readme":
            readme.write_text("user edit")

    export_images.side_effect = interrupt_export
    with pytest.raises(error, match=message):
        run_report(campaign, [], output, readme)
    assert (output / "results.json").read_text() == "old results"
    assert (output / "zipformer_rnnt-fp32.svg").read_text() == "old figure"
    assert readme.read_text() == (
        "user edit" if failure == "concurrent-readme" else text
    )
    assert not list(tmp_path.glob(".benchmark-report-*"))


@pytest.mark.parametrize("filename", ("results.md", "parakeet_v3-fp16.svg"))
def test_main_rejects_directory_targets_before_replacing_any_files(
    tmp_path: Path,
    export_images: Mock,
    run_report,
    campaign: dict[str, JSONValue],
    filename: str,
) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "results.json").write_text("old results")
    (output / filename).mkdir()

    with pytest.raises(ValueError, match="Generated report path"):
        run_report(campaign, [], output)

    assert (output / "results.json").read_text() == "old results"
    assert (output / filename).is_dir()
    export_images.assert_not_called()


def test_readme_edits_during_publication_are_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_images: Mock,
    run_report,
    campaign: dict[str, JSONValue],
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"{report.START}\nold\n{report.END}")
    replace = Path.replace

    def edit_readme(path: Path, target: Path) -> Path:
        """Simulate a user edit while the generated report files are promoted."""
        promoted = replace(path, target)
        if path.name == "results.json":
            readme.write_text("user edit during publication")
        return promoted

    monkeypatch.setattr(Path, "replace", edit_readme)
    with pytest.raises(ValueError, match="README changed"):
        run_report(campaign, [], tmp_path / "public", readme)

    assert readme.read_text() == "user edit during publication"


@pytest.mark.parametrize("location", ("run", "ancestor", "child"))
def test_main_rejects_output_overlapping_evidence(
    campaign, run_dir, run_report, location
):
    output = {"run": run_dir, "ancestor": run_dir.parent, "child": run_dir / "report"}[
        location
    ]
    before = file_hash(run_dir / "result.json")
    with pytest.raises(ValueError, match="overlap"):
        run_report(campaign, [run_dir], output)
    assert file_hash(run_dir / "result.json") == before


def test_main_discovers_unstarted_and_interrupted_runs_without_duplicates(
    tmp_path: Path,
    run_dir: Path,
    campaign: dict[str, JSONValue],
    monkeypatch: pytest.MonkeyPatch,
    export_images: Mock,
):
    root = run_dir.parent
    write_json(root / "matrix.json", [run_dir.name, "unstarted", "interrupted"])
    interrupted = root / "interrupted"
    interrupted.mkdir()
    write_json(interrupted / "run.json", {})
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    path = tmp_path / "campaign.json"
    write_json(path, campaign)
    output = tmp_path / "public"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--campaign",
            str(path),
            "--runs",
            str(root),
            str(alias),
            str(run_dir),
            "--output",
            str(output),
        ],
    )
    render = Mock(wraps=report.render_report)
    monkeypatch.setattr(report, "render_report", render)

    report.main()

    render.assert_called_once()
    assert [(r["run"], r["status"]) for r in render.call_args.args[1]] == [
        (run_dir.name, "complete"),
        ("interrupted", "incomplete"),
        ("unstarted", "not_run"),
    ]


@pytest.mark.parametrize(
    "matrix",
    (
        [],
        {},
        "name",
        [1],
        [""],
        ["."],
        [".."],
        ["../outside"],
        ["/tmp/outside"],
        ["nested/run"],
        ["nested\\run"],
        ["run", "run"],
    ),
)
def test_main_rejects_invalid_matrix_entries(
    tmp_path: Path,
    campaign: dict[str, JSONValue],
    monkeypatch: pytest.MonkeyPatch,
    matrix: JSONValue,
):
    write_json(tmp_path / "matrix.json", matrix)
    path = tmp_path / "campaign.json"
    write_json(path, campaign)
    output = tmp_path / "public"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--campaign",
            str(path),
            "--runs",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Matrix|Duplicate matrix"):
        report.main()
    assert not output.exists()


@pytest.mark.parametrize("kind", ("missing", "empty", "file", "escaping-symlink"))
def test_main_rejects_bad_roots_and_escaping_symlinks(
    tmp_path: Path,
    campaign: dict[str, JSONValue],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    root = tmp_path / "runs"
    if kind == "file":
        root.write_text("not a directory")
    elif kind != "missing":
        root.mkdir()
    if kind == "escaping-symlink":
        (root / "outside").symlink_to(tmp_path, target_is_directory=True)
        write_json(root / "matrix.json", ["outside"])
    path = tmp_path / "campaign.json"
    write_json(path, campaign)
    output = tmp_path / "public"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--campaign",
            str(path),
            "--runs",
            str(root),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Run root|No run records|Run path escapes"):
        report.main()
    assert not output.exists()


def test_render_findings_compare_matched_batches_and_all_precisions(
    tmp_path: Path, export_images: Mock, campaign
) -> None:
    single, best, fp32, fp16 = [
        {
            "status": "complete",
            "spec": {
                "model": "zipformer_cr_ctc_rnnt",
                "gpu": "H100",
                "batch_size": batch,
                "precision": precision,
            },
            "rtfx_median": rtfx,
            "rtfx_range": [rtfx, rtfx],
            "inference_seconds_median": 1.0,
            "passes": [{"mean_wer_percent": wer + i} for i in range(3)],
        }
        for batch, precision, rtfx, wer in (
            (1, "bf16", 10, 8),
            (64, "bf16", 40, 9),
            (64, "fp32", 20, 7),
            (64, "fp16", 30, 8),
        )
    ]
    body = report.render_report(campaign, [fp32, fp16, best, single], tmp_path, ".")
    lines = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(lines) == 3
    assert "40.0 RTFx" in lines[0]
    assert "4.00x" in lines[1]
    assert "7.00-11.00%" in lines[2]
    for results in ([best, fp32], [single]):
        body = report.render_report(campaign, results, tmp_path, ".")
        assert len([line for line in body.splitlines() if line.startswith("- ")]) == 1

    for key, value in (
        ("gpu", "A100"),
        ("model", "parakeet_v3"),
        ("precision", "fp32"),
    ):
        baseline = copy.deepcopy(single)
        baseline["spec"][key] = value
        body = report.render_report(campaign, [best, baseline], tmp_path, ".")
        assert "- Scaling:" not in body, key

    for key, value in (
        ("gpu", "A100"),
        ("model", "parakeet_v3"),
        ("batch_size", 32),
    ):
        unmatched = copy.deepcopy(fp16)
        unmatched["spec"][key] = value
        body = report.render_report(campaign, [best, fp32, unmatched], tmp_path, ".")
        assert "- Precision:" not in body, key


def test_module_entry_point_loads_campaign_and_reports_real_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_images: Mock,
    campaign,
    run_dir: Path,
) -> None:
    path = tmp_path / "campaign.json"
    write_json(path, campaign)
    output = tmp_path / "public"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--campaign",
            str(path),
            "--runs",
            str(run_dir.parent),
            "--output",
            str(output),
        ],
    )

    # Let runpy execute a fresh module without its already-imported warning.
    monkeypatch.delitem(sys.modules, report.__name__)
    runpy.run_module(report.__name__, run_name="__main__")

    result = json.loads((output / "results.json").read_text())
    assert result["campaign_id"] == campaign["id"]
    assert result["runs"][0]["status"] == "complete"


@pytest.mark.parametrize("hashes", ({}, ["hash"], {"encoder.trt": "invalid"}))
def test_validate_rejects_malformed_matching_hash_inventories(
    campaign, run_dir, hashes
):
    result = json.loads((run_dir / "result.json").read_text())
    result["bundle_files"] = hashes
    write_json(run_dir / "result.json", result)
    write_json(run_dir / "bundle-hashes.json", hashes)
    with pytest.raises(ValueError, match="hash inventory"):
        report.validate_result(campaign, run_dir)


def test_validate_rejects_unknown_failure_stage(campaign, run_dir):
    result = json.loads((run_dir / "result.json").read_text())
    result.update(status="failed", stage="unknown")
    write_json(run_dir / "result.json", result)
    with pytest.raises(ValueError, match="Unknown failing stage"):
        report.validate_result(campaign, run_dir)


def test_main_refuses_file_output_and_readme_collision(tmp_path, campaign, run_report):
    output = tmp_path / "public"
    output.write_text("existing file")
    with pytest.raises(ValueError, match="must be a directory"):
        run_report(campaign, [], output)
    assert output.read_text() == "existing file"
    readme = tmp_path / "results.md"
    text = f"{report.START}\n{report.END}"
    readme.write_text(text)
    with pytest.raises(ValueError, match="README conflicts"):
        run_report(campaign, [], tmp_path, readme)
    assert readme.read_text() == text


@pytest.mark.parametrize(
    ("collision", "message"),
    (
        ("output", "Private campaign must be outside"),
        ("readme", "README must not overwrite the private campaign"),
    ),
)
def test_main_preserves_private_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign,
    collision: str,
    message: str,
) -> None:
    path = tmp_path / "campaign.json"
    write_json(path, campaign)
    argv = [
        "report",
        "--campaign",
        str(path),
        "--runs",
        str(tmp_path),
        "--output",
        str(tmp_path if collision == "output" else tmp_path / "public"),
    ]
    if collision == "readme":
        argv.extend(("--readme", str(path)))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=message):
        report.main()
    assert json.loads(path.read_text()) == campaign


def test_main_sorts_tables_by_configuration_not_attempt_path(
    tmp_path: Path, export_images: Mock, run_report, campaign
) -> None:
    directories = [tmp_path / name for name in ("z-b1", "a-b64", "m-b2")]
    for directory, batch in zip(directories, (1, 64, 2), strict=True):
        write_run(campaign, directory, batch=batch)
    readme = tmp_path / "README.md"
    readme.write_text(f"{report.START}\n{report.END}")
    output = tmp_path / "public"

    run_report(campaign, directories, output, readme)

    runs = json.loads((output / "results.json").read_text())["runs"]
    assert [r["spec"]["batch_size"] for r in runs] == [1, 2, 64]
    rows = [
        line
        for line in (output / "results.md").read_text().splitlines()
        if line.startswith("| zipformer_cr_ctc_rnnt |")
    ]
    assert [int(line.split("|")[3]) for line in rows] == [1, 2, 64]


def test_main_refuses_readme_inside_run_evidence(
    tmp_path, campaign, run_dir, run_report
):
    path = run_dir / "result.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="README must not overwrite run evidence"):
        run_report(campaign, [run_dir], tmp_path / "public", path)
    assert path.read_bytes() == before
