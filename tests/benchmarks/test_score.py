#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU tests for scoring contracts, evidence integrity, and summary publication."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks import score
from benchmarks.common import (
    BEAMS,
    DATASETS,
    PROTOCOL,
    file_hash,
    write_json,
)


def write_lines(
    path: Path, rows: list[dict[str, str | int | float | None | list[str]]]
) -> None:
    """Write collector-style JSONL without changing reference or hypothesis text.

    Parameters
    ----------
    path : Path
        Destination in an existing directory.
    rows : list[dict[str, str | int | float | None | list[str]]]
        Records serialized with raw Unicode and empty strings preserved.
    """

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def scoring_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> SimpleNamespace:
    """Build a complete collection while replacing only the upstream scorer.

    Parameters
    ----------
    tmp_path : Path
        Root for the frozen campaign, metadata, and configured passes of evidence.
    monkeypatch : pytest.MonkeyPatch
        Supplies CLI arguments and a scorer that records its input manifests.
    request : pytest.FixtureRequest
        Optional measured pass count; defaults to three.

    Returns
    -------
    SimpleNamespace
        Paths, campaign, mutable upstream scores, captured manifests, and mocks.
        Tied durations, partial batches, and pass-specific predictions exercise
        ordering and pooled timing without audio, engines, GPU, or networking.
    """

    campaign = {
        "protocol": {**PROTOCOL, "passes": getattr(request, "param", 3)},
        "source": {"commit": "frozen collection source"},
        "models": {},
        "datasets": {},
    }
    for dataset_index, name in enumerate(DATASETS, 1):
        rows = [
            {
                "id": "a",
                "frames": 64000 * dataset_index,
                "rate": 16000,
                "text": f"Hello, WORLD! caf\u00e9 {dataset_index} \t",
            },
            {"id": "c", "frames": 16000, "rate": 16000, "text": "next\nline"},
            {"id": "b", "frames": 8000, "rate": 8000, "text": ""},
        ]
        if name == "earnings22_cleaned_aa_chunked_test":
            for row, parent, index in zip(
                rows, ("call", "other", "call"), (0, 0, 1), strict=True
            ):
                row.update(parent_id=parent, chunk_index=index, text=parent)

        campaign["datasets"][name] = sorted(rows, key=lambda row: row["id"])

    campaign["id"] = hashlib.sha256(
        json.dumps(campaign, sort_keys=True).encode()
    ).hexdigest()
    campaign_path = tmp_path / "campaign.json"
    write_json(campaign_path, campaign)
    directory = tmp_path / "run"
    directory.mkdir()
    spec = {
        "campaign_id": campaign["id"],
        "model": "zipformer_cr_ctc_rnnt",
        "gpu": "H100",
        "gpu_uuid": "GPU-test",
        "batch_size": 2,
        "precision": "fp16",
        "beam": 6,
    }

    write_json(directory / "run.json", spec)
    write_json(directory / "hardware.json", {"gpu": {"uuid": "GPU-test"}})
    write_json(directory / "bundle-hashes.json", {"encoder.trt": "a" * 64})
    write_json(directory / "plugin-hashes.json", {"feature.so": "b" * 64})

    evidence = {}
    for index, factor in enumerate((1, 4, 2)[: campaign["protocol"]["passes"]], 1):
        folder = directory / f"pass-{index}"
        folder.mkdir()
        records = []
        for dataset_index, name in enumerate(DATASETS, 1):
            predictions = [
                {"id": identifier, "pred_text": text, "word_timestamps": []}
                for identifier, text in (
                    ("b", f"  {name} pass {index}!  "),
                    ("c", ""),
                    ("a", "Hello world"),
                )
            ]
            write_lines(folder / f"{name}.jsonl", predictions)
            for batch_index, (ids, seconds) in enumerate(
                ((["b", "c"], 2), (["a"], 4 * dataset_index)), 1
            ):
                records.append(
                    {
                        "dataset": name,
                        "ids": ids,
                        "audio_seconds": seconds,
                        "inference_seconds": batch_index * dataset_index * factor,
                    }
                )
        write_lines(folder / "batches.jsonl", records)
        write_json(folder / "totals.json", {"rtfx": "must be recomputed"})
        evidence.update(
            {
                f"pass-{index}/{name}.jsonl": file_hash(folder / f"{name}.jsonl")
                for name in (*DATASETS, "batches")
            }
        )
    write_json(
        directory / "collection.json",
        {
            "campaign_id": campaign["id"],
            "source": campaign["source"],
            "evidence": evidence,
        },
    )
    manifests = []
    scores = {
        f"benchmark/model | {name}": {
            "wer": 7.0 * index,
            "ins": index,
            "del": 1,
            "sub": 2,
            "audio_length": None,
            "inference_time": None,
            "rtfx": None,
        }
        for index, name in enumerate(DATASETS, 1)
    }

    def evaluate(
        temporary: str, language: str
    ) -> tuple[dict[str, float], dict[str, dict[str, int | float | str | None]]]:
        """Record real manifest files and supply distinct scores on each call.

        Parameters
        ----------
        temporary : str
            Temporary directory passed to the upstream scorer.
        language : str
            Must be English for the frozen benchmark suite.

        Returns
        -------
        tuple[dict[str, float], dict[str, dict[str, int | float | str | None]]]
            Unused composite output and dataset metrics; successive WER values
            differ to detect reused pass results and partial summary replacement.
        """

        assert language == "en"
        assert {path.name for path in Path(temporary).iterdir()} == {
            f"MODEL_benchmark-model_DATASET_{name}.jsonl" for name in DATASETS
        }
        manifest = {}
        for name in DATASETS:
            path = Path(temporary) / f"MODEL_benchmark-model_DATASET_{name}.jsonl"
            with path.open(encoding="utf-8") as stream:
                manifest[name] = [json.loads(line) for line in stream]
        manifests.append(manifest)
        sys.stdout.write("scorer output\n")
        sys.stderr.write("scorer diagnostic\n")
        return {}, {
            name: {**metrics, "wer": metrics["wer"] * len(manifests)}
            for name, metrics in scores.items()
        }

    implementation = Mock(side_effect=evaluate)
    loader = Mock(return_value=SimpleNamespace(score_results=implementation))
    monkeypatch.setattr(score, "upstream", loader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--campaign",
            str(campaign_path),
            "--run-dir",
            str(directory),
            "--scorer",
            str(tmp_path / "scorer"),
        ],
    )
    return SimpleNamespace(
        campaign=campaign,
        campaign_path=campaign_path,
        directory=directory,
        scores=scores,
        implementation=implementation,
        loader=loader,
        manifests=manifests,
    )


@pytest.mark.parametrize(("model", "beam"), BEAMS.items())
@pytest.mark.parametrize("scoring_run", (1, 3), indirect=True)
def test_main_scores_all_passes_and_preserves_evidence(
    scoring_run: SimpleNamespace, model: str, beam: int
) -> None:
    run = scoring_run
    spec = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    spec.update(model=model, beam=beam)
    write_json(run.directory / "run.json", spec)
    original = {
        path: path.read_bytes()
        for path in (run.campaign_path, *run.directory.rglob("*"))
        if path.is_file()
    }

    score.main()

    result = json.loads((run.directory / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["campaign_id"] == run.campaign["id"]
    passes = run.campaign["protocol"]["passes"]
    assert len(result["passes"]) == passes
    assert {path: path.read_bytes() for path in original} == original
    run.loader.assert_called_once_with(run.directory.parent / "scorer")
    assert run.implementation.call_count == passes
    for index, saved in enumerate(result["passes"], 1):
        factor = (1, 4, 2)[index - 1]
        assert saved == json.loads(
            (run.directory / f"pass-{index}/scores.json").read_text(encoding="utf-8")
        )
        assert saved["audio_seconds"] == 126
        assert saved["inference_seconds"] == 84 * factor
        assert saved["rtfx"] == 1.5 / factor
        assert saved["mean_wer_percent"] == 28 * index
        assert tuple(saved["datasets"]) == DATASETS
        for dataset_index, name in enumerate(DATASETS, 1):
            metrics = saved["datasets"][name]
            assert metrics == pytest.approx(
                {
                    "wer": 7.0 * dataset_index * index,
                    "ins": dataset_index,
                    "del": 1,
                    "sub": 2,
                    "audio_length": 2 + 4 * dataset_index,
                    "samples": 3,
                    "inference_time": 3 * dataset_index * factor,
                    "rtfx": (2 + 4 * dataset_index) / (3 * dataset_index * factor),
                }
            )
            rows = run.manifests[index - 1][name]
            source = {row["id"]: row for row in run.campaign["datasets"][name]}
            assert rows == [
                {
                    **source[identifier],
                    "pred_text": text,
                    "duration": seconds,
                    "time": None,
                }
                for identifier, text, seconds in (
                    ("b", f"  {name} pass {index}!  ", 1),
                    ("c", "", 1),
                    ("a", "Hello world", 4 * dataset_index),
                )
            ]
        assert (run.directory / f"pass-{index}/scorer.log").read_text() == (
            "scorer output\nscorer diagnostic\n"
        )
    assert result["rtfx_median"] == (1.5 if passes == 1 else 0.75)
    assert result["rtfx_range"] == ([1.5, 1.5] if passes == 1 else [0.375, 1.5])
    assert result["inference_seconds_median"] == (84 if passes == 1 else 168)
    for key, filename in (
        ("spec", "run"),
        ("hardware", "hardware"),
        ("bundle_files", "bundle-hashes"),
        ("plugins", "plugin-hashes"),
    ):
        assert result[key] == json.loads(
            (run.directory / f"{filename}.json").read_text(encoding="utf-8")
        )
    assert json.loads(
        (run.directory / "collection.json").read_text(encoding="utf-8")
    )["evidence"] == result["evidence"]
    assert not list(run.directory.rglob("*.tmp"))
    assert all(
        not Path(call.args[0]).exists() for call in run.implementation.call_args_list
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("missing-marker", "Collection did not finish"),
        ("campaign", "source snapshot differs"),
        ("source", "source snapshot differs"),
        ("empty-hashes", "evidence hashes"),
        ("missing-hash", "evidence hashes"),
        ("extra-hash", "evidence hashes"),
        ("changed-file", "Collected evidence changed"),
        ("changed-timing", "Collected evidence changed"),
        ("missing-file", DATASETS[0]),
    ),
)
def test_invalid_collection_is_rejected_before_loading_scorer(
    scoring_run: SimpleNamespace, change: str, message: str
) -> None:
    run = scoring_run
    path = run.directory / "collection.json"
    marker = json.loads(path.read_text(encoding="utf-8"))
    name = (
        "pass-1/batches.jsonl"
        if change == "changed-timing"
        else f"pass-1/{DATASETS[0]}.jsonl"
    )
    if change == "missing-marker":
        path.unlink()
    elif change in ("changed-file", "changed-timing", "missing-file"):
        if change != "missing-file":
            (run.directory / name).write_text("changed")
        else:
            (run.directory / name).unlink()
    else:
        if change == "campaign":
            marker["campaign_id"] = "other"
        elif change == "source":
            marker["source"] = {}
        elif change == "empty-hashes":
            marker["evidence"] = {}
        elif change == "missing-hash":
            del marker["evidence"][name]
        else:
            marker["evidence"]["../outside.jsonl"] = "unexpected"
        write_json(path, marker)

    with pytest.raises(
        FileNotFoundError if change == "missing-file" else ValueError,
        match=message,
    ):
        score.main()
    run.loader.assert_not_called()
    assert not list(run.directory.rglob("scores.json"))
    assert not (run.directory / "result.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("campaign_id", "other", "different campaign"),
        ("model", "unknown", "benchmark matrix"),
        ("gpu", "unknown", "benchmark matrix"),
        ("precision", "int8", "benchmark matrix"),
        ("batch_size", 3, "benchmark matrix"),
        ("batch_size", 2.0, "benchmark matrix"),
        ("beam", 1, "benchmark matrix"),
        ("beam", 6.0, "benchmark matrix"),
        ("gpu_uuid", "other", "Recorded GPU differs"),
    ),
)
def test_invalid_run_configuration_is_rejected(
    scoring_run: SimpleNamespace, field: str, value: str | int | float, message: str
) -> None:
    path = scoring_run.directory / "run.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec[field] = value
    write_json(path, spec)
    with pytest.raises(ValueError, match=message):
        score.main()
    scoring_run.loader.assert_not_called()
    assert not (scoring_run.directory / "result.json").exists()


@pytest.mark.parametrize("name", ("bundle-hashes", "plugin-hashes"))
def test_empty_artifact_inventory_is_rejected(
    scoring_run: SimpleNamespace, name: str
) -> None:
    write_json(scoring_run.directory / f"{name}.json", {})
    with pytest.raises(ValueError, match="Missing bundle or plugin hashes"):
        score.main()
    scoring_run.loader.assert_not_called()


@pytest.mark.parametrize(
    ("name", "change", "message"),
    (
        (DATASETS[0], "missing", "Incomplete or reordered transcripts"),
        (DATASETS[0], "duplicate", "Incomplete or reordered transcripts"),
        (DATASETS[0], "reordered", "Incomplete or reordered transcripts"),
        (DATASETS[0], "nontext", "Hypotheses must be strings"),
        ("batches", "batch-order", "order, or utterance mismatch"),
        ("batches", "missing", "batch count differs"),
        ("batches", "wrong-dataset", "order, or utterance mismatch"),
        ("batches", "padded-duration", "Audio duration includes padding"),
        ("batches", "invalid-time", "Invalid inference time"),
        ("batches", "nonfinite-time", "Invalid inference time"),
        ("batches", "string-time", "Batch timings must be numbers"),
    ),
)
def test_invalid_hashed_records_are_not_scored(
    scoring_run: SimpleNamespace, name: str, change: str, message: str
) -> None:
    run = scoring_run
    path = run.directory / f"pass-1/{name}.jsonl"
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[1] = records[0]
    elif change == "reordered":
        records.reverse()
    elif change == "nontext":
        records[0]["pred_text"] = None
    elif change == "batch-order":
        records[0]["ids"].reverse()
    elif change == "wrong-dataset":
        records[0]["dataset"] = DATASETS[1]
    elif change == "padded-duration":
        records[0]["audio_seconds"] += 1
    elif change == "invalid-time":
        records[0]["inference_seconds"] = 0
    elif change == "nonfinite-time":
        records[0]["inference_seconds"] = float("inf")
    elif change == "string-time":
        records[0]["inference_seconds"] = "1"
    write_lines(path, records)
    marker = json.loads((run.directory / "collection.json").read_text(encoding="utf-8"))
    marker["evidence"][path.relative_to(run.directory).as_posix()] = file_hash(path)
    write_json(run.directory / "collection.json", marker)

    with pytest.raises(ValueError, match=message):
        score.main()
    run.implementation.assert_not_called()
    assert not list(run.directory.rglob("scores.json"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("wer", -1),
        ("wer", float("nan")),
        ("wer", float("inf")),
        ("wer", "0"),
        ("ins", -1),
        ("del", 0.5),
        ("sub", -1),
    ),
)
def test_invalid_upstream_metrics_are_rejected(
    scoring_run: SimpleNamespace, field: str, value: int | float | str
) -> None:
    run = scoring_run
    run.scores[f"benchmark/model | {DATASETS[0]}"][field] = value
    with pytest.raises(ValueError, match="Invalid upstream WER or error counts"):
        score.main()
    assert not list(run.directory.rglob("scores.json"))


def test_main_accepts_zero_wer_and_error_counts(scoring_run: SimpleNamespace) -> None:
    for metrics in scoring_run.scores.values():
        metrics.update({"wer": 0, "ins": 0, "del": 0, "sub": 0})

    score.main()

    result = json.loads(
        (scoring_run.directory / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "complete"
    assert len(result["passes"]) == 3
    for saved in result["passes"]:
        assert saved["mean_wer_percent"] == 0
        assert set(saved["datasets"]) == set(DATASETS)
        for metrics in saved["datasets"].values():
            assert [metrics[key] for key in ("wer", "ins", "del", "sub")] == [0] * 4


@pytest.mark.parametrize("change", ("missing", "extra", "wrong-model"))
def test_upstream_must_return_exactly_one_result_per_dataset(
    scoring_run: SimpleNamespace, change: str
) -> None:
    run = scoring_run
    key = f"benchmark/model | {DATASETS[0]}"
    if change != "missing":
        run.scores[f"other/model | {DATASETS[0]}"] = run.scores[key]
    if change != "extra":
        del run.scores[key]
    with pytest.raises(ValueError, match="exactly the seven datasets"):
        score.main()
    assert not list(run.directory.rglob("scores.json"))


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("evidence", "Collected evidence changed"),
        ("marker", "Collection marker changed"),
        ("run", "Run metadata changed"),
        ("hardware", "Run metadata changed"),
        ("bundle-hashes", "Run metadata changed"),
        ("plugin-hashes", "Run metadata changed"),
        ("scorer", "upstream failed"),
    ),
)
def test_failed_rescore_preserves_previous_summaries(
    scoring_run: SimpleNamespace, change: str, message: str
) -> None:
    run = scoring_run
    score.main()
    saved = {path: path.read_bytes() for path in run.directory.rglob("*.json")}
    if change not in ("evidence", "scorer"):
        filename = "collection.json" if change == "marker" else f"{change}.json"
        del saved[run.directory / filename]
    evaluate = run.implementation.side_effect

    def disrupt_rescore(
        temporary: str, language: str
    ) -> tuple[dict[str, float], dict[str, dict[str, int | float | str | None]]]:
        """Fail pass two or alter the snapshot after the final scorer call.

        Parameters
        ----------
        temporary : str
            Manifest directory forwarded to the recording scorer.
        language : str
            Scoring language, forwarded unchanged.

        Returns
        -------
        tuple[dict[str, float], dict[str, dict[str, int | float | str | None]]]
            Original scorer output when only the snapshot is changed.

        Raises
        ------
        RuntimeError
            Scorer failure is selected and pass two has logged its diagnostics.
        """

        result = evaluate(temporary, language)
        if change == "scorer" and len(run.manifests) == 5:
            raise RuntimeError("upstream failed on the second pass")
        if len(run.manifests) == 6:
            if change == "evidence":
                (run.directory / f"pass-1/{DATASETS[0]}.jsonl").write_text("changed")
            else:
                path = run.directory / (
                    "collection.json" if change == "marker" else f"{change}.json"
                )
                value = json.loads(path.read_text(encoding="utf-8"))
                value["changed"] = True
                write_json(path, value)
        return result

    run.implementation.side_effect = disrupt_rescore
    with pytest.raises(
        RuntimeError if change == "scorer" else ValueError, match=message
    ):
        score.main()
    assert {path: path.read_bytes() for path in saved} == saved
    assert run.implementation.call_count == (5 if change == "scorer" else 6)
    assert all(
        not Path(call.args[0]).exists() for call in run.implementation.call_args_list
    )
    assert (run.directory / "pass-2/scorer.log").read_text() == (
        "scorer output\nscorer diagnostic\n"
    )


@pytest.mark.parametrize("filename", ("pass-2/scores.json", "result.json"))
def test_summary_write_failure_preserves_previous_final_result(
    scoring_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    run = scoring_run
    score.main()
    paths = [
        run.directory / name
        for name in ("result.json", "pass-1/scores.json", "pass-2/scores.json")
    ]
    previous = [path.read_bytes() for path in paths]
    target = run.directory / filename
    replace = Path.replace

    def fail_replace(path: Path, destination: Path) -> Path:
        """Fail a real temporary-file replacement without altering its destination.

        Parameters
        ----------
        path : Path
            Fully written temporary JSON file.
        destination : Path
            Existing summary that the writer is about to replace.

        Returns
        -------
        Path
            Result of the real replacement for other destinations.

        Raises
        ------
        PermissionError
            The selected summary is being replaced.
        """

        if destination == target:
            raise PermissionError("summary write failed")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="summary write failed"):
        score.main()
    assert paths[0].read_bytes() == previous[0]
    assert paths[1].read_bytes() != previous[1]
    assert (paths[2].read_bytes() == previous[2]) == (filename == "pass-2/scores.json")
    assert json.loads(
        target.with_suffix(".json.tmp").read_text(encoding="utf-8")
    ) != json.loads(target.read_text(encoding="utf-8"))


def test_cli_help_does_not_import_cuda() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.modules['cupy'] = None\n"
            "sys.modules['fast_gpu_asr'] = None\n"
            "sys.argv = ['score', '--help']\n"
            "runpy.run_module('benchmarks.score', run_name='__main__')\n",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert all(
        option in result.stdout for option in ("--campaign", "--run-dir", "--scorer")
    )


@pytest.mark.parametrize("valid_digest", (False, True))
def test_old_scorer_revision_does_not_bypass_campaign_validation(
    scoring_run: SimpleNamespace, valid_digest: bool
) -> None:
    run = scoring_run
    campaign = json.loads(run.campaign_path.read_text(encoding="utf-8"))
    campaign["scorer_revision"] = "old-scorer-commit"
    if valid_digest:
        campaign["id"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in campaign.items() if key != "id"},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for filename in ("collection.json", "run.json"):
            metadata = json.loads(
                (run.directory / filename).read_text(encoding="utf-8")
            )
            metadata["campaign_id"] = campaign["id"]
            write_json(run.directory / filename, metadata)
    write_json(run.campaign_path, campaign)
    if valid_digest:
        score.main()
        run.loader.assert_called_once_with(run.directory.parent / "scorer")
        assert json.loads(
            (run.directory / "result.json").read_text(encoding="utf-8")
        )["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            score.main()
        run.loader.assert_not_called()
        assert not (run.directory / "result.json").exists()


def test_main_propagates_scorer_loading_failure(scoring_run: SimpleNamespace) -> None:
    scoring_run.loader.side_effect = ValueError("wrong scorer checkout")
    with pytest.raises(ValueError, match="wrong scorer checkout"):
        score.main()
    scoring_run.implementation.assert_not_called()
    assert not list(scoring_run.directory.rglob("scores.json"))
    assert not (scoring_run.directory / "result.json").exists()


@pytest.mark.parametrize("name", (DATASETS[0], "batches"))
def test_scoring_rejects_malformed_jsonl(
    scoring_run: SimpleNamespace, name: str
) -> None:
    run = scoring_run
    path = run.directory / f"pass-1/{name}.jsonl"
    path.write_text('{"id": "truncated"\n')
    marker = json.loads((run.directory / "collection.json").read_text(encoding="utf-8"))
    marker["evidence"][f"pass-1/{name}.jsonl"] = file_hash(path)
    write_json(run.directory / "collection.json", marker)

    with pytest.raises(json.JSONDecodeError):
        score.main()
    run.implementation.assert_not_called()
    assert not list(run.directory.rglob("scores.json"))
