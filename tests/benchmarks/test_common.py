#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Benchmark provenance, persistence, audio conversion, and timing validation."""

import hashlib
import importlib
import json
import struct
import subprocess
import sys
import wave
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks import common
from benchmarks.common import JSONValue


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated Git checkout with sources and an importable scorer.

    Parameters
    ----------
    tmp_path : Path
        Parent of the temporary checkout; no user repository is modified.
    monkeypatch : pytest.MonkeyPatch
        Disables user/system Git configuration for deterministic local commits.

    Returns
    -------
    Path
        Committed source tree with bytecode ignored and unrelated files included.
    """

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "test checkout"
    root.mkdir()
    for name, content in {
        ".gitignore": "__pycache__/\n",
        "pyproject.toml": "[project]\nname = 'test'\n",
        "uv.lock": "version = 1\n",
        "src/nested/encoder.py": "ENCODER = 1\n",
        "src/kernels/compute.cu": "// CUDA source\n",
        "src/kernels/compute.h": "// Kernel header\n",
        "benchmarks/run.py": "RUN = 1\n",
        "benchmarks/README.md": "Not source code\n",
        "src/engine.trt": "Not source code\n",
        "tests/test_example.py": "Not benchmark/runtime source\n",
        "normalizer/__init__.py": "",
        "normalizer/eval_utils.py": "SCORER = 'local checkout'\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for args in (
        ("init", "--quiet", "--initial-branch=main", "--template="),
        ("config", "user.name", "Benchmark tests"),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "commit.gpgsign", "false"),
        ("add", "."),
        ("commit", "-qm", "Test checkout"),
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.fixture
def scorer(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the local main-branch scorer and restore its modules afterward.

    Parameters
    ----------
    checkout : Path
        Clean Git repository containing a tiny, dependency-free scorer package.
    monkeypatch : pytest.MonkeyPatch
        Isolates the import search path.

    Yields
    ------
    Path
        Checkout whose Git validation and Python imports both remain real.
    """

    monkeypatch.setattr(sys, "path", sys.path.copy())
    previous = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == "normalizer" or name.startswith("normalizer.")
    }
    for name in previous:
        del sys.modules[name]
    try:
        yield checkout
    finally:
        for name in tuple(sys.modules):
            if name == "normalizer" or name.startswith("normalizer."):
                del sys.modules[name]
        sys.modules.update(previous)


@pytest.fixture
def campaign() -> dict[str, JSONValue]:
    """Return a fresh, unsigned campaign covering the ordered dataset suite.

    Returns
    -------
    dict[str, JSONValue]
        Mutable metadata; nested protocol settings do not alias production data.
    """

    return {
        "protocol": deepcopy(common.PROTOCOL),
        "source": {"commit": "frozen", "files": {"src/model.py": "a" * 64}},
        "datasets": {
            name: [{"id": "a", "text": "  Raw reference  "}] for name in common.DATASETS
        },
    }


def write_campaign(path: Path, campaign: dict[str, JSONValue]) -> None:
    """Sign and write an unsigned campaign independently of the production helpers."""
    campaign["id"] = hashlib.sha256(
        json.dumps(campaign, sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(campaign))


def write_wav(
    path: Path, pcm: bytes, rate: int = 16000, channels: int = 1, width: int = 2
) -> None:
    """Write PCM bytes with an explicit WAV format, including invalid test cases.

    Parameters
    ----------
    path : Path
        Destination in an existing directory.
    pcm : bytes
        Interleaved PCM sample bytes.
    rate : int
        Source sample rate.
    channels : int
        Channel count stored in the header.
    width : int
        Bytes per sample per channel.
    """

    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)


@pytest.fixture
def rows() -> list[dict[str, JSONValue]]:
    """Provide unsorted durations with a tie at different source sample rates.

    Returns
    -------
    list[dict[str, JSONValue]]
        Three rows expected to sort as a, b, c, lasting one, one, and two seconds.
    """

    return [
        {"id": "c", "frames": 32000, "rate": 16000},
        {"id": "b", "frames": 8000, "rate": 8000},
        {"id": "a", "frames": 16000, "rate": 16000},
    ]


@pytest.fixture
def evidence(
    rows: list[dict[str, JSONValue]],
) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    """Pair two datasets with independently specified capacity-two timing records.

    Parameters
    ----------
    rows : list[dict[str, JSONValue]]
        First dataset, with a partial final batch.

    Returns
    -------
    tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]
        Campaign and records totaling 7.5 audio seconds and three compute seconds.
        The second dataset reuses an ID but must remain in a separate batch.
    """

    campaign = {
        "datasets": {
            "first": rows,
            "second": [
                {"id": "a", "frames": 48000, "rate": 16000},
                {"id": "d", "frames": 8000, "rate": 16000},
            ],
        }
    }
    records = [
        {
            "dataset": name,
            "ids": ids,
            "audio_seconds": audio,
            "inference_seconds": elapsed,
        }
        for name, ids, audio, elapsed in (
            ("first", ["a", "b"], 2.0, 0.25),
            ("first", ["c"], 2.0, 0.75),
            ("second", ["d", "a"], 3.5, 2.0),
        )
    ]
    return campaign, records


@pytest.mark.parametrize(
    "data", (b"", bytes(range(256)) * 4097), ids=("empty", "binary")
)
def test_file_hash(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(data)
    assert common.file_hash(path) == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileNotFoundError):
        common.file_hash(tmp_path / "missing.bin")


@pytest.mark.parametrize("existing", (False, True))
def test_write_json_creates_or_replaces_target_and_stale_temporary(
    tmp_path: Path, existing: bool
) -> None:
    path = tmp_path / "result.json"
    temporary = tmp_path / "result.json.tmp"
    if existing:
        path.write_text("old result")
    temporary.write_text("interrupted write")
    value = {"text": 'caf\u00e9\n"quoted"', "items": [None, True, 1, 0.5, {"a": []}]}
    common.write_json(path, value)
    content = path.read_text(encoding="utf-8")
    assert json.loads(content) == value
    assert content.endswith("\n")
    assert not temporary.exists()


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_write_json_rejects_nonfinite_values_without_replacing_target(
    tmp_path: Path, value: float
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"old": true}\n')
    with pytest.raises(ValueError, match="Out of range float values"):
        common.write_json(path, {"metrics": [value]})
    assert path.read_text() == '{"old": true}\n'


def test_write_json_failed_promotion_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"old": true}\n')
    replace = Mock(side_effect=OSError("promotion failed"))
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="promotion failed"):
        common.write_json(path, {"new": [1, 2]})
    replace.assert_called_once_with(path)
    assert path.read_text() == '{"old": true}\n'
    assert json.loads((tmp_path / "result.json.tmp").read_text()) == {"new": [1, 2]}


def test_load_campaign_propagates_missing_file_and_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    with pytest.raises(FileNotFoundError):
        common.load_campaign(path)
    path.write_text('{"incomplete":')
    with pytest.raises(json.JSONDecodeError):
        common.load_campaign(path)


def test_source_identity_hashes_only_relevant_files(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "ROOT", checkout)
    (checkout / "src/directory.py").mkdir()
    names = (
        "benchmarks/run.py",
        "pyproject.toml",
        "src/kernels/compute.cu",
        "src/kernels/compute.h",
        "src/nested/encoder.py",
        "uv.lock",
    )
    assert common.source_identity() == {
        "commit": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {
            name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
            for name in names
        },
    }


@pytest.mark.parametrize(
    "filename",
    ("pyproject.toml", "uv.lock", "src/nested/encoder.py", "benchmarks/new.py"),
)
def test_source_identity_detects_uncommitted_changes(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.setattr(common, "ROOT", checkout)
    before = common.source_identity()
    (checkout / filename).write_bytes(b"changed source\n")
    assert common.source_identity() == {
        "commit": before["commit"],
        "files": {
            **before["files"],
            filename: hashlib.sha256(b"changed source\n").hexdigest(),
        },
    }


def test_source_identity_detects_deleted_sources_and_requires_lockfile(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "ROOT", checkout)
    before = common.source_identity()
    (checkout / "src/nested/encoder.py").unlink()
    del before["files"]["src/nested/encoder.py"]
    assert common.source_identity() == before
    (checkout / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        common.source_identity()


def test_campaign_fingerprint_ignores_key_order_but_preserves_list_order(
    tmp_path: Path, campaign: dict[str, JSONValue]
) -> None:
    rows = campaign["datasets"][common.DATASETS[0]]
    rows.append({"id": "b", "text": "another reference"})
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    campaign["protocol"] = dict(reversed(campaign["protocol"].items()))
    path.write_text(json.dumps(dict(reversed(campaign.items())), indent=4))
    assert common.load_campaign(path) == campaign
    rows.reverse()
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="Campaign fingerprint mismatch"):
        common.load_campaign(path)


@pytest.mark.parametrize("changed", (False, True))
def test_load_campaign_checks_sources_only_when_requested(
    tmp_path: Path,
    campaign: dict[str, JSONValue],
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    original = path.read_text()
    identity = deepcopy(campaign["source"])
    if changed:
        identity["files"]["src/model.py"] = "b" * 64
    source = Mock(return_value=identity)
    monkeypatch.setattr(common, "source_identity", source)
    assert common.load_campaign(path) == campaign
    source.assert_not_called()
    if changed:
        with pytest.raises(ValueError, match="Sources or lockfile changed"):
            common.load_campaign(path, True)
    else:
        assert common.load_campaign(path, True) == campaign
    source.assert_called_once_with()
    assert path.read_text() == original


@pytest.mark.parametrize("passes", (1, 5))
def test_load_campaign_accepts_positive_pass_counts(
    tmp_path: Path, campaign: dict[str, JSONValue], passes: int
) -> None:
    campaign["protocol"]["passes"] = passes
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    assert common.load_campaign(path) == campaign


@pytest.mark.parametrize("passes", (0, -1, 1.0, "1", None))
def test_load_campaign_rejects_invalid_pass_counts(
    tmp_path: Path, campaign: dict[str, JSONValue], passes: JSONValue
) -> None:
    campaign["protocol"]["passes"] = passes
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    with pytest.raises(ValueError, match="different protocol"):
        common.load_campaign(path)


def test_load_campaign_rejects_tampered_reference_before_checking_source(
    tmp_path: Path, campaign: dict[str, JSONValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    campaign["datasets"][common.DATASETS[0]][0]["text"] = "changed reference"
    path.write_text(json.dumps(campaign))
    source = Mock()
    monkeypatch.setattr(common, "source_identity", source)
    with pytest.raises(ValueError, match="Campaign fingerprint mismatch"):
        common.load_campaign(path, True)
    source.assert_not_called()


@pytest.mark.parametrize("value", (None, [], "campaign", {}))
def test_load_campaign_rejects_nonobjects_or_missing_fingerprint(
    tmp_path: Path, value: JSONValue
) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Campaign fingerprint mismatch"):
        common.load_campaign(path)


@pytest.mark.parametrize(
    "key,value",
    (
        ("protocol", None),
        ("protocol", []),
        ("datasets", None),
        ("datasets", list(common.DATASETS)),
    ),
)
def test_load_campaign_requires_protocol_and_datasets_objects(
    tmp_path: Path, campaign: dict[str, JSONValue], key: str, value: JSONValue
) -> None:
    campaign[key] = value
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    with pytest.raises(ValueError, match="different protocol or dataset suite"):
        common.load_campaign(path)


@pytest.mark.parametrize("change", ("protocol", "missing", "extra", "order"))
def test_load_campaign_rejects_incompatible_suite_even_with_valid_digest(
    tmp_path: Path, campaign: dict[str, JSONValue], change: str
) -> None:
    if change == "protocol":
        campaign["protocol"]["audio_seconds"][2] = 60
    elif change == "missing":
        del campaign["datasets"][common.DATASETS[-1]]
    elif change == "extra":
        campaign["datasets"]["unexpected"] = []
    else:
        campaign["datasets"] = dict(reversed(campaign["datasets"].items()))
    path = tmp_path / "campaign.json"
    write_campaign(path, campaign)
    with pytest.raises(ValueError, match="different protocol or dataset suite"):
        common.load_campaign(path)


def test_upstream_propagates_git_failure(tmp_path: Path) -> None:
    original = sys.path.copy()
    with pytest.raises(subprocess.CalledProcessError):
        common.upstream(tmp_path / "missing-checkout")
    assert sys.path == original


def test_upstream_loads_main_scorer_and_restores_search_path(
    scorer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (scorer / "unrelated.txt").write_text("Outside the normalizer cleanliness check")
    monkeypatch.chdir(scorer.parent)
    original = sys.path.copy()
    module = common.upstream(Path(scorer.name))
    assert module.SCORER == "local checkout"
    assert Path(module.__file__).resolve() == scorer / "normalizer/eval_utils.py"
    assert sys.path == original
    assert common.upstream(scorer) is module
    assert sys.path == original


def test_upstream_accepts_new_commits_on_main(scorer: Path) -> None:
    (scorer / "normalizer/eval_utils.py").write_text("SCORER = 'updated main'\n")
    subprocess.run(
        ["git", "-C", str(scorer), "commit", "-qam", "Update scorer on main"],
        check=True,
        capture_output=True,
    )
    assert common.upstream(scorer).SCORER == "updated main"


@pytest.mark.parametrize(
    "args", (("-qb", "experiment"), ("-q", "--detach")), ids=("branch", "detached")
)
def test_upstream_requires_main_branch(scorer: Path, args: tuple[str, str]) -> None:
    subprocess.run(
        ["git", "-C", str(scorer), "checkout", *args], check=True, capture_output=True
    )
    original = sys.path.copy()
    with pytest.raises(ValueError, match="Scorer must be checked out on main"):
        common.upstream(scorer)
    assert sys.path == original
    assert "normalizer.eval_utils" not in sys.modules


@pytest.mark.parametrize("filename", ("eval_utils.py", "new.py"))
def test_upstream_rejects_dirty_normalizer(scorer: Path, filename: str) -> None:
    (scorer / "normalizer" / filename).write_text("CHANGED = True\n")
    original = sys.path.copy()
    with pytest.raises(ValueError, match="Scorer normalizer directory must be clean"):
        common.upstream(scorer)
    assert sys.path == original
    assert "normalizer.eval_utils" not in sys.modules


def test_upstream_rejects_cached_module_from_another_checkout(
    scorer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = scorer.with_name(scorer.name + "-other")
    package = other / "normalizer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "eval_utils.py").write_text("SCORER = 'wrong checkout'\n")
    monkeypatch.syspath_prepend(str(other))
    foreign = importlib.import_module("normalizer.eval_utils")
    original = sys.path.copy()
    with pytest.raises(
        ValueError, match="Another normalizer package was already imported"
    ):
        common.upstream(scorer)
    assert sys.path == original
    assert sys.modules["normalizer.eval_utils"] is foreign


def test_upstream_restores_search_path_after_import_error(scorer: Path) -> None:
    (scorer / "normalizer/eval_utils.py").write_text(
        "raise ImportError('scorer dependency unavailable')\n"
    )
    subprocess.run(
        ["git", "-C", str(scorer), "commit", "-qam", "Broken scorer import"],
        check=True,
        capture_output=True,
    )
    original = sys.path.copy()
    with pytest.raises(ImportError, match="scorer dependency unavailable"):
        common.upstream(scorer)
    assert sys.path == original
    assert "normalizer.eval_utils" not in sys.modules


@pytest.mark.parametrize(
    "rate,expected",
    (
        (16000, [-32768, -16384, 0, 16384, 32767]),
        (8000, [-32768, -24576, -16384, -8192, 0, 8192, 16384, 24575, 32767]),
        (32000, [-32768, 0, 32767]),
    ),
)
@pytest.mark.parametrize("verify_hash", (False, True))
def test_read_audio_pcm_scaling_resampling_and_original_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: int,
    expected: list[int],
    verify_hash: bool,
) -> None:
    path = tmp_path / "audio.wav"
    pcm = struct.pack("<5h", -32768, -16384, 0, 16384, 32767)
    write_wav(path, pcm, rate)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    resample = Mock(wraps=common.ratecv)
    monkeypatch.setattr(common, "ratecv", resample)
    audio, metadata = common.read_audio(path, checksum if verify_hash else None)
    np.testing.assert_array_equal(audio, np.array(expected, dtype=np.float32) / 32768.0)
    assert audio.dtype == np.float32
    assert metadata == {"sha256": checksum, "frames": 5, "rate": rate}
    if rate == 16000:
        resample.assert_not_called()
    else:
        resample.assert_called_once_with(pcm, 2, 1, rate, 16000, None)


def test_read_audio_checks_hash_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"not a WAV file")
    with pytest.raises(ValueError, match="Audio changed"):
        common.read_audio(path, "wrong hash")
    with pytest.raises(wave.Error):
        common.read_audio(path)


@pytest.mark.parametrize(
    "pcm,channels,width",
    ((b"", 1, 2), (b"\0" * 8, 2, 2), (b"\0" * 4, 1, 1)),
    ids=("empty", "stereo", "pcm8"),
)
def test_read_audio_rejects_unsupported_wav(
    tmp_path: Path, pcm: bytes, channels: int, width: int
) -> None:
    path = tmp_path / "audio.wav"
    write_wav(path, pcm, channels=channels, width=width)
    with pytest.raises(ValueError, match="Expected nonempty mono PCM16 WAV"):
        common.read_audio(path)


@pytest.mark.parametrize("change", ("truncated", "zero-rate"))
def test_read_audio_rejects_invalid_payload_or_rate(
    tmp_path: Path, change: str
) -> None:
    path = tmp_path / "audio.wav"
    write_wav(path, b"\0" * 8)
    data = bytearray(path.read_bytes())
    if change == "truncated":
        del data[-2:]
    else:
        struct.pack_into("<I", data, 24, 0)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Truncated or invalid WAV"):
        common.read_audio(path)


@pytest.mark.parametrize(
    "capacity,expected",
    ((1, [["a"], ["b"], ["c"]]), (2, [["a", "b"], ["c"]]), (256, [["a", "b", "c"]])),
)
def test_batches_sort_by_duration_then_id_without_changing_rows(
    rows: list[dict[str, JSONValue]], capacity: int, expected: list[list[str]]
) -> None:
    before = deepcopy(rows)
    result = common.batches(rows, capacity)
    assert [[row["id"] for row in batch] for batch in result] == expected
    assert [row for batch in result for row in batch] == [rows[2], rows[1], rows[0]]
    assert rows == before
    assert common.batches([], capacity) == []


@pytest.mark.parametrize("capacity", (-1, 0, 3, 512))
def test_batches_reject_unsupported_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match=f"Unsupported batch capacity: {capacity}"):
        common.batches([], capacity)


def test_pass_totals_pool_real_durations_and_compute_time(
    evidence: tuple[dict[str, JSONValue], list[dict[str, JSONValue]]],
) -> None:
    campaign, records = evidence
    before = deepcopy(evidence)
    assert common.pass_totals(campaign, records, 2) == {
        "audio_seconds": 7.5,
        "inference_seconds": 3.0,
        "rtfx": 2.5,
    }
    assert evidence == before
    records[0]["audio_seconds"] += 1e-12
    assert common.pass_totals(campaign, records, 2)["audio_seconds"] == 7.5


@pytest.mark.parametrize("change", ("missing", "extra", "reordered"))
def test_pass_totals_reject_incomplete_or_reordered_batches(
    evidence: tuple[dict[str, JSONValue], list[dict[str, JSONValue]]], change: str
) -> None:
    campaign, records = evidence
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append(records[-1].copy())
    else:
        records.reverse()
    message = "batch count differs" if change != "reordered" else "utterance mismatch"
    with pytest.raises(ValueError, match=message):
        common.pass_totals(campaign, records, 2)


@pytest.mark.parametrize(
    "key,value,message",
    (
        ("dataset", "second", "utterance mismatch"),
        ("ids", ["b", "a"], "utterance mismatch"),
        ("ids", ["a", "a"], "utterance mismatch"),
        ("ids", ["a"], "utterance mismatch"),
        ("ids", ["a", "b", "c"], "utterance mismatch"),
        ("audio_seconds", 4.0, "Audio duration"),
        ("audio_seconds", 2.0 + 1e-10, "Audio duration"),
        ("audio_seconds", float("nan"), "Audio duration"),
        ("inference_seconds", 0.0, "Invalid inference time"),
        ("inference_seconds", -0.5, "Invalid inference time"),
        ("inference_seconds", float("nan"), "Invalid inference time"),
        ("inference_seconds", float("inf"), "Invalid inference time"),
        ("inference_seconds", float("-inf"), "Invalid inference time"),
    ),
)
def test_pass_totals_reject_inconsistent_records(
    evidence: tuple[dict[str, JSONValue], list[dict[str, JSONValue]]],
    key: str,
    value: JSONValue,
    message: str,
) -> None:
    campaign, records = evidence
    records[0][key] = value
    with pytest.raises(ValueError, match=message):
        common.pass_totals(campaign, records, 2)
