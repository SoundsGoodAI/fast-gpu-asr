#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Campaign preparation with real WAVs, manifests, and checkpoint hashes."""

import csv
import json
import sys
import wave
from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from benchmarks import common, prepare

CHUNKED_DATASET = "earnings22_cleaned_aa_chunked_test"
REVISIONS = {
    "zipformer_rnnt": "0" * 40,
    "zipformer_cr_ctc_rnnt": "1" * 40,
    "parakeet_v2": "2" * 40,
    "parakeet_v3": "3" * 40,
}
REFERENCE = '  Caf\u00e9\t"quoted"\nnext line  '


def write_wav(path: Path, frames: int = 1600, rate: int = 16000) -> None:
    """Write a deterministic nonempty mono PCM16 waveform.

    Parameters
    ----------
    path : Path
        Destination whose parent directory already exists.
    frames : int
        Number of samples stored in the WAV.
    rate : int
        Original sample rate recorded in its header.
    """

    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x20" * frames)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write TSV rows with explicit quoting and the optional identity fields.

    Parameters
    ----------
    path : Path
        Manifest destination in an existing dataset directory.
    rows : list[dict[str, str]]
        Waveform paths, references, and optional IDs; omitted fields stay empty.
    """

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("wav_path", "original_text", "id", "parent_id", "chunk_index"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create the complete suite with unsorted IDs and resampled audio.

    Parameters
    ----------
    tmp_path : Path
        Isolated root for dataset directories and manifests.

    Returns
    -------
    Path
        Seven datasets with tied IDs, quoted/empty references, and chunk metadata.
    """

    root = tmp_path / "datasets"
    for name in common.DATASETS:
        directory = root / name
        directory.mkdir(parents=True)
        write_wav(directory / "a.wav", frames=800, rate=8000)
        write_wav(directory / "b.wav", frames=3200)
        write_manifest(
            directory / "references.tsv",
            [
                {
                    "wav_path": str(directory / "b.wav"),
                    "original_text": REFERENCE if name == CHUNKED_DATASET else "",
                    "id": "z",
                    "parent_id": "session",
                    "chunk_index": "1",
                },
                {
                    "wav_path": "a.wav",
                    "original_text": REFERENCE,
                    "parent_id": "session",
                    "chunk_index": "0",
                },
            ],
        )
    return root


@pytest.fixture
def checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide local model files and matching Git/LFS metadata without networking.

    Parameters
    ----------
    tmp_path : Path
        Root for four separate checkpoint directories and Zipformer sidecars.
    monkeypatch : pytest.MonkeyPatch
        Replaces only the Hugging Face request; hashing remains real.

    Returns
    -------
    SimpleNamespace
        Paths, mutable repository entries, and a request mock accepting only the
        pinned URLs. Each request gets a fresh response stream.
    """

    paths = {
        "zipformer_rnnt": tmp_path / "zipformer_rnnt" / "model.pt",
        "zipformer_cr_ctc_rnnt": tmp_path / "zipformer_cr_ctc_rnnt" / "model.pt",
        "parakeet_v2": tmp_path / "parakeet_v2" / "model.nemo",
        "parakeet_v3": tmp_path / "parakeet_v3" / "model.nemo",
    }
    entries = {name: [] for name in paths}
    for name, path in paths.items():
        path.parent.mkdir()
        filenames = (
            (path.name, "config.yaml", "bpe.model")
            if path.suffix == ".pt"
            else (path.name,)
        )
        for filename in filenames:
            data = f"contents of {name}/{filename}\n".encode()
            path.with_name(filename).write_bytes(data)
            entry = {"rfilename": filename}
            if filename == path.name:
                entry["lfs"] = {"sha256": sha256(data).hexdigest()}
            else:
                entry["blobId"] = sha1(
                    f"blob {len(data)}\0".encode() + data
                ).hexdigest()

            entries[name].append(entry)

    responses = {
        (
            f"https://huggingface.co/api/models/{common.MODELS[name]}"
            f"/revision/{REVISIONS[name]}?blobs=true"
        ): {"siblings": files}
        for name, files in entries.items()
    }
    query = Mock(
        side_effect=lambda url, timeout: BytesIO(json.dumps(responses[url]).encode())
    )
    monkeypatch.setattr(prepare, "urlopen", query)

    return SimpleNamespace(paths=paths, entries=entries, query=query)


@pytest.fixture
def campaign_run(
    dataset_root: Path, checkpoints: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Configure the real preparation CLI while isolating source/scorer checks.

    Parameters
    ----------
    dataset_root : Path
        Complete temporary dataset suite.
    checkpoints : SimpleNamespace
        Local checkpoints with mocked remote metadata.
    monkeypatch : pytest.MonkeyPatch
        Supplies CLI arguments and stable source/scorer identities.

    Returns
    -------
    SimpleNamespace
        Output/scorer paths and identity mocks; all local validation remains real.
    """

    output = dataset_root.parent / "new/campaign.json"
    scorer = dataset_root.parent / "scorer"
    argv = [
        "prepare",
        "--datasets-root",
        str(dataset_root),
        "--scorer",
        str(scorer),
        "--output",
        str(output),
    ]
    for name, path in checkpoints.paths.items():
        argv.extend((f"--{name}", str(path), f"--{name}-revision", REVISIONS[name]))
    monkeypatch.setattr(sys, "argv", argv)
    source = Mock(return_value={"commit": "frozen", "files": {"source.py": "hash"}})
    upstream = Mock()
    monkeypatch.setattr(prepare, "source_identity", source)
    monkeypatch.setattr(prepare, "upstream", upstream)
    return SimpleNamespace(
        output=output, scorer=scorer, source=source, upstream=upstream
    )


@pytest.mark.parametrize("passes", (1, 2, 3, 5))
def test_main_freezes_pass_count(campaign_run, monkeypatch, passes):
    default_passes = prepare.PROTOCOL["passes"]
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--passes", str(passes)])
    prepare.main()
    campaign = json.loads(campaign_run.output.read_text())
    assert campaign["protocol"]["passes"] == passes
    assert prepare.PROTOCOL["passes"] == default_passes


@pytest.mark.parametrize("passes", (0, -1))
def test_main_rejects_nonpositive_pass_count(
    campaign_run: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, passes: int
) -> None:
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--passes", str(passes)])
    with pytest.raises(ValueError, match="Measured pass count must be positive"):
        prepare.main()
    campaign_run.upstream.assert_not_called()
    campaign_run.source.assert_not_called()
    assert not campaign_run.output.exists()


@pytest.mark.parametrize("name", REVISIONS)
@pytest.mark.parametrize("revision", ("main", "a" * 39, "a" * 41, "g" * 40))
def test_main_rejects_mutable_or_invalid_revision(
    campaign_run: SimpleNamespace,
    checkpoints: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    revision: str,
) -> None:
    argv = sys.argv.copy()
    argv[argv.index(f"--{name}-revision") + 1] = revision
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="full Hugging Face commit SHA"):
        prepare.main()
    assert checkpoints.query.call_count == tuple(REVISIONS).index(name)
    assert not campaign_run.output.exists()


@pytest.mark.parametrize(
    "name,filename",
    (
        ("zipformer_rnnt", "model.pt"),
        ("zipformer_rnnt", "config.yaml"),
        ("zipformer_rnnt", "bpe.model"),
        ("zipformer_cr_ctc_rnnt", "model.pt"),
        ("zipformer_cr_ctc_rnnt", "config.yaml"),
        ("zipformer_cr_ctc_rnnt", "bpe.model"),
        ("parakeet_v2", "model.nemo"),
        ("parakeet_v3", "model.nemo"),
    ),
)
@pytest.mark.parametrize("change", ("modified", "missing", "not-in-revision"))
def test_main_rejects_unverified_checkpoint_files(
    campaign_run: SimpleNamespace,
    checkpoints: SimpleNamespace,
    name: str,
    filename: str,
    change: str,
) -> None:
    path = checkpoints.paths[name].with_name(filename)
    if change == "modified":
        path.write_bytes(b"changed")
        error, match = ValueError, "Checkpoint file differs"
    elif change == "missing":
        path.unlink()
        error, match = FileNotFoundError, filename
    else:
        checkpoints.entries[name][:] = [
            entry
            for entry in checkpoints.entries[name]
            if entry["rfilename"] != filename
        ]
        error, match = KeyError, filename
    with pytest.raises(error, match=match):
        prepare.main()
    assert not campaign_run.output.exists()


def test_freeze_preserves_references_paths_metadata_and_chunk_order(
    campaign_run: SimpleNamespace, dataset_root: Path
) -> None:
    prepare.main()
    result = json.loads(campaign_run.output.read_text(encoding="utf-8"))["datasets"]
    assert tuple(result) == common.DATASETS
    for name, rows in result.items():
        expected = []
        for filename, identifier, frames, rate, text in (
            ("a.wav", "a", 800, 8000, REFERENCE),
            ("b.wav", "z", 3200, 16000, ""),
        ):
            row = {
                "id": identifier,
                "path": f"{name}/{filename}",
                "text": text,
                "frames": frames,
                "rate": rate,
                "sha256": common.file_hash(dataset_root / name / filename),
            }
            if name == CHUNKED_DATASET:
                row.update(
                    text=REFERENCE, parent_id="session", chunk_index=len(expected)
                )
            expected.append(row)
        assert rows == expected


@pytest.mark.parametrize(
    "contents",
    (
        "wav_path\toriginal_text\na.wav\n",
        "wav_path\toriginal_text\na.wav\treference\textra\n",
        "wav_path\twrong_column\na.wav\treference\n",
        "wav_path\toriginal_text\toriginal_text\na.wav\tfirst\tsecond\n",
        "",
    ),
    ids=(
        "missing-reference",
        "extra-field",
        "missing-column",
        "duplicate-column",
        "no-header",
    ),
)
def test_freeze_rejects_malformed_manifests(
    campaign_run: SimpleNamespace, dataset_root: Path, contents: str
) -> None:
    manifest = dataset_root / common.DATASETS[0] / "references.tsv"
    manifest.write_text(contents)
    with pytest.raises(ValueError, match="manifest"):
        prepare.main()
    assert not campaign_run.output.exists()


def test_freeze_rejects_unterminated_quoted_reference(
    campaign_run: SimpleNamespace, dataset_root: Path
) -> None:
    manifest = dataset_root / common.DATASETS[0] / "references.tsv"
    manifest.write_text('wav_path\toriginal_text\na.wav\t"unterminated\n')
    with pytest.raises(csv.Error, match="unexpected end of data"):
        prepare.main()
    assert not campaign_run.output.exists()


def test_freeze_accepts_minimal_manifest_and_empty_reference(
    campaign_run: SimpleNamespace, dataset_root: Path
) -> None:
    name = common.DATASETS[0]
    (dataset_root / name / "references.tsv").write_text(
        "wav_path\toriginal_text\na.wav\t\n"
    )
    prepare.main()
    rows = json.loads(campaign_run.output.read_text(encoding="utf-8"))["datasets"][name]
    assert [(row["id"], row["text"]) for row in rows] == [("a", "")]


@pytest.mark.parametrize("missing", ("parent_id", "chunk_index"))
def test_freeze_requires_parent_columns(
    campaign_run: SimpleNamespace, dataset_root: Path, missing: str
) -> None:
    manifest = dataset_root / CHUNKED_DATASET / "references.tsv"
    fields = ["wav_path", "original_text", "parent_id", "chunk_index"]
    fields.remove(missing)
    manifest.write_text("\t".join(fields) + "\n")
    with pytest.raises(ValueError, match="Missing or duplicate manifest columns"):
        prepare.main()
    assert not campaign_run.output.exists()


@pytest.mark.parametrize(
    "problem", ("empty", "duplicate-id", "outside-root", "symlink")
)
def test_freeze_rejects_invalid_datasets(
    campaign_run: SimpleNamespace,
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    reader = Mock(wraps=prepare.read_audio)
    monkeypatch.setattr(prepare, "read_audio", reader)
    directory = dataset_root / common.DATASETS[0]
    rows = [{"wav_path": "a.wav", "original_text": "", "id": "a"}]
    if problem == "empty":
        rows = []
    elif problem == "duplicate-id":
        rows.append({**rows[0], "wav_path": "b.wav"})
    else:
        outside = dataset_root.parent / "outside.wav"
        write_wav(outside)
        if problem == "symlink":
            (directory / "link.wav").symlink_to(outside)
            rows[0]["wav_path"] = "link.wav"
        else:
            rows[0]["wav_path"] = "../../outside.wav"
    write_manifest(directory / "references.tsv", rows)
    with pytest.raises(
        ValueError,
        match="Empty dataset or duplicate utterance IDs"
        if problem in ("empty", "duplicate-id")
        else "outside.wav",
    ):
        prepare.main()
    assert not campaign_run.output.exists()
    if problem in ("outside-root", "symlink"):
        reader.assert_not_called()


@pytest.mark.parametrize("problem", ("stereo", "pcm8", "empty", "truncated"))
def test_freeze_rejects_invalid_audio(
    campaign_run: SimpleNamespace, dataset_root: Path, problem: str
) -> None:
    path = dataset_root / common.DATASETS[0] / "a.wav"
    channels = 2 if problem == "stereo" else 1
    width = 1 if problem == "pcm8" else 2
    frames = 0 if problem == "empty" else 1600
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, width, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * frames * channels * width)
    if problem == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(
        ValueError,
        match="Truncated or invalid WAV"
        if problem == "truncated"
        else "nonempty mono PCM16",
    ):
        prepare.main()
    assert not campaign_run.output.exists()


@pytest.mark.parametrize("rate", (8000, 16000, 44100))
@pytest.mark.parametrize("extra_sample", (0, 1))
def test_freeze_enforces_duration_boundary(
    campaign_run: SimpleNamespace, dataset_root: Path, rate: int, extra_sample: int
) -> None:
    directory = dataset_root / common.DATASETS[0]
    frames = 40 * rate + extra_sample
    write_wav(directory / "a.wav", frames=frames, rate=rate)
    if extra_sample:
        with pytest.raises(ValueError, match="40-second profile"):
            prepare.main()
        assert not campaign_run.output.exists()
    else:
        prepare.main()
        row = json.loads(campaign_run.output.read_text(encoding="utf-8"))["datasets"][
            common.DATASETS[0]
        ][0]
        assert (row["frames"], row["rate"]) == (frames, rate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("parent_id", ""),
        ("chunk_index", "2"),
        ("chunk_index", "0"),
        ("original_text", "conflicting reference"),
    ),
)
def test_freeze_preserves_parent_metadata_for_scoring(
    campaign_run: SimpleNamespace, dataset_root: Path, field: str, value: str
) -> None:
    manifest = dataset_root / CHUNKED_DATASET / "references.tsv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows[0][field] = value
    write_manifest(manifest, rows)
    prepare.main()
    rows = json.loads(campaign_run.output.read_text(encoding="utf-8"))["datasets"][
        CHUNKED_DATASET
    ]
    assert [row["id"] for row in rows] == ["a", "z"]
    output_field = "text" if field == "original_text" else field
    assert rows[1][output_field] == (int(value) if field == "chunk_index" else value)


def test_freeze_rejects_noninteger_chunk_index(
    campaign_run: SimpleNamespace, dataset_root: Path
) -> None:
    manifest = dataset_root / CHUNKED_DATASET / "references.tsv"
    write_manifest(
        manifest,
        [
            {
                "wav_path": "a.wav",
                "original_text": "",
                "parent_id": "session",
                "chunk_index": "1.5",
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid literal"):
        prepare.main()
    assert not campaign_run.output.exists()


def test_main_writes_loadable_campaign(
    campaign_run: SimpleNamespace, checkpoints: SimpleNamespace
) -> None:
    prepare.main()
    campaign = common.load_campaign(campaign_run.output)
    assert campaign["source"] == campaign_run.source.return_value
    assert "scorer_revision" not in campaign
    assert campaign["models"] == {
        name: {
            "repo": common.MODELS[name],
            "revision": REVISIONS[name],
            "files": {
                filename: common.file_hash(path.with_name(filename))
                for filename in (
                    ("model.pt", "config.yaml", "bpe.model")
                    if path.suffix == ".pt"
                    else ("model.nemo",)
                )
            },
        }
        for name, path in checkpoints.paths.items()
    }
    campaign_run.upstream.assert_called_once_with(campaign_run.scorer)
    assert campaign_run.source.call_args_list == [call(), call()]
    assert checkpoints.query.call_args_list == [
        call(
            f"https://huggingface.co/api/models/{repo}/revision/{REVISIONS[name]}?blobs=true",
            timeout=60,
        )
        for name, repo in common.MODELS.items()
    ]
    assert not campaign_run.output.with_suffix(".json.tmp").exists()


def test_main_refuses_existing_campaign(campaign_run: SimpleNamespace) -> None:
    campaign_run.output.parent.mkdir()
    campaign_run.output.write_bytes(b"existing campaign")
    with pytest.raises(FileExistsError, match="Output exists"):
        prepare.main()
    assert campaign_run.output.read_bytes() == b"existing campaign"
    campaign_run.source.assert_not_called()
    campaign_run.upstream.assert_not_called()


@pytest.mark.parametrize("failure", ("scorer", "request", "dataset", "source", "write"))
def test_main_never_publishes_failed_campaign(
    campaign_run: SimpleNamespace,
    checkpoints: SimpleNamespace,
    dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "scorer":
        campaign_run.upstream.side_effect = ValueError("wrong scorer branch")
    elif failure == "request":
        checkpoints.query.side_effect = TimeoutError("metadata request timed out")
    elif failure == "dataset":
        (dataset_root / common.DATASETS[-1] / "b.wav").unlink()
    elif failure == "source":
        campaign_run.source.side_effect = [{"commit": "before"}, {"commit": "after"}]
    else:
        monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("disk full")))
    with pytest.raises(
        (ValueError, OSError),
        match={
            "scorer": "wrong scorer branch",
            "request": "metadata request timed out",
            "dataset": "b.wav",
            "source": "Sources changed",
            "write": "disk full",
        }[failure],
    ):
        prepare.main()
    assert not campaign_run.output.exists()
    if failure == "write":
        common.load_campaign(campaign_run.output.with_suffix(".json.tmp"))
