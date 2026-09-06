#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Freeze portable dataset manifests and verified checkpoint/source identities.

Run via ``python -m benchmarks.prepare`` before exporting benchmark bundles.
Preparation reads local audio and checkpoints; only checkpoint metadata is fetched
from Hugging Face. References remain unchanged for the upstream scorer on main.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from re import fullmatch
from urllib.request import urlopen

from .common import (
    DATASETS,
    MODELS,
    PROTOCOL,
    JSONValue,
    file_hash,
    read_audio,
    source_identity,
    upstream,
    write_json,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse inputs and the destination for a frozen benchmark campaign.

    Returns
    -------
    argparse.Namespace
        Dataset root, scorer checkout, output path, measured pass count, and each
        model's checkpoint path and immutable Hugging Face revision.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root", type=Path, required=True, help="Materialized dataset root."
    )
    parser.add_argument(
        "--scorer", type=Path, required=True, help="Upstream scorer checkout on main."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New campaign JSON file."
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=PROTOCOL["passes"],
        help="Positive measured pass count per configuration, frozen in the campaign.",
    )

    for name in MODELS:
        parser.add_argument(
            f"--{name}", type=Path, required=True, help=f"Local {name} checkpoint."
        )
        parser.add_argument(
            f"--{name}-revision", required=True, help="Full Hugging Face commit SHA."
        )

    return parser.parse_args()


def main() -> None:
    """Verify inputs and write a new campaign only after preparation succeeds.

    The scorer checkout is verified before reading checkpoints and datasets.
    Checkpoints must retain their repository-root filenames and match full,
    lowercase Hugging Face commit SHAs. Zipformer additionally requires adjacent
    ``config.yaml`` and ``bpe.model`` files. Large LFS checkpoints are hashed
    incrementally; ordinary Git files are read into memory.

    Dataset manifests contain mono PCM16 WAV paths that resolve inside the root.
    Rows are saved in ID order with portable paths, exact audio hashes, and raw
    references. Earnings22 additionally retains parent IDs and integer chunk
    indexes for upstream scoring.
    Empty reference strings are valid; missing reference fields are not. Text is
    never stripped or normalized. Missing IDs default to the WAV filename stem.
    Audio is resampled only to check the encoder's sample limit; saved durations
    use the original sample counts and rates. No recording is filtered or cropped.

    Source fingerprints must match before and after preparation. The final digest
    covers protocol settings, source files, verified checkpoint identities, and
    every dataset row, including raw reference text. Failures propagate without
    publishing a campaign; existing output is refused.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    if args.passes < 1:
        raise ValueError("Measured pass count must be positive.")

    if args.output.exists():
        raise FileExistsError("Output exists; preserve it and use a new campaign path.")

    upstream(args.scorer)

    campaign: dict[str, JSONValue] = {
        "protocol": {**PROTOCOL, "passes": args.passes},
        "source": source_identity(),
        "models": {},
        "datasets": {},
    }

    for name, repo in MODELS.items():
        path = getattr(args, name)
        revision = getattr(args, f"{name}_revision")
        if fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Model revision must be a full Hugging Face commit SHA.")

        paths = [path]
        if name in ("zipformer_rnnt", "zipformer_cr_ctc_rnnt"):
            paths += [path.with_name("config.yaml"), path.with_name("bpe.model")]

        url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        with urlopen(url, timeout=60) as response:
            info = json.load(response)

        remote = {r["rfilename"]: r for r in info["siblings"]}
        files = {}
        for p in paths:
            entry = remote[p.name]
            if "lfs" in entry:
                sha256 = file_hash(p)
                matches = sha256 == entry["lfs"]["sha256"]
            else:
                content = p.read_bytes()
                sha256 = hashlib.sha256(content).hexdigest()
                # Git hashes the blob header and content; LFS hashes content only.
                blob = b"blob " + str(len(content)).encode() + b"\0" + content
                matches = hashlib.sha1(blob).hexdigest() == entry["blobId"]

            if not matches:
                raise ValueError(f"Checkpoint file differs from {revision}: {p}")

            files[p.name] = sha256

        campaign["models"][name] = {"repo": repo, "revision": revision, "files": files}

    root = args.datasets_root.resolve()
    max_seconds = PROTOCOL["audio_seconds"][2]
    max_samples = round(max_seconds * PROTOCOL["sample_rate"])
    for name in DATASETS:
        rows = []

        required = {"wav_path", "original_text"}
        if name == "earnings22_cleaned_aa_chunked_test":
            required.update(("parent_id", "chunk_index"))

        manifest = root / name / "references.tsv"
        with open(manifest, encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", strict=True)
            fields = reader.fieldnames or []
            if not required.issubset(fields) or len(fields) != len(set(fields)):
                raise ValueError(f"Missing or duplicate manifest columns: {manifest}")

            for item in reader:
                if None in item or any(item[key] is None for key in required):
                    raise ValueError(
                        f"Malformed manifest row at line {reader.line_num}: {manifest}"
                    )

                path = Path(item["wav_path"])
                if not path.is_absolute():
                    path = root / name / path

                relative = path.resolve().relative_to(root)
                audio, metadata = read_audio(path)
                if len(audio) > max_samples:
                    raise ValueError(
                        f"Audio exceeds the fixed {max_seconds}-second profile: {path}"
                    )

                row = {
                    "id": item.get("id") or relative.stem,
                    "path": relative.as_posix(),
                    "text": item["original_text"],
                    **metadata,
                }
                if name == "earnings22_cleaned_aa_chunked_test":
                    row.update(
                        parent_id=item["parent_id"],
                        chunk_index=int(item["chunk_index"]),
                    )

                rows.append(row)

        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty dataset or duplicate utterance IDs: {name}")

        campaign["datasets"][name] = sorted(rows, key=lambda r: r["id"])
        logger.info("Frozen %s: %d utterances", name, len(rows))

    if source_identity() != campaign["source"]:
        raise ValueError("Sources changed during preparation; freeze a new campaign.")

    campaign["id"] = hashlib.sha256(
        json.dumps(campaign, sort_keys=True).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, campaign)
    logger.info("Campaign: %s", campaign["id"])


if __name__ == "__main__":
    main()
