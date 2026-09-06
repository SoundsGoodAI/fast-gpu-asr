# Full-Corpus GPU Benchmarks

This workflow measures the public `ASR` call. It does not use private evaluation
helpers or results from previous precision sweeps.

The commands below define a four-model H100 campaign using an isolated source
checkout and locked environment, with one measured pass per configuration.
Collect fresh measurements; do not reuse runs from the earlier two-model matrix.

## Matrix

| Model ID | Model | Search | Beam |
|---|---|---|---:|
| zipformer_rnnt | soundsgoodai/Zipformer-transducer-XL-290M | Transducer modified beam | 6 |
| zipformer_cr_ctc_rnnt | soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M | Transducer modified beam | 6 |
| parakeet_v2 | nvidia/parakeet-tdt-0.6b-v2 | TDT modified beam | 6 |
| parakeet_v3 | nvidia/parakeet-tdt-0.6b-v3 | TDT modified beam | 6 |

- GPUs: A100, H100, H200, B200, B300. Record the exact SKU and VRAM, not just this label.
- Fixed batch capacities: 1, 2, 4, 8, 16, 32, 64, 128, 256; H100 is capped at 128.
- Requested encoder **and** decoder precision: FP32, FP16, BF16.
- With the H100 cap: 528 configurations, 96 on H100 and 108 on each other GPU.
  Build engines separately on each target GPU.
- Duration profile: **0.1 / 8 / 40 seconds**. Builder optimization level: **5**.

The H100 sweep uses batches **1, 2, 4, 8, 16, 32, 64, 128**, all three precisions,
all four models: 96 configurations, with one measured pass each.
No machines are provisioned by these commands. Use one already available GPU,
one inference process, and no MPS or competing GPU workloads.

Precision labels are requests, not strict arithmetic guarantees. Keep exporter
defaults: FP32 permits TF32 and eligible reduced-math cuBLAS tactics. Frontend,
projection, search, and log-softmax operations can retain other configured
precisions. Changing precision can affect WER.

## Inputs

The frozen suite contains these seven datasets, separately batched:

| Local directory | Public source |
|---|---|
| ami_cleaned_test | hf-audio/open-asr-leaderboard, ami_cleaned, test |
| earnings22_cleaned_aa_chunked_test | ArtificialAnalysis/Earnings22-Cleaned-AA-chunked, test |
| gigaspeech_cleaned_test | hf-audio/open-asr-leaderboard, gigaspeech_cleaned, test |
| librispeech_test.clean | hf-audio/open-asr-leaderboard, librispeech, test.clean |
| librispeech_test.other | hf-audio/open-asr-leaderboard, librispeech, test.other |
| spgispeech_test | hf-audio/open-asr-leaderboard, spgispeech, test |
| voxpopuli_cleaned_aa_test | hf-audio/open-asr-leaderboard, voxpopuli_cleaned_aa, test |

Obtain the audio under each dataset's license from
[the public suite](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
and [Earnings22 chunks](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA-chunked).
Earnings22 parent references come from
[Earnings22-Cleaned-AA](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA).
Private, multilingual, and long-form datasets, including Earnings21, are excluded.

Place a `references.tsv` in each directory. Required columns:

```text
wav_path	original_text
soundfiles/utterance.wav	The original reference transcript.
```

`wav_path` may be absolute or relative to that dataset directory, but must resolve
inside the dataset root. An optional `id` supplies a stable utterance ID; otherwise
the WAV stem is used. IDs must be unique within a dataset. WAV files must be
nonempty mono PCM16; resampling to 16 kHz happens before timing. The importer
does not silently filter empty references or other rows: freeze the intended
upstream split, then every frozen row is measured and scored.

For Earnings22, also supply `parent_id` and integer `chunk_index` (starting at
zero with no gaps). Repeat the **complete parent reference**, not an invented
chunk reference, on each chunk's row. References for a parent must agree.

The campaign hashes every WAV and reference and stores portable relative paths.
It also verifies checkpoint files against immutable Hugging Face revisions,
records the repository HEAD, and fingerprints actual source files and `uv.lock`.
Uncommitted source changes are captured by hashes for pilot auditing, but publishable
runs should be frozen again from a committed source tree. Do not edit sources
during a campaign; collection rejects a changed fingerprint. Historical outputs
remain separate and cannot be mixed into this campaign's plots.

## Prepare

From the repository root, install the locked optional collection/scoring/plotting
dependencies and check out the public scorer's `main` branch:

```bash
uv sync --frozen --extra benchmark --extra dev
git clone --branch main https://github.com/huggingface/open_asr_leaderboard.git ../leaderboard-scorer
```

For an existing checkout, run `git -C ../leaderboard-scorer switch main` and
`git -C ../leaderboard-scorer pull --ff-only` before preparation or scoring.
The scripts require `main` and a clean normalizer directory, but do not fetch or
switch branches automatically. The scorer is not commit-pinned, so upstream
changes can change rescored WERs. Restart scoring after updating the checkout.

Build the native plugins first, as in the main README. Set paths to your local
copies and use the exact 40-character Hugging Face commits you downloaded:

```bash
export DATASETS=/path/to/datasets
export ZIPFORMER_RNNT=/path/to/zipformer-rnnt/model.pt
export ZIPFORMER_CR_CTC_RNNT=/path/to/zipformer-cr-ctc-rnnt/model.pt
export PARAKEET_V2=/path/to/parakeet-tdt-0.6b-v2.nemo
export PARAKEET_V3=/path/to/parakeet-tdt-0.6b-v3.nemo
export ZIPFORMER_RNNT_REVISION="40-character-model-commit"
export ZIPFORMER_CR_CTC_RNNT_REVISION="40-character-model-commit"
export PARAKEET_V2_REVISION="40-character-model-commit"
export PARAKEET_V3_REVISION="40-character-model-commit"
export CAMPAIGN=/path/to/benchmark-work/campaign.json

uv run --frozen --extra benchmark python -m benchmarks.prepare \
  --datasets-root "$DATASETS" --scorer ../leaderboard-scorer \
  --zipformer_rnnt "$ZIPFORMER_RNNT" \
  --zipformer_rnnt-revision "$ZIPFORMER_RNNT_REVISION" \
  --zipformer_cr_ctc_rnnt "$ZIPFORMER_CR_CTC_RNNT" \
  --zipformer_cr_ctc_rnnt-revision "$ZIPFORMER_CR_CTC_RNNT_REVISION" \
  --parakeet_v2 "$PARAKEET_V2" --parakeet_v2-revision "$PARAKEET_V2_REVISION" \
  --parakeet_v3 "$PARAKEET_V3" --parakeet_v3-revision "$PARAKEET_V3_REVISION" \
  --passes 1 --output "$CAMPAIGN"
```

The campaign contains references and belongs with your local datasets, not in a
release. Preparation refuses existing output files and audio over 40 seconds;
it never truncates a recording to make a configuration work.
Each Zipformer checkpoint must have its own matching `config.yaml` and `bpe.model`
alongside it; all three files are verified against that variant's revision.

Preparation defaults to one measured pass. Set `--passes N` to any positive
integer, for example `--passes 1` for a single-pass campaign. Collection, scoring,
and reporting use that saved count automatically; do not combine campaigns with
different pass counts.

## Collect and Score

Run the H100 sweep, with no other GPU process running:

```bash
uv run --frozen --extra benchmark python -m benchmarks.run \
  --campaign "$CAMPAIGN" --datasets-root "$DATASETS" \
  --zipformer_rnnt "$ZIPFORMER_RNNT" --zipformer_cr_ctc_rnnt "$ZIPFORMER_CR_CTC_RNNT" \
  --parakeet_v2 "$PARAKEET_V2" --parakeet_v3 "$PARAKEET_V3" \
  --scorer ../leaderboard-scorer --gpu H100 --device-id 0 \
  --batches 1 2 4 8 16 32 64 128 --precisions fp32 fp16 bf16 \
  --output /path/to/benchmark-work/h100-full --discard-engines
```

The workflow uses separate sequential export, inference, and scoring subprocesses
so builder allocations cannot leak into measured inference. `--discard-engines`
removes only newly generated run bundles; their hashes and export logs remain.
Omit it to retain engines. Existing output directories are refused. To retry a
failure, use a new run root and retain the old failure record.

Without `--batches`, all nine capacities run on every GPU. For the documented H100
sweep, explicitly pass `--batches 1 2 4 8 16 32 64 128`.
`--models parakeet_v3`
selects one variant; `--models zipformer_rnnt zipformer_cr_ctc_rnnt` selects both
Zipformers. The CLI still requires all four checkpoint path options. Other GPUs
use the same command and the corresponding `--gpu` label.
A model's source files, beam, precision, profile,
and batch capacity must not be substituted after a failure. Checkpoint filenames
must match the frozen campaign; checkpoint and sidecar hashes are verified before
the matrix starts and after every export, before collection can begin.

Each dataset is sorted by descending audio duration (`frames / rate`), with
ascending utterance IDs breaking ties, before batches are formed. Its final
partial batch contains the shortest clips and is included; datasets never share
a batch. Freeze a new campaign for this ordering; ascending-order campaigns
require their original benchmark code for scoring and reporting. Warm-up is
five untimed real batches selected at duration quantiles across the suite. Then
the campaign's measured passes run, without additional hidden warm-up calls.

The timer starts after a stream synchronization and stops after the full
`ASR(audios)` call and another synchronization. This includes staging, transfers,
features, acoustic encoding, search, text, and word timestamps. Disk reading,
resampling, validation/hashing, sorting, serialization, WER scoring, and model
loading are outside the timer. Shape-dependent allocation and graph capture
inside an ASR call remain included.

Per pass:

```text
pooled RTFx = sum(real input audio seconds) / sum(full ASR batch seconds)
```

Padding and unused engine rows never contribute audio seconds. Do not average
dataset RTFx. Three-pass charts use the median of complete-pass ratios, with their
minimum and maximum. Single-pass charts show the one measured ratio and do not
estimate repeatability. Host staging and postprocessing are included, so these are
end-to-end **system** measurements, not isolated GPU-kernel comparisons.

Scoring calls upstream `main`'s
[`score_results`](https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/eval_utils.py),
which normalizes references
and hypotheses and uses `kaldialign.batch_error_rate(..., merge_compounds=True)`.
Earnings22 hypotheses are joined in chunk-index order by parent before scoring.
All per-dataset WERs from all passes are retained, including unfavorable values
and empty hypotheses.
The displayed mean WER is the unweighted mean of seven upstream-rounded WERs;
it is distinct from pooled timing. These are not official leaderboard submissions.

Scoring requires hashes for every configured pass and rechecks evidence and run
metadata before replacing summaries. A scoring or validation failure preserves
existing summaries; scorer logs are replaced for attempted passes. JSON files are
replaced individually, with `result.json` written last. A disk-write failure can
leave partially updated per-pass summaries; standalone rescoring retains the
previous final result in that case.

Run directories retain `run.json`, build/collection/scoring logs, hardware
metadata, bundle/plugin hashes, warm-up IDs, and for each pass:
`batches.jsonl`, dataset transcript JSONLs (with timestamps), totals, and scores.
Warm-up entries include both the dataset name and utterance IDs. Collection
verifies bundle and plugin inventories/hashes before model loading and after all
passes, and refuses directories containing previous collection output. The CUDA
device used for inference must match the recorded physical GPU UUID. MIG mode
is recorded as metadata, not rejected. The orchestrator supplies the device
mapping through `CUDA_VISIBLE_DEVICES`;
direct `benchmarks.collect` invocations must set it before starting Python.
GPU process checks run before model loading, after warm-up, and after each dataset.
They are snapshots, not an exclusive reservation; the machine must still be kept
free of competing workloads throughout the run.
`result.json` records success or the failing stage and exception. OOM/build
diagnostics remain in `build.log`. A killed process can leave a `running` record;
that is incomplete, never a valid plotted result.
The root `matrix.json` records every requested configuration, including those
not reached after an interruption. `collection.json` is written only after all
passes and the final source check succeed. Scoring requires this marker and
verifies its evidence hashes; it cannot promote a source-invalidated collection.

## Report

Plots use Plotly, with Kaleido exporting twelve SVGs, one per model and precision,
embedded in the README.
Static export requires Chrome or Chromium on the reporting host, not on GPU
collection hosts. If no compatible browser is installed, install one once:

```bash
uv run --frozen --extra benchmark plotly_get_chrome -y
```

An existing browser can also be selected with the `BROWSER_PATH` environment
variable. See [Plotly's static export setup](https://plotly.com/python/static-image-export/).

```bash
uv run --frozen --extra benchmark python -m benchmarks.report \
  --campaign "$CAMPAIGN" --runs /path/to/benchmark-work/h100-full \
  --output docs/benchmarks --readme README.md
```

Run this again with all GPU result roots after the full matrix finishes. Only
complete suites with matching evidence can enter plots. Duplicate complete
configurations are rejected rather than cherry-picked. Public `results.json`
contains only complete measurements. All figures share GPU colors, axis limits, and
powers-of-two batch ticks. Missing measurements stay explicitly unmeasured.

Reporting cross-checks `run.json`, hardware and export hash manifests, the
`collection.json` source snapshot, exact transcript/timing hashes, and every
pass's `scores.json`. Pooled and per-dataset durations are recomputed from frozen
frame counts and sample rates, and inference time from complete batch timings;
mean WERs and summary medians are derived again. Transcript rows are checked as
they are read, without keeping all hypothesis text or word timestamps in memory.
Saved WERs and error counts are validated for consistency, not rescored. These
checks detect inconsistent artifacts, not coordinated changes to all copies of
the evidence.
Discarded engines are not required on the reporting host. Reporting accepts all
configured batch capacities, including H100 batch 256; the narrower H100 sweep
above is a collection choice, not a reporting restriction. Complete runs sharing
a GPU label must have matching hardware/software snapshots. Cross-run comparisons
parse `lscpu` JSON and ignore only live `CPU MHz:` and `CPU(s) scaling MHz:` values,
including nested entries. CPU identity, topology, maximum clock, and other
metadata still must match. Original snapshots are preserved in public results;
each run's result/hardware consistency check remains exact.

`--runs` accepts matrix roots or individual run directories. Missing/empty roots
and malformed matrices fail explicitly; malformed individual results are logged
and excluded. Failed and incomplete attempts remain in their local run directories,
not in public JSON. Their raw diagnostics may contain transcript text.

Use an output directory separate from the private campaign and run evidence.
Generated files and any README update are staged before replacing an existing
report, so validation, staging, or browser export failures leave it unchanged.
Generated filenames cannot conflict with existing directories.
README updates require exactly one ordered
`<!-- benchmark-results:start -->` / `<!-- benchmark-results:end -->`
marker pair. Paths in README links are relative and URL-escaped; content outside
the markers is preserved. README content is rechecked immediately before its
replacement; a detected edit leaves the README untouched. This check is not a
filesystem lock: use one report writer and avoid editing the README during
publication. Each file is replaced individually, not as a single multi-file
transaction; an error after publication starts can leave a
partially updated report.

The report writes twelve SVGs, `results.md`, machine-readable measurements, and a
public input manifest with reference hashes instead of reference text. It updates
only the marked README block. Publish these generated artifacts, not local audio,
checkpoints, TensorRT engines, or private reference manifests. Preserve raw
per-pass transcripts locally for audit and rescoring:

```bash
uv run --frozen --extra benchmark python -m benchmarks.score \
  --campaign "$CAMPAIGN" --scorer ../leaderboard-scorer \
  --run-dir /path/to/benchmark-work/h100-full/H100-zipformer_cr_ctc_rnnt-fp16-b64
```

Reference hashes in the public manifest use SHA-256 of
`json.dumps(reference_text, sort_keys=True).encode()`, with default JSON spacing
and ASCII escaping. The public manifest is a redacted view
of the campaign, not a replacement for the private campaign file passed to the CLI.
