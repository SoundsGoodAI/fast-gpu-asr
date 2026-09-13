# Benchmark Methodology

The benchmark lineup is **A100, H200, and B300**. The [results](results.md)
cover **156 configurations**: 52 each on A100, H200, and B300.
Each GPU covers two models and three precisions: batches 1, 2, 4, 8, 16, 32, 64,
and 128 in FP16/BF16/FP32, plus batch 256 in FP16/BF16.

## Models and Source

Both models use **transducer modified beam search with beam 6**. The CR-CTC
checkpoint is decoded through its transducer head, not its CTC head.

- [Zipformer CR-CTC-transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M)
- [Parakeet TDT 0.6B V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

Model checkpoints, source code, the dependency lockfile, dataset manifests, and
the upstream scorer were frozen for the campaign. Exact revisions and checksums
remain with the local experiment records, rather than being listed here.
These measurements describe that frozen environment, not subsequent changes to
the code, checkpoints, or scorer.

## Hardware and Execution

| Label | GPU | Nominal VRAM (GB) | Power limit | CPU affinity (logical CPUs) | CPU model |
|---|---|---:|---:|---:|---|
| A100 | NVIDIA A100 SXM4 | 80 | 400 W | 22 | AMD EPYC 7643 |
| H200 | NVIDIA H200 SXM5 | 141 | 700 W | 32 | AMD EPYC 9654 |
| B300 | NVIDIA B300 SXM6 | 288 | 1,100 W | 30 | AMD EPYC 9575F |

Common recorded software: NVIDIA driver **580.126.20**, CUDA runtime **13.2**,
CUDA driver API **13.0**, TensorRT-cu13 **11.2.1.2**,
Python **3.14.5**, CuPy-cuda13x **14.2.0**, NumPy **2.5.2**, PyTorch
**2.14.0+cpu**, and cuda-bindings **13.3.1**. CUDA runtime and driver API version
numbers are reported separately; they are not inferred from a toolkit label.

Each run uses one GPU and one inference process. GPU process checks guard against
competing workloads before model loading, after warm-up, and after each dataset.
The CPU counts above describe process affinity. Host CPU resources differ, so
these are end-to-end **system measurements**, not isolated GPU-kernel or
hardware-only comparisons.

Each configuration gets separate engines built on its target GPU. Both encoder
and decoder request the named precision. All use **0.1 / 8 / 40-second**
minimum/typical/maximum audio segment duration profiles and TensorRT builder
optimization level **5**. Exporter optimizations remain enabled: FP32 permits
TF32 and eligible reduced-math tactics; some frontend/output operations retain
their configured precision.

## Dataset Suite

All configurations use the same **72,341 clips**, totaling
**157.77 audio hours**.

| Local dataset | Clips | Audio hours |
|---|---:|---:|
| `ami_cleaned_test` | 7,715 | 7.98 |
| `earnings22_cleaned_aa_chunked_test` | 341 | 1.92 |
| `gigaspeech_cleaned_test` | 18,757 | 35.14 |
| `librispeech_test.clean` | 2,620 | 5.40 |
| `librispeech_test.other` | 2,939 | 5.34 |
| `spgispeech_test` | 39,341 | 100.00 |
| `voxpopuli_cleaned_aa_test` | 628 | 1.98 |

Sources are the [public English leaderboard suite](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
and [Earnings22 chunks](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA-chunked),
with [Earnings22 parent references](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA).
Private, multilingual, and long-form suites, including Earnings21, are outside
these measurements.

## Timing and Scoring

Each configuration receives **five untimed warm-up calls** spanning real batch
duration quantiles, then **one complete measured pass**. Within each dataset, clips
are sorted by **descending duration**, breaking ties by ascending utterance ID.
Datasets never share a batch; the final partial batch is always included.
Audio is read and resampled to 16 kHz before timing. Nothing is cropped or
dropped to fit an engine, and padding never contributes audio seconds.

```text
RTFx = total real audio seconds / total synchronized ASR call seconds
```

**Pooled RTFx** means summing real audio seconds across all seven datasets and
dividing by their summed inference seconds, within one configuration. It is not
the average of the seven dataset RTFx values. RTFx measures throughput, not
individual-request latency.

The GPU stream is synchronized immediately before and after each `ASR(audios)`
call. Timing includes staging, transfers, features, encoding, decoding, text,
word timestamps, and shape-dependent allocation or graph capture inside the call.
Model loading, file reading, resampling, sorting, hashing, serialization, and
scoring are excluded.

Scoring uses the campaign's frozen [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)
tooling via `score_results(..., language="en")`. It applies
`EnglishTextNormalizer` to both references and hypotheses, splits the normalized
text into words, and calls
`kaldialign.batch_error_rate(..., merge_compounds=True)`. Compound-word merging
tolerates matching split/joined forms such as "white paper" and "whitepaper";
these scores therefore use upstream's conventions, not plain unnormalized
word-string comparison. Earnings22 hypotheses are joined by parent and chunk
index before normalization and scoring against complete parent references.

```text
Dataset WER (%) = 100 * (substitutions + deletions + insertions) / reference words
Mean WER (%) = sum(seven upstream-rounded dataset WER percentages) / 7
```

Error counts follow upstream's alignment and compound-merging rules. Upstream
rounds each dataset WER to two decimal places. **Each dataset has equal weight**,
regardless of duration, clip count, or reference-word count. We report **mean WER**
across all seven datasets. Every configuration's per-dataset WER is retained,
including unfavorable results and empty hypotheses.

## Interpretation and Evidence

Each timing/WER pair comes from the same pass. Single-pass results do not estimate
repeatability or establish significance for small differences.

Only complete suite passes with validated timing and scoring enter the report.
The public CSV preserves each finalized configuration. It contains summary
timings and all seven WERs, not transcripts or per-batch evidence. It cannot
independently establish what audio was decoded or reproduce WER scoring by itself.
The CSV was cross-checked against the finalized A100, H200, and B300 reports.

## Rebuild Plots

The renderer reads [measurements.csv](measurements.csv), derives RTFx and mean
WER, and generates both SVGs, [results.md](results.md), and the main README's
marked performance block. It needs **no GPU, checkpoints, audio, or private
helpers**. From the repository root in the locked environment:

```bash
uv sync --frozen --extra benchmark
uv run --frozen --extra benchmark plotly_get_chrome -y
uv run --frozen --extra benchmark python docs/benchmarks/render.py
```

Skip the browser installation when Chrome/Chromium is already available, or set
`BROWSER_PATH` to its executable. The renderer rejects duplicate configurations,
missing batches within an included GPU/model/precision series, missing dataset
scores, and invalid numeric values. It stages all SVG exports before replacing
generated documentation. SVG IDs are normalized for stable regeneration with the
same plotting stack. Content inside the README benchmark markers is regenerated.

The CSV contains one row per GPU/model/precision/batch configuration. Timing
columns are seconds; `wer_*` columns are percentages in the dataset order above.
The common beam is 6. All three GPUs have complete 52-row matrices.
Only A100, H200, and B300 are accepted by this publication renderer.
FP32 curves stop at batch 128; FP16/BF16 continue to 256.
SVGs are separate assets embedded in the main README; the results page contains
the complete per-GPU timing and WER tables.

## Collect New Measurements

This workflow collects a new campaign; it is not an exact replay of the published
measurements. Freeze and record its source, model, and scorer revisions. Keep new
evidence outside the documentation directory and do not present newly rescored
values as the published results. The repeated-waveform microbenchmark remains
separate from full-corpus measurements.

Use an isolated source checkout, one already available **GPU**, one inference
process, no MPS, and no competing GPU workload. `--device-id` selects the physical
GPU by its `nvidia-smi` index; `--gpu` supplies a reporting label such as `H200` or
`RTX_PRO_6000`, not a hardware filter. Labels must start with a letter or digit and
contain only letters, digits, underscores, or hyphens. The actual device name and
UUID are recorded; UUID checks verify device identity. The GPU must support the
runtime, plugins, and requested precisions. Build the
[native plugins](../../README.md#build-from-source)
before collecting. Freeze the source and environment before preparation; do not
edit them during a campaign.

The two-model matrix uses **Zipformer CR-CTC-transducer and Parakeet V3, beam 6**:
batches **1-128 in powers of two** for FP32/FP16/BF16, plus **256 for FP16/BF16**.
That is **52 configurations per GPU**, using one measured pass, five warm-ups,
**0.1 / 8 / 40-second** profiles, and optimization level **5**. Both engines request
the named precision with exporter math defaults unchanged.

### Dataset Inputs

Materialize these seven datasets under one directory:

| Local directory | Public dataset/configuration, split |
|---|---|
| `ami_cleaned_test` | hf-audio/open-asr-leaderboard, ami_cleaned, test |
| `earnings22_cleaned_aa_chunked_test` | ArtificialAnalysis/Earnings22-Cleaned-AA-chunked, test |
| `gigaspeech_cleaned_test` | hf-audio/open-asr-leaderboard, gigaspeech_cleaned, test |
| `librispeech_test.clean` | hf-audio/open-asr-leaderboard, librispeech, test.clean |
| `librispeech_test.other` | hf-audio/open-asr-leaderboard, librispeech, test.other |
| `spgispeech_test` | hf-audio/open-asr-leaderboard, spgispeech, test |
| `voxpopuli_cleaned_aa_test` | hf-audio/open-asr-leaderboard, voxpopuli_cleaned_aa, test |

Obtain audio under its dataset license from the
[public English suite](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
and [Earnings22 chunks](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA-chunked).
Private, multilingual, and long-form suites, including Earnings21, are excluded.

Each directory needs a tab-separated `references.tsv`:

```text
wav_path	original_text
soundfiles/utterance.wav	The original reference transcript.
```

Paths may be absolute or relative to the dataset directory, but must resolve
inside the dataset root. Use nonempty mono PCM16 WAVs within the protocol's
maximum audio duration, currently **40 seconds**. Resampling to 16 kHz happens
before timing. Optional `id` values must be unique
within the dataset; otherwise the WAV stem is used. No audio or references are
silently dropped, cropped, or pre-normalized.

Earnings22 also requires `parent_id` and contiguous, zero-based `chunk_index`
columns. Repeat the **complete parent reference** from
[Earnings22-Cleaned-AA](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA)
on each parent's rows, not a chunk-level reference.

### Prepare a Campaign

Run from the repository root:

```bash
uv sync --frozen --extra benchmark --extra dev
git clone --branch main https://github.com/huggingface/open_asr_leaderboard.git ../leaderboard-scorer
```

For an existing scorer checkout, switch to `main` and `git pull --ff-only` first.
The CLI requires `main` and a clean normalizer directory. Scorer updates can change
WER; restart scoring after an update.

Use `--models` to select checkpoints during preparation and collection. Preparation
requires paths and revisions only for the selected models; collection requires
only their paths, and every selected model must be in the prepared campaign.
Omitting `--models` selects all four supported models in either command.
Keep original checkpoint filenames; each Zipformer needs its matching
`config.yaml` and `bpe.model` beside `model.pt`.
Use the full 40-character Hugging Face revisions you downloaded:

```bash
export DATASETS=/path/to/datasets
export ZIPFORMER_CR_CTC_RNNT=/path/to/zipformer-cr-ctc-rnnt/model.pt
export PARAKEET_V3=/path/to/parakeet-tdt-0.6b-v3.nemo
export ZIPFORMER_CR_CTC_RNNT_REVISION="40-character-model-commit"
export PARAKEET_V3_REVISION="40-character-model-commit"
export CAMPAIGN=/path/to/benchmark-work/campaign.json
export RUN_ROOT=/path/to/benchmark-work/h200-full
export GPU=H200
mkdir -p "$(dirname "$CAMPAIGN")"

uv run --frozen --extra benchmark python -m benchmarks.prepare \
  --datasets-root "$DATASETS" --scorer ../leaderboard-scorer \
  --models zipformer_cr_ctc_rnnt parakeet_v3 \
  --zipformer_cr_ctc_rnnt "$ZIPFORMER_CR_CTC_RNNT" \
  --zipformer_cr_ctc_rnnt-revision "$ZIPFORMER_CR_CTC_RNNT_REVISION" \
  --parakeet_v3 "$PARAKEET_V3" --parakeet_v3-revision "$PARAKEET_V3_REVISION" \
  --passes 1 --output "$CAMPAIGN"
```

Preparation verifies checkpoint/sidecar hashes against those revisions and freezes
every WAV, reference, protocol setting, and source/lockfile fingerprint. It refuses
existing output. The campaign contains private references; keep it outside Git.
`--passes N` supports multiple complete passes, but do not mix pass counts or
source/scorer environments when comparing results.

### Collect and Score

Set `GPU` and a new `RUN_ROOT` for each target. Engines are built on that GPU;
export, inference, and scoring run in separate sequential subprocesses:

```bash
uv run --frozen --extra benchmark python -m benchmarks.run \
  --campaign "$CAMPAIGN" --datasets-root "$DATASETS" \
  --zipformer_cr_ctc_rnnt "$ZIPFORMER_CR_CTC_RNNT" --parakeet_v3 "$PARAKEET_V3" \
  --scorer ../leaderboard-scorer --gpu "$GPU" --device-id 0 \
  --models zipformer_cr_ctc_rnnt parakeet_v3 \
  --batches 1 2 4 8 16 32 64 128 --precisions fp32 fp16 bf16 \
  --output "$RUN_ROOT/b1-b128" --discard-engines

uv run --frozen --extra benchmark python -m benchmarks.run \
  --campaign "$CAMPAIGN" --datasets-root "$DATASETS" \
  --zipformer_cr_ctc_rnnt "$ZIPFORMER_CR_CTC_RNNT" --parakeet_v3 "$PARAKEET_V3" \
  --scorer ../leaderboard-scorer --gpu "$GPU" --device-id 0 \
  --models zipformer_cr_ctc_rnnt parakeet_v3 \
  --batches 256 --precisions fp16 bf16 \
  --output "$RUN_ROOT/b256" --discard-engines
```

Keep the explicit subsets: without them the CLI runs all four models and all nine
batches, including FP32/256.
`--discard-engines` removes each attempt's bundle, retaining logs and hashes.
Existing run directories are refused; retry in a new run root and retain the
old evidence. Never substitute a smaller batch, different precision, or cropped
audio after an OOM/build failure.

Clips are sorted by **descending duration**, with ascending IDs breaking ties,
within each dataset. Final partial batches are included. Five real batches warm
up the runtime before each configuration's measured passes. Timing surrounds the
stream-synchronized full `ASR` call; loading, file I/O, resampling, and scoring are
excluded. RTFx pools **real audio seconds / inference seconds**, never padding or
an average of dataset ratios.

Upstream scoring normalizes both references and hypotheses with compound-word
merging enabled. Earnings22 chunks are reassembled by parent/index first. Keep
every dataset WER, including unfavorable results; mean WER gives each dataset
equal weight. See [timing and scoring](#timing-and-scoring)
for the distinction between this workflow and the published, pinned measurements.

Runs retain hardware/software metadata, hashes, logs, warm-up IDs, transcripts,
word timestamps, per-batch timings, and scores. GPU-process checks are snapshots,
not a reservation: keep the machine otherwise idle. Interrupted or source-invalid
collections are incomplete; only fully collected and scored passes enter reports.

### Report or Rescore

On a CPU reporting host with benchmark dependencies and Chrome/Chromium installed:

```bash
uv run --frozen --extra benchmark python -m benchmarks.report \
  --campaign "$CAMPAIGN" --runs "$RUN_ROOT/b1-b128" "$RUN_ROOT/b256" \
  --output /path/to/benchmark-work/h200-report

# Optional: rescore one complete run, then regenerate the report.
uv run --frozen --extra benchmark python -m benchmarks.score \
  --campaign "$CAMPAIGN" --scorer ../leaderboard-scorer \
  --run-dir "$RUN_ROOT/b1-b128/$GPU-zipformer_cr_ctc_rnnt-fp16-b64"
```

**Write fresh reports outside `docs/benchmarks` and omit `--readme`.** The generic
reporter emits twelve plots plus JSON and Markdown, including unmeasured model
panels; it is not the curated two-model CSV renderer. `--runs` accepts multiple
matrix roots or individual runs. Duplicate complete configurations are rejected,
and measurements sharing a GPU label must have consistent hardware/software.

Reporting verifies saved evidence and scoring output without rerunning WER.
Interrupted/failed attempts are excluded, not treated as measurements. Single-pass
charts have no repeatability interval; multiple-pass charts show the median and
range of complete-pass RTFx. Validation/rendering failures preserve prior reports,
but final writes are not a multi-file transaction.

Keep raw evidence for auditing and rescoring. Do not commit audio, checkpoints,
engines, transcripts, or private manifests. Only the compact CSV, generated
Markdown, and two combined SVGs belong in the repository's benchmark results.
