# Fast GPU ASR

[![CI](https://github.com/SoundsGoodAI/fast-gpu-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/SoundsGoodAI/fast-gpu-asr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fast-gpu-asr)](https://pypi.org/project/fast-gpu-asr/)
[![Python: 3.12-3.14](https://img.shields.io/badge/python-3.12--3.14-blue)](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/pyproject.toml)
[![Platform: Linux x86-64](https://img.shields.io/badge/platform-Linux_x86--64-526078)](#quick-start)
[![Typing: typed](https://img.shields.io/badge/typing-typed-3b82f6)](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/src/fast_gpu_asr/py.typed)
[![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-30203d?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/LICENSE)

**Zipformer and Parakeet. Built for speed.**

Batched offline speech recognition for NVIDIA GPUs. Raw audio in, text and word
timestamps out, through one Python API. TensorRT engines, native CUDA plugins,
and GPU beam search handle inference; no NeMo or k2 installation is needed.

<!-- benchmark-results:start -->

## Batched speech recognition at up to 25,000 RTFx on B300

<table>
  <tbody>
    <tr>
      <th width="50%"><div align="center"><big>Zipformer CR-CTC Transducer beam 6</big></div></th>
      <th width="50%"><div align="center"><big>Parakeet V3 TDT beam 6</big></div></th>
    </tr>
    <tr>
      <td width="50%"><a href="https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/zipformer-fp16-bf16-fp32.svg"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/zipformer-fp16-bf16-fp32.svg" width="100%" alt="Zipformer CR-CTC Transducer, FP16 / BF16 / FP32" /></a></td>
      <td width="50%"><a href="https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/parakeet-fp16-bf16-fp32.svg"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/parakeet-fp16-bf16-fp32.svg" width="100%" alt="Parakeet V3 TDT, FP16 / BF16 / FP32" /></a></td>
    </tr>
  </tbody>
</table>

**RTFx** = total audio duration / total inference time: throughput, not request latency.
Each configuration processes **157.8 hours of audio across seven English datasets**.

**FP16, decoder beam 6:**

| GPU | Model | Batch&nbsp;1<br>RTFx | Batch&nbsp;256<br>RTFx | Batch&nbsp;1<br>Suite&nbsp;time | Batch&nbsp;256<br>Suite&nbsp;time | Batch&nbsp;1<br>Mean&nbsp;WER | Batch&nbsp;256<br>Mean&nbsp;WER |
|---|---|---:|---:|---:|---:|---:|---:|
| A100 | [Zipformer CR-CTC Transducer](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#a100) | 589.0 | 10,298.5 | 964.35 s | 55.15 s | 5.254% | 5.259% |
| A100 | [Parakeet V3 TDT](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#a100) | 560.8 | 6,482.0 | 1012.79 s | 87.62 s | 4.824% | 4.816% |
| H200 | [Zipformer CR-CTC Transducer](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#h200) | 578.2 | 18,100.2 | 982.29 s | 31.38 s | 5.254% | 5.260% |
| H200 | [Parakeet V3 TDT](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#h200) | 593.6 | 12,352.7 | 956.80 s | 45.98 s | 4.803% | 4.804% |
| B300 | [Zipformer CR-CTC Transducer](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#b300) | 879.8 | 25,108.6 | 645.57 s | 22.62 s | 5.257% | 5.261% |
| B300 | [Parakeet V3 TDT](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md#b300) | 897.5 | 19,398.7 | 632.82 s | 29.28 s | 4.814% | 4.810% |

We reproduced the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) English evaluation with fast-gpu-asr, using its datasets and WER scorer and reporting both **RTFx and WER**. [Methodology](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/methodology.md#timing-and-scoring).

### Key Observations

- **Batching matters.** On B300 with FP16, batch 256 versus batch 1 delivers 28.5x for Zipformer CR-CTC Transducer and 21.6x for Parakeet V3 TDT.
- **Gains taper.** On that same GPU with FP16, doubling capacity from 128 to 256 changes throughput by +25.1% for Zipformer CR-CTC Transducer and +23.0% for Parakeet V3 TDT.
- **BF16 is supported too.** At B300/batch 256, BF16 throughput is slightly higher for Zipformer CR-CTC Transducer and Parakeet V3 TDT than FP16. WER is recorded for every configuration; precision can marginally change outputs.
- **FP16 vs. FP32.** At B300/batch 128, FP16 changes throughput by +22.2% for Zipformer CR-CTC Transducer and +58.5% for Parakeet V3 TDT relative to FP32.
- **Mean-WER is consistent across precisions and batches.** On B300, the recorded mean-WER span across all measured precisions and batches is 0.014 percentage points for Zipformer CR-CTC Transducer and 0.034 percentage points for Parakeet V3 TDT.

[All results](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md) | [Download CSV](https://raw.githubusercontent.com/SoundsGoodAI/fast-gpu-asr/main/docs/benchmarks/measurements.csv)

<!-- benchmark-results:end -->

## Quick Start

**Requirements:** Linux x86-64, Python 3.12-3.14, a Turing (SM75) or newer NVIDIA GPU, and
[NVIDIA driver 580 or newer](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Driver 580 is the CUDA 13 family minimum; PTX JIT or newer CUDA features may
require a newer driver. The package uses CUDA 13 and TensorRT.
The wheel includes all nine TensorRT plugin libraries; no local compilation or
TensorRT development headers are needed.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fast-gpu-asr
```

Use a fresh environment and install CPU-only PyTorch first. TensorRT and CuPy
provide GPU inference; PyTorch does not need CUDA support. Package metadata cannot
select PyTorch's CPU index for pip automatically.

**Supported families: Parakeet TDT/CTC and Zipformer Transducer/CTC.**
You can export compatible checkpoints and fine-tuned models, not just the models
shown below. Checkpoints must match the exporters' supported architectures and
checkpoint/configuration formats.

The examples export **FP16, batch size 8**, with a **0.1 / 8 / 40-second**
minimum/typical/maximum audio duration profile. Transducer examples use **beam 6**;
CTC uses **greedy decoding**. Checkpoints are downloaded from each
model's `main` branch and are not bundled with the package.

Build engines **on the GPU you will use**, with the same TensorRT/plugin stack
as inference. **Export deletes and recreates its output directory.** Keep
checkpoints and unrelated files outside it. Building engines takes time and
requires additional host/GPU memory for tactic selection.

### Zipformer CR-CTC Transducer

[Zipformer CR-CTC Transducer XL 290M checkpoint.](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M)

```bash
mkdir -p checkpoints/zipformer
base=https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M/resolve/main
for file in model.pt config.yaml bpe.model; do
  curl --fail --location "$base/$file" --output "checkpoints/zipformer/$file"
done

fast-gpu-asr-export-zipformer \
  --model-path checkpoints/zipformer/model.pt --output-dir exported/zipformer \
  --batch-size 8 --decoder-type transducer_modified_beam_search --beam 6 \
  --encoder-precision fp16 --decoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

### Zipformer CTC

Export the same CR-CTC checkpoint's CTC head instead of its transducer head:

```bash
fast-gpu-asr-export-zipformer \
  --model-path checkpoints/zipformer/model.pt --output-dir exported/zipformer-ctc \
  --batch-size 8 --decoder-type ctc_greedy_search --beam 1 \
  --encoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

CTC requires a checkpoint with a CTC head. It needs no transducer decoder engine
or predictor table, so `--decoder-precision` is omitted. The benchmark results
above use transducer decoding, not CTC.

### Parakeet V3

[Parakeet TDT 0.6B V3 checkpoint.](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

```bash
mkdir -p checkpoints/parakeet
base=https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main
curl --fail --location "$base/parakeet-tdt-0.6b-v3.nemo" \
  --output checkpoints/parakeet/parakeet-tdt-0.6b-v3.nemo

fast-gpu-asr-export-parakeet \
  --model-path checkpoints/parakeet/parakeet-tdt-0.6b-v3.nemo \
  --output-dir exported/parakeet --batch-size 8 \
  --decoder-type transducer_modified_beam_search --beam 6 \
  --encoder-precision fp16 --decoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

### Parakeet CTC

Both [0.6B](https://huggingface.co/nvidia/parakeet-ctc-0.6b) and
[1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b) are supported. Choose either
checkpoint by setting `model` below:

```bash
model=parakeet-ctc-0.6b  # Or parakeet-ctc-1.1b.
mkdir -p checkpoints/parakeet
curl --fail --location \
  "https://huggingface.co/nvidia/$model/resolve/main/$model.nemo" \
  --output "checkpoints/parakeet/$model.nemo"

fast-gpu-asr-export-parakeet \
  --model-path "checkpoints/parakeet/$model.nemo" \
  --output-dir "exported/$model" --batch-size 8 \
  --decoder-type ctc_greedy_search --beam 1 --encoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

CTC uses one encoder engine with an integrated CTC head and GPU greedy decoding.
These checkpoints are separate from the TDT models used in the benchmarks above.

### Transcribe

Use any exported bundle with the same API. This example reads two mono PCM16
WAV files at 16 kHz, each no longer than 40 seconds:

```python
import wave

import numpy as np

from fast_gpu_asr import ASR

paths = ["sample-1.wav", "sample-2.wav"]
audios = []
for path in paths:
    with wave.open(path) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if (channels, sample_width, sample_rate) != (1, 2, 16000):
            raise ValueError(f"Expected a mono PCM16 WAV at 16 kHz: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    audios.append(samples.astype(np.float32) / 32768.0)

# Or use "exported/zipformer-ctc" or "exported/parakeet".
asr = ASR("exported/zipformer")
texts, timestamps = asr(audios)
for path, text, words in zip(paths, texts, timestamps, strict=True):
    print(f"{path}: {text}")
    for word, start, end in words:
        print(f"  {start:.2f}-{end:.2f}: {word}")
```

Pass one or more nonempty 1D NumPy `float32` waveforms as a list, normalized to
`[-1, 1]` at the bundle's sample rate. **Partial batches work**; short clips are
padded internally. Stay within the exported batch and duration limits; resample
or split audio beforehand.

`ASR` returns `(texts, word_timestamps)` with one transcript and one list of
`(word, start, end)` tuples per clip. Times are in seconds, rounded to milliseconds.
They are encoder-frame estimates, not forced alignments: word intervals extend to the
next word boundary or the clip's duration. Select a GPU with `ASR(..., device_id=0)`.

Direct encoder/decoder use requires a `cp.cuda.Device`, a CUDA stream, and
serialized calls. Encoder outputs reuse GPU buffers: use the same stream or
synchronize across streams, and copy outputs you need to keep after the next call.

## Models and Inference Precision

| Family | Example checkpoints | Decoder modes |
|---|---|---|
| Zipformer Transducer | [CR-CTC Transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M), [Transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-transducer-XL-290M) | Modified beam search / beam one |
| Zipformer CTC | [CR-CTC Transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M) | CTC greedy |
| Parakeet TDT | [V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Modified beam search / beam one |
| Parakeet CTC | [0.6B](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CTC greedy |

**FP32, FP16, and BF16** are supported. BF16 requires Ampere or newer.
Published measurements cover A100, H200, and B300 with Zipformer
CR-CTC Transducer and Parakeet V3 TDT, using beam 6 and English audio.

This is an **offline batch inference library**, not a streaming engine.
Engines have fixed batch capacity and dynamic audio length
within their exported profiles. Choose a smaller batch size or separate duration
profiles when memory or latency matters more than aggregate throughput.

## Implementation Details

### Zipformer Encoder and Parakeet FastConformer

A single TensorRT engine turns raw waveforms into decoder-ready outputs.
Features and activations stay on the GPU, and the decoder consumes the outputs
directly, without a round trip through CPU memory.

- **GPU-native audio frontend.** Batched cuFFT transforms, cuBLAS mel projection,
  and custom framing, windowing, and normalization kernels keep feature extraction
  inside the encoder. There is no CPU feature-extraction stage or feature upload.
- **Fused, layout-aware kernels.** Nine native TensorRT plugin libraries across
  both model families combine operations such as relative-position alignment, masking,
  and softmax, or depthwise convolution and activation. Attention reads projection
  layouts directly where possible, avoiding explicit transpose buffers; dedicated
  Zipformer kernels handle resampling and output assembly.
- **Optimizations at export time.** Parakeet combines query, key, and value into
  one projection, folds evaluation-mode BatchNorm into convolution weights and
  biases, and absorbs constant feed-forward scaling into its output projections.
  This reduces projection launches and removes separate BatchNorm and scaling
  operations at inference.
- **Tuned for the target GPU.** TensorRT selects implementations for the exported
  batch size, duration profile, and precision, including eligible CUDA and cuBLAS
  tactics exposed by the plugins. Tactic selection happens during engine building.
- **Reuse memory, replay execution.** Pinned host staging, asynchronous transfers,
  and reusable device buffers and workspaces reduce setup work. Consecutive batches
  with the same shape can use CUDA graph replay to reduce kernel-launch overhead.

### Beam Search Decoder

Both transducer decoders keep scores, candidate selection, and token histories
on the GPU. Backpointers avoid copying histories; reusable buffers and CUDA graphs
reduce allocation and launch overhead. Final selection uses length-normalized
log probability.

#### Zipformer RNN-T

Like k2/Icefall's Zipformer reference
[`modified_beam_search`](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py),
our search emits at most one nonblank token per encoder frame. Key differences:

- **Merge first, then prune.** k2/Icefall selects the top `beam` candidates before
  merging duplicate histories. We combine their scores with log-sum-exp first,
  then retain the top `beam` **unique hypotheses**. Duplicates cannot waste beam
  slots, and combined scores can rescue histories that early pruning would lose.
- **Precompute the predictor.** GPU table lookups replace per-frame stateless
  predictor evaluation; only the joiner runs at each step. Export precomputes
  projected outputs for contexts of up to two tokens, trading memory for less work.
- **Specialize the search.** CUDA kernels tailored to beam, vocabulary, and
  context size replace Python lists and ragged bookkeeping. Prefix checks find
  possible duplicates; exact history comparisons guard against hash collisions.

#### Parakeet TDT

Like NeMo's batched
[`ModifiedALSDBatchedTDTComputer`](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/submodules/tdt_malsd_batched_computer.py)
implementation, our decoder supports batched hypotheses, GPU-resident LSTM
states, reusable buffers, and CUDA graphs. Key differences:

- **Merge before pruning.** NeMo selects the top `beam` candidates, then
  recombines survivors. We merge candidates first, checking exact token histories,
  encoder positions, and zero-duration emission counts, then retain the top
  `beam` unique states.
- **Search duration alternatives.** NeMo pairs each parent's top token candidates
  with its most probable duration, forcing zero-duration blanks to advance. We expand
  nonblank tokens across the **full vocabulary and all configured durations**,
  plus blanks over every positive duration, before final pruning. Exact-history
  grouping avoids materializing the full expansion; token shortlists are reused
  only where they cannot discard a merged winner.
- **Advance differently at the symbol limit.** NeMo forces a blank when its
  emission cap is reached. Our `max_symbols_per_timestep` advances one frame
  after the capped zero-duration token without adding a blank transition.
- **TensorRT and custom CUDA kernels.** Our predictor/joiner runs in TensorRT;
  CuPy-launched kernels handle search and state routing. NeMo uses PyTorch and
  supports a full conditional CUDA-graph loop; ours replays fixed search chunks
  with periodic host completion checks.

For both models, these search changes can affect transcripts; they do not
guarantee better WER or identical upstream results. These are implementation
comparisons, not matched speedup measurements against upstream decoders.

### Decoder Settings

`transducer_modified_beam_search` uses the exported beam width.
`transducer_greedy_search` performs greedy, single-hypothesis decoding by forcing
`beam=1`, keeping only the best hypothesis at each step. It reuses the modified
beam-search implementation rather than a separate decoder.
`ctc_greedy_search` requires a CTC head and uses neither a
transducer decoder engine nor a predictor table.

Both exporters accept `--blank-penalty` (default `0.0`), subtracted from blank log
probabilities after normalization. Positive values discourage blanks in every
mode, including beam one. k2/Icefall's reference applies its penalty before softmax, so equal
nonzero settings are not equivalent. Mode, beam, and penalty are saved in
`model_config.yaml`.

### Export and Runtime Notes

`--encoder-precision` and `--decoder-precision` default to `fp32`. Waveform
frontends and CTC heads remain FP32 in both models. FP32 permits TF32 and eligible
reduced-math plugin tactics; check WER alongside throughput when changing precision.

- **Zipformer precision.** The final encoder projection and transducer log
  probabilities remain FP32. BF16 exports use FP16 for the first subsampling
  convolution.
- **Parakeet precision.** When input scaling is enabled,
  FP16 exports rescale the first block's output weights, biases, and LayerNorm
  epsilons to keep residuals in range while retaining FP16 subsampling and
  Conformer layers. FP32/BF16 exports instead fold the scale into the subsampling
  projection's weights and bias.
- **Export.** Build engines for the target GPU and matching TensorRT/plugin versions.
  Use `--debug` to keep intermediate ONNX files; successful exports otherwise
  remove them.
- **Runtime.** Each `ASR` instance reuses buffers and serializes calls with a lock.
  Model-bundle validation is enabled by default (`validate=True`).

## Build From Source

Provide a CUDA-compatible C++20 host compiler and TensorRT development headers
(`NvInfer.h`) matching the runtime in `uv.lock`. If the headers are outside the
compiler's search paths, set `CPLUS_INCLUDE_PATH` to their directory. From the
repository root:

```bash
uv sync --frozen --extra dev
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
```

`uv` installs `nvcc`, CUDA headers/libraries, TensorRT, and CPU-only PyTorch.
PyTorch handles ONNX export. Prefix exporter commands with `uv run --frozen` when
using the checkout.
For TensorRT headers, [CI](https://github.com/SoundsGoodAI/fast-gpu-asr/actions/workflows/ci.yml)
uses GitHub releases, with a checksum-verified NVIDIA fallback when the tag is unavailable.

**Native targets:** `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89`, `sm_90`,
`sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121`, plus a `compute_80` PTX
fallback. Verify execution and memory requirements on your GPU.

## Tests and Packaging

After the source-build setup above, run from the repository root:

```bash
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python src/fast_gpu_asr/decoder/lint_gpu_kernels.py --check
```

GPU tests skip when CUDA is unavailable; BF16 tests require SM80 (Ampere) or newer.
Tests check decoder behavior and numerical tolerances, not identical transcripts
across precisions. Formatting limits are 88 columns for Python and 100 for CUDA/C++.

**Build a wheel:** `scripts/build_wheel.sh` rebuilds plugins and produces a repaired
`manylinux_2_27_x86_64` wheel in `dist/`. Pass an output directory to override it.
This also requires `binutils` and `patchelf` in addition to the source-build
prerequisites. CUDA and TensorRT remain external dependencies. Source
distributions are unsupported.

**CI:** The [workflow](https://github.com/SoundsGoodAI/fast-gpu-asr/actions/workflows/ci.yml)
runs Python 3.12-3.14 CPU checks on pushes to `main` and pull requests. Manual
dispatch with `run_gpu_tests=true` adds native tests and wheel smoke tests on the
separately billed `gpu-t4` runner; SM80-only cases skip there.

## Reproduce the Measurements

[Complete results and CSV](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/results.md) |
[Hardware, protocol, and plot reproduction](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/methodology.md) |
[Fresh collection workflow](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/docs/benchmarks/methodology.md#collect-new-measurements)

The CSV regenerates the published tables and plots on a CPU-only host. Collecting
new measurements requires the datasets, checkpoints, and a target GPU. The
published campaign pins its source and scorer revisions.

### Rebuild Published Plots

From the repository root:

```bash
uv sync --frozen --extra benchmark
uv run --frozen --extra benchmark plotly_get_chrome -y
uv run --frozen --extra benchmark python docs/benchmarks/render.py
```

Skip browser installation when Chrome/Chromium is already available, or set
`BROWSER_PATH` to its executable. This regenerates two SVGs, `results.md`, and
only the marked performance block in this README from `measurements.csv`.

## License and Acknowledgments

Code is [Apache-2.0](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/LICENSE). **Model weights and datasets retain their own
licenses**; the code license does not grant commercial rights to noncommercial
checkpoints. Built on the work of [k2](https://github.com/k2-fsa),
[NeMo](https://github.com/NVIDIA/NeMo), and their contributors, with NVIDIA
TensorRT and CUDA libraries. See [NOTICE](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/NOTICE) for component attribution.
