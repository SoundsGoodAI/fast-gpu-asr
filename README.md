# Fast GPU ASR

*Zipformer and Parakeet. Built for throughput.*

Batched offline speech recognition with TensorRT: raw audio in, text and
word timestamps out. Features and acoustic encoding share an engine; search
runs on the GPU. No Icefall or NeMo installation is needed at inference time.

## Benchmarks

**H100:** two models, batches **1, 2, 4, 8, 16, 32, 64, and 128**,
three precisions, and **one measured pass per attempt**.
**Batch 256** is also measured for **FP16 and BF16**: **52 configurations** and
**56 complete measurements**, including four independent FP16 repeats.
One GPU and one inference process per measurement. A100, H200, B200, and B300
measurements are planned; H100 FP32 measurements stop at batch 128.

- **Zipformer CR-CTC XL 290M**: transducer modified beam search, beam **6**.
- **Parakeet TDT 0.6B V3**: transducer modified beam search, beam **6**.

<!-- benchmark-results:start -->

| Zipformer (beam 6) | Parakeet V3 (beam 6) |
|---|---|
| ![Zipformer fp32](docs/benchmarks/zipformer-fp32.svg) | ![Parakeet fp32](docs/benchmarks/parakeet-fp32.svg) |
| ![Zipformer fp16](docs/benchmarks/zipformer-fp16.svg) | ![Parakeet fp16](docs/benchmarks/parakeet-fp16.svg) |
| ![Zipformer bf16](docs/benchmarks/zipformer-bf16.svg) | ![Parakeet bf16](docs/benchmarks/parakeet-bf16.svg) |

Fastest observed configuration per model (selected by RTFx, not WER):

| Model | Batch | Precision | Best RTFx | Suite time | Mean WER |
|---|---:|---|---:|---:|---:|
| [zipformer](docs/benchmarks/results.md) | 256 | FP16 | 20,242.2 | 28.06 s | 5.244% |
| [parakeet](docs/benchmarks/results.md) | 256 | FP16 | 13,561.8 | 41.88 s | 4.879% |

**Best observed, not median:** each configuration selects its fastest complete single-pass attempt. FP16 batches 128 and 256 have two attempts per model; other configurations have one. WER and suite time come from the same selected run. All 56 measurements across 52 configurations are retained locally, including slower repeats.

Measurements used NVIDIA H100 80GB HBM3 GPUs.

[Complete selected table](docs/benchmarks/results.md) | [Methodology and reproduction](docs/benchmarks/methodology.md)

<!-- benchmark-results:end -->

These are pre-release H100 measurements from isolated, frozen source
snapshots, not the completed multi-GPU matrix. Repeat the measurements from a
committed release snapshot before publishing release comparisons.

Each plotted point selects **one complete pass** over seven public English
datasets; repeated configurations select their fastest observed attempt. RTFx
counts real audio seconds, not padding, and times the synchronized **full ASR
call**, including transfers, text, and timestamps. Loading and file I/O are excluded.

Precision labels describe the requested encoder and decoder precision, not
strict arithmetic throughout the pipeline. **FP32 allows TF32 and eligible
reduced-math tactics**; some frontend/output operations retain their configured
precision. WER is measured separately for every configuration and pass.

[Current collection workflow](benchmarks/README.md)

## Quick Start

Linux x86-64, Python 3.12-3.14, a CUDA 13-compatible NVIDIA driver, and TensorRT
`>=11.2.1.2,<12` are required. CUDA is mandatory for inference.

The native plugins target `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89`,
`sm_90`, `sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121`, with a `compute_80`
PTX fallback. T4 (`sm_75`) targets FP32/FP16; BF16 requires Ampere or newer.
Native T4 code is cross-compiled; verify execution and performance on your target
hardware. Compilation support alone does not establish performance on every GPU.

Install the published wheel, which includes all nine native TensorRT plugins.
No local plugin compilation or TensorRT development headers are needed:

```bash
python -m pip install fast-gpu-asr
```

For a source checkout instead, provide a CUDA-compatible host compiler with
C++20 support and TensorRT development headers, including `NvInfer.h`, matching
the locked runtime. Then build the plugins:

```bash
uv sync --frozen
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
```

Python dependencies provide `nvcc`, CUDA headers, cuBLAS, cuFFT, the CUDA runtime,
and TensorRT bindings/runtime libraries. The plugin builder resolves and links
those wheel-provided libraries. ONNX and ONNXScript are included for export.
The checkout's `uv` configuration selects CPU Torch wheels; PyPI installations
use the installer's configured indexes. CUDA-enabled Torch is not required:
inference uses CuPy, TensorRT, and the native plugins.

After exporting a bundle below, transcribe a mono PCM16 WAV at 16 kHz:

```python
import wave

import numpy as np

from fast_gpu_asr import ASR

with wave.open("sample.wav") as wav:
    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
audio = audio.astype(np.float32) / 32768.0

asr = ASR("exported/zipformer")
texts, timestamps = asr([audio])
print(texts[0])
for word, start, end in timestamps[0]:
    print(f"{start:.2f}-{end:.2f}: {word}")
```

Pass a nonempty list of one-dimensional NumPy FP32 waveforms normalized to
`[-1.0, 1.0]`, at the sample rate in `model_config.yaml` (16 kHz for the listed
models). Each waveform must be nonempty. Partial batches are supported, but the
list cannot exceed `asr.encoder.batch_size` and each recording must fit the
bundle's maximum duration. Short inputs are padded to the minimum execution
profile while retaining their valid lengths.

The runtime returns one transcript and a list of `(word, start, end)` tuples
per input, with times in seconds; the final word ends at the waveform duration.
Use `ASR(..., device_id=0)` to select a GPU. Bundle validation is enabled by
default; use `validate=False` only for an already validated artifact. Each ASR
instance reuses mutable device buffers and serializes calls with an internal
lock; recurring shapes use CUDA graph replay when capture is supported.

For advanced composition, the package also exports `Encoder`, `CTCGreedyDecoder`,
`ZipformerModifiedBeamSearchDecoder`, `ParakeetModifiedBeamSearchDecoder`, and
`PostProcessor`.

## Export

Build engines **on the target GPU**, with the same TensorRT/plugin stack used
for inference. Export directories are deleted and recreated: keep source
checkpoints and unrelated files elsewhere.

After a PyPI installation, run the exporter commands below without the
`uv run --frozen` prefix. Checkpoints and exported engines are not included in
the package.

Supported checkpoint targets include:

- Zipformer [transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-transducer-XL-290M)
  and [CR-CTC/transducer XL 290M](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M).
- Parakeet TDT 0.6B [V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
  and [V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

Other noncausal, six-stack Icefall Zipformer checkpoints must satisfy the
exporter's configuration and checkpoint-layout validation. TensorRT engines
are tied to the GPU architecture, TensorRT version, and plugin binaries used
to build them. Engine construction requires sufficient host and GPU memory
for tactic selection; tune batch capacity and duration profiles for deployment.

### Zipformer

Download `model.pt`, `config.yaml`, and `bpe.model` from
[soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M)
into one directory.

```bash
uv run --frozen fast-gpu-asr-export-zipformer \
  --model-path checkpoints/zipformer/model.pt --output-dir exported/zipformer \
  --batch-size 64 --decoder-type transducer_modified_beam_search --beam 6 \
  --encoder-precision fp16 --decoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

### Parakeet

Download the original `.nemo` archive from
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

```bash
uv run --frozen fast-gpu-asr-export-parakeet \
  --model-path checkpoints/parakeet-tdt-0.6b-v3.nemo \
  --output-dir exported/parakeet --batch-size 64 \
  --decoder-type transducer_greedy_search --beam 1 \
  --encoder-precision fp16 --decoder-precision fp16 \
  --min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 \
  --optimization-level 5
```

### Modes and Precision

Both families support `transducer_modified_beam_search` and
`transducer_greedy_search`; greedy uses the same implementation with `beam=1`,
overriding any supplied beam. Zipformer additionally supports `ctc_greedy_search`
when its checkpoint includes a CTC head; CTC requires beam one and needs neither
a decoder engine nor a predictor context table. Decoder mode, beam, blank ID,
and blank penalty are saved in `model_config.yaml`; exports default the penalty
to `0.0`.

`--encoder-precision` and `--decoder-precision` accept `fp32`, `fp16`, and `bf16`
and default to `fp32`. Waveform frontends remain FP32. Zipformer's final encoder
projection also remains FP32; runtime search converts those embeddings to the
decoder precision. Its BF16 export uses FP16 for the first subsampling
convolution, then returns to BF16. Transducer token log probabilities, and
Parakeet duration log probabilities, remain FP32. Reduced precision can change
decisions near score ties: recheck WER when changing precision.

Parakeet's maximum duration profile must fit the attention plugin's limit of
512 encoder frames. Smaller batch capacities or separate short/long-duration bundles
can reduce memory pressure. Zipformer transducer bundles also include a
precomputed predictor context table, which occupies GPU memory during inference.
Pass `--debug` to either exporter to retain intermediate ONNX artifacts;
otherwise they are removed after successful build and validation.

## Reproduce

The [full-corpus workflow](https://github.com/SoundsGoodAI/fast-gpu-asr/blob/main/benchmarks/README.md) freezes inputs, builds each
configuration, retains every measured pass, calls the upstream scorer on main, and
generates plots and tables from complete, validated suites. It does not provision
machines or publish releases.

## Development

```bash
uv sync --frozen --extra dev
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
uv run --frozen pytest
uv run --frozen ruff check src tests scripts benchmarks
uv run --frozen ruff format --check src tests scripts benchmarks
uv run --frozen python src/fast_gpu_asr/decoder/lint_gpu_kernels.py --check
```

Ruff formats Python at 88 columns; `.clang-format` applies a 100-column limit to
CUDA/C++ and embedded kernels. Use `ruff format` or the CUDA formatter's `--fix`
to apply formatting. The CUDA formatter itself needs no GPU. Device tests skip
on CPU-only hosts; the complete runtime/plugin suite needs a supported GPU.

Build a platform wheel from a clean checkout with:

```bash
scripts/build_wheel.sh
```

The optional sole argument selects the output directory (default: `dist`).
The script rebuilds all nine plugins and repairs the wheel for
`manylinux_2_27_x86_64`, without absolute plugin `RPATH`/`RUNPATH` entries.
CUDA and TensorRT libraries remain package dependencies, not copies inside the
project wheel. Source distributions are intentionally unsupported.

GitHub-hosted CI runs lint, workflow/lockfile checks, and Python 3.12-3.14 CPU
tests on pushes and pull requests. For GPU validation, manually dispatch
[CI](https://github.com/SoundsGoodAI/fast-gpu-asr/actions/workflows/ci.yml) with
`run_gpu_tests` enabled. The separately billed `gpu-t4` job rebuilds plugins,
tests them, checks linkage, and builds and smoke-tests the installed wheel.
SM80-only tests are expected to skip on T4; other skips fail that job. CI requires
driver 580 or newer and checks that CUDA build components come from Python
packages and TensorRT headers match the locked runtime.

Publishing is disabled by default.

## License

Code: Apache-2.0. Checkpoints and datasets retain their own licenses.
