# Fast GPU ASR

**Fast GPU ASR** exports supported offline Zipformer and NVIDIA Parakeet TDT
checkpoints to fixed-capacity TensorRT bundles and runs batched speech recognition
without Icefall or NeMo in the inference environment.

Validated model targets include:

- [`soundsgoodai/Zipformer-transducer-XL-290M`](https://huggingface.co/soundsgoodai/Zipformer-transducer-XL-290M)
- [`soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M`](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M)
- [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

Other noncausal Icefall Zipformer checkpoints can be exported when their
configuration and checkpoint layout satisfy the exporter's validation rules.

The runtime is designed for throughput-oriented GPU inference:

- fixed batch capacity with a dynamic audio-duration profile;
- waveform feature extraction and acoustic encoding in one TensorRT engine;
- a second TensorRT engine for transducer prediction-network and joiner inference;
- CTC greedy, Zipformer modified-beam, and Parakeet TDT modified-beam decoders;
- decoder state, hypothesis histories, and encoder embeddings retained on the GPU;
- reusable CuPy device buffers and pinned host buffers;
- CUDA graph replay for recurring input shapes when capture is supported;
- text and word-level timestamps returned for every input waveform.

## Requirements

Fast GPU ASR currently supports Linux x86-64, Python 3.12 through 3.14, CUDA 13,
and TensorRT 11.2.1.2 or newer within the TensorRT 11 release family. CUDA is a
mandatory runtime dependency.

The packaged plugins contain native code for `sm_80`, `sm_86`, `sm_87`, `sm_88`,
`sm_89`, `sm_90`, `sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121`, plus a
`compute_80` PTX fallback. A sufficiently recent NVIDIA driver is still required.

Building the native plugins from a repository checkout additionally requires:

- a CUDA-compatible host compiler with C++20 support;
- TensorRT 11 development headers, including `NvInfer.h`;
- enough host and GPU memory for TensorRT tactic selection.

`nvcc`, CUDA headers, cuBLAS, cuFFT, the CUDA runtime, TensorRT Python bindings,
and TensorRT runtime libraries are supplied by the required Python packages. The
plugin build resolves and links those wheel-provided libraries directly.

## Installation

From a repository checkout, create the locked runtime environment and compile the
nine native TensorRT plugins:

```bash
uv sync --frozen
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
```

The project resolves Torch from the PyTorch CPU wheel index when installed with
`uv`. GPU execution is provided by CuPy, TensorRT, and the native CUDA plugins;
CUDA-enabled Torch is not required.

ONNX and ONNXScript are included in the default installation for model export.

Serialized TensorRT engines depend on the TensorRT version, plugin binaries, and
GPU architecture used to build them. Build and validate each bundle on the target
deployment architecture and software stack.

## Runtime

Both model families accept nonempty, one-dimensional NumPy waveforms normalized
to `[-1.0, 1.0]` and sampled at the rate stored in `model_config.yaml`. The
currently validated targets use 16 kHz audio.

```python
import numpy as np

from fast_gpu_asr import ASR

model = ASR("/path/to/exported/model", device_id=0)
audios = [
    np.zeros(16000, dtype=np.float32),
    np.zeros(24000, dtype=np.float32),
]
texts, word_timestamps = model(audios)

print(texts[0])
for word, start, end in word_timestamps[0]:
    print(word, start, end)
```

The input list may contain fewer waveforms than the engine capacity, but it must
contain at least one and cannot exceed `model.encoder.batch_size`. No waveform may
exceed the maximum duration profile stored in the bundle. Inputs shorter than the
minimum profile are padded to its execution shape; their valid lengths remain
separate.

Word timestamps are returned as `(word, start, end)` tuples in seconds. The final
word ends at the input waveform duration. Bundle validation is enabled by default;
`ASR(..., validate=False)` is intended only for an artifact that was already
validated. An `ASR` instance owns mutable TensorRT contexts and serializes calls
with an internal lock.

The top-level package also exposes `Encoder`, `CTCGreedyDecoder`,
`ZipformerModifiedBeamSearchDecoder`, `ParakeetModifiedBeamSearchDecoder`, and
`PostProcessor` for advanced composition.

### Decoder modes

Zipformer supports:

- `ctc_greedy_search`;
- `transducer_greedy_search`;
- `transducer_modified_beam_search`.

Parakeet supports the two transducer modes. For either model family,
`transducer_greedy_search` uses the same modified-beam implementation with
`beam=1`; the exporters override any other beam value for greedy mode. Zipformer
CTC also requires `beam=1`.

The Zipformer modified search permits at most one nonblank symbol per encoder
frame. It scores the complete hypothesis-by-vocabulary table, selects its top
`beam` candidates, and merges identical retained token histories with log-sum-exp.
Parakeet applies the corresponding TDT search over token and duration outputs.
The selected decoder type, beam, blank penalty, blank token ID, and model-specific
dimensions are stored in `model_config.yaml`. Exporters currently initialize the
blank penalty to `0.0`.

## Zipformer Export

The Zipformer exporter expects `model.pt` beside `config.yaml` and `bpe.model`.
It reconstructs the supported six-stack offline encoder and selects either the
checkpoint's transducer projection or CTC head.

> **Warning:** the exporter deletes and recreates `--output-dir`. Never place the
> source checkpoint, configuration, tokenizer, or unrelated files inside it.

```bash
uv run --frozen fast-gpu-asr-export-zipformer \
  --model-path /path/to/Zipformer-cr-ctc-transducer-XL-290M/model.pt \
  --output-dir exported/zipformer-cr-ctc-xl \
  --batch-size 64 \
  --decoder-type transducer_modified_beam_search \
  --beam 6 \
  --encoder-precision fp16 \
  --decoder-precision fp16 \
  --min-audio-seconds 0.1 \
  --opt-audio-seconds 8 \
  --max-audio-seconds 40
```

`--encoder-precision` and `--decoder-precision` accept `fp32`, `fp16`, and
`bf16`; both default to `fp32`. Encoder precision controls subsampling and all six
Zipformer stacks. The waveform frontend and final output projection remain FP32,
so both transducer encoder embeddings and CTC log probabilities are FP32. For
BF16 export, the first subsampling convolution uses FP16 because that TensorRT
path is faster, then returns to BF16.

Decoder precision controls the precomputed stateless-predictor context table and
the joiner. Runtime search kernels convert FP32 encoder embeddings to the decoder
precision while staging each frame. The 512-token, context-size-two FP16 table
used by the validated XL models is approximately 257 MiB. Log-softmax output
remains FP32. Reduced precision can alter decisions near score ties, so compare
WER after changing precision.

The engine uses native plugins for cuFFT feature extraction, convolution,
relative-attention scoring and softmax, attention-value products, temporal
resampling, and final encoder-output assembly. A CTC bundle contains no decoder
engine or predictor context table. Pass `--debug` to retain intermediate ONNX
artifacts; otherwise they are removed after a successful build and validation.

## Parakeet Export

The Parakeet exporter reads the original `.nemo` archive and reconstructs the
feature extractor, FastConformer encoder, TDT prediction network, and joiner
without importing NeMo.

> **Warning:** the exporter deletes and recreates `--output-dir`. Do not put the
> source `.nemo` archive or unrelated files inside it.

```bash
uv run --frozen fast-gpu-asr-export-parakeet \
  --model-path /path/to/parakeet-tdt-0.6b-v3.nemo \
  --output-dir exported/parakeet-tdt-0.6b-v3 \
  --batch-size 64 \
  --decoder-type transducer_greedy_search \
  --beam 1 \
  --encoder-precision fp16 \
  --decoder-precision fp16 \
  --min-audio-seconds 0.1 \
  --opt-audio-seconds 8 \
  --max-audio-seconds 40
```

Both precision arguments accept `fp32`, `fp16`, and `bf16` and default to
`fp32`. Encoder precision controls convolutional subsampling and FastConformer
layers; the waveform frontend remains FP32. Decoder precision controls the
prediction network, recurrent state, and joiner. Token and duration log-softmax
outputs remain FP32.

Parakeet feature extraction, Conformer convolution, and full-context
relative-position attention use native TensorRT plugins. The attention plugin
fuses query preparation, relative alignment, masking, softmax, and value
aggregation while using TensorRT-owned workspace for its score matrices. The
maximum profile is validated against the plugin's 512-frame encoder limit.

For mixed-duration traffic, separate engines tuned for short and long utterances
can be more efficient than one maximum-duration profile. Batch size, profile
durations, precision, and decoder mode all affect memory use and throughput.

## Benchmark

From a repository checkout, benchmark one mono PCM16 WAV repeated across a full
or partial engine batch:

```bash
uv run --frozen scripts/benchmark.py \
  --model-dir exported/zipformer-cr-ctc-xl \
  --wav sample-16khz-mono-pcm16.wav \
  --device-id 0 \
  --batch-size 64 \
  --warmups 3 \
  --runs 10
```

The script logs median encoder, decoder, postprocessing, and independently
measured end-to-end latency, along with RTFx and CuPy memory-pool usage. It
synchronizes the shared CUDA stream around timed GPU work. Postprocessing
includes SentencePiece decoding and word timestamp construction.

RTFx is the batch's total audio duration divided by synchronized end-to-end wall
time. Because component and end-to-end medians come from separate runs, the total
need not equal the sum of component medians. A repeated-waveform benchmark is a
controlled latency measurement, not a substitute for pooled RTFx and WER over a
real dataset. Reportable dataset RTFx should use total audio seconds divided by
total synchronized inference seconds across the complete evaluation set.

## Development

Install all test and export dependencies, build the native plugins, and run the
quality checks:

```bash
uv sync --frozen --extra dev
uv run --frozen python -m fast_gpu_asr.tensorrt_plugins.build
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

CPU-only tests skip device execution when no compatible GPU is available. The
complete plugin and runtime suite requires a supported NVIDIA GPU.

### Building a wheel

Build a publishable platform wheel from a clean repository checkout:

```bash
scripts/build_wheel.sh
```

An optional destination directory may be passed as the only argument. The script
rebuilds all native plugins, creates a Python-ABI-independent Linux wheel, and
repairs it for `manylinux_2_27_x86_64`. CUDA, cuBLAS, cuFFT, and TensorRT remain
required package dependencies rather than being copied into the project wheel.
The packaged plugins contain no absolute `RPATH` or `RUNPATH`. Source
distributions are intentionally unsupported because they cannot provide portable
TensorRT plugin binaries.

### Continuous integration

GitHub-hosted jobs check the lockfile, Actions workflow syntax, lint, formatting,
and the Python 3.12, 3.13, and 3.14 test matrix. A self-hosted Linux x86-64 GPU
runner rebuilds all nine CUDA plugins, runs the complete test suite, inspects
native linkage, builds and repairs the wheel, and smoke-tests the installed wheel.

The GPU runner requires a CUDA 13-capable driver, a C++20 host compiler,
`readelf`, and TensorRT 11 development headers. CI verifies that `nvcc`, CUDA
headers, and CUDA libraries resolve from Python `site-packages`; it rejects plugin
`RPATH` or `RUNPATH` entries and incomplete wheels. Self-hosted GPU jobs run for
pushes to `main`, manual dispatches, and same-repository pull requests. Fork pull
requests run only on GitHub-hosted workers.

## License

Fast GPU ASR runtime and export code is licensed under Apache-2.0. Model
checkpoints retain the licenses stated by their source repositories.
