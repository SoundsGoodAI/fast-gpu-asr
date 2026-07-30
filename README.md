# Fast GPU ASR

**Fast GPU ASR** is a batched TensorRT runtime and exporter for offline
Zipformer transducers and NVIDIA Parakeet TDT models.

The initial model targets are:

- `soundsgoodai/Zipformer-transducer-XL-290M`
- `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M`
- `nvidia/parakeet-tdt-0.6b-v2`
- `nvidia/parakeet-tdt-0.6b-v3`

The runtime is designed for throughput-oriented GPU inference:

- fixed utterance batches with dynamic audio duration;
- waveform feature extraction and acoustic encoding in one TensorRT engine;
- encoder embeddings retained on the GPU;
- a second fixed-capacity TensorRT engine for prediction-network and joiner
  inference;
- batched Zipformer modified beam search;
- batched transducer and CTC greedy search;
- batched Parakeet TDT greedy and modified beam search with recurrent states
  retained on the GPU;
- reusable CuPy device buffers and pinned host buffers;
- no Icefall or NeMo dependency at inference time.

The current production runtime targets high-throughput deployment. On an
NVIDIA H100, the batch-64 Zipformer greedy, CTC, and beam-6 transducer paths
exceed the initial 6,100-6,200 RTFx target for the 15-second optimization
profile. Batched Parakeet v2 and v3 inference also exceeds 7,000 aggregate
RTFx across the seven public cleaned Open ASR Leaderboard datasets.

## Installation

From a repository checkout, install the runtime for the CUDA major version
available on the deployment machine:

```bash
uv sync --extra cuda12
# or
uv sync --extra cuda13
```

Choose exactly one CUDA extra. CuPy and TensorRT wheels are tied to a CUDA
major version, so `cuda12` and `cuda13` should not be installed together. The
project resolves Torch from the PyTorch CPU wheel index when installed with
`uv`; GPU execution is provided by the selected CuPy and TensorRT wheels.

Install ONNX dependencies for model export:

```bash
uv sync --extra cuda12 --extra export
# or
uv sync --extra cuda13 --extra export
```

TensorRT engines are specific to the TensorRT version and target GPU. Build
them on a compatible NVIDIA machine.

## Runtime

Both model families accept normalized mono float32 audio sampled at the model
sample rate stored in the bundle configuration. The current Zipformer and
Parakeet targets use 16 kHz audio. One `ASR` instance owns reusable TensorRT
contexts and is intended to process batches serially.

```python
import numpy as np

from fast_gpu_asr import ASR

model = ASR(
    "/path/to/exported/model",
    blank_penalty=0.0,
)
audios = [
    np.zeros(16000, dtype=np.float32),
    np.zeros(24000, dtype=np.float32),
]
transcripts = model(audios)
```

The number of waveforms can be smaller than the engine batch size, but cannot
exceed `model.batch_size`. Audio duration must remain inside the TensorRT
profile used during export.

Zipformer supports `transducer_greedy_search`,
`transducer_modified_beam_search`, and `ctc_greedy_search`. Parakeet supports
`transducer_greedy_search` and `transducer_modified_beam_search`. The selected
decoder type is stored in the bundle's
`model_config.yaml`, together with production-style runtime metadata such as
the model sample rate, vocabulary size, beam, blank penalty, and
model-specific decoder parameters.

Zipformer transducer greedy search uses modified beam search with `beam=1`.
Parakeet `transducer_greedy_search` also forces `beam=1` at export time and
uses a dedicated GPU TDT label-looping kernel internally. Parakeet modified
beam search with `beam=1` uses the same greedy runtime path.

## Zipformer Export

The Zipformer exporter loads `model.pt`, `config.yaml`, and `bpe.model` from
one directory. Its encoder engine combines a Kaldi-compatible filterbank
frontend with the condensed Zipformer encoder. Greedy transducer bundles use
one modified-beam hypothesis per utterance, larger-beam bundles reserve
`batch_size * beam` slots, and CTC bundles do not contain a decoder engine.

```bash
uv run fast-gpu-asr-export-zipformer \
  --model-path /path/to/Zipformer-cr-ctc-transducer-XL-290M/model.pt \
  --output-dir exported/zipformer-cr-ctc-xl \
  --batch-size 64 \
  --decoder-type transducer_modified_beam_search \
  --beam 6 \
  --min-audio-seconds 0.5 \
  --opt-audio-seconds 15 \
  --max-audio-seconds 120
```

## Parakeet Export

The Parakeet exporter reads the original `.nemo` archive and reconstructs the
feature extractor, FastConformer encoder, TDT prediction network, and joiner
without importing NeMo.

```bash
uv run fast-gpu-asr-export-parakeet \
  --model-path /path/to/parakeet-tdt-0.6b-v3.nemo \
  --output-dir exported/parakeet-tdt-0.6b-v3 \
  --batch-size 128 \
  --decoder-type transducer_modified_beam_search \
  --beam 6 \
  --min-audio-seconds 0.5 \
  --opt-audio-seconds 15 \
  --max-audio-seconds 19
```

For mixed-duration datasets, pair the high-throughput batch-128 engine with a
smaller-batch engine for the long tail. For example, batch 32 with a 40-second
maximum covers the public short-form leaderboard clips without forcing every
short utterance through a long-duration profile.

Use `--decoder-type transducer_greedy_search` for Parakeet greedy bundles; the
exporter sets `--beam` to 1 automatically. Use
`--decoder-type transducer_modified_beam_search` for larger search beams.

## Benchmark

Benchmark one mono PCM16 WAV repeated across the exported engine batch:

```bash
uv run fast-gpu-asr-benchmark \
  --model-dir exported/zipformer-cr-ctc-xl \
  --wav sample-16khz-mono-pcm16.wav \
  --device cuda:0 \
  --warmups 3 \
  --runs 10
```

The command reports median encoder, decoder, postprocessing, total time, RTFx,
and CuPy memory-pool usage. Use `--batch-size` to benchmark a partial
engine batch.

### H100 Results

The following end-to-end measurements use an NVIDIA H100 80GB HBM3,
TensorRT 11.1.0.106, batch size 64, three warmups, and 20 measured runs.
Each batch repeats a real Earnings21 waveform trimmed to the stated duration.

| Decoder | 5 s | 15 s | 30 s |
| --- | ---: | ---: | ---: |
| Zipformer transducer greedy | 8,599 RTFx | 8,523 RTFx | 6,352 RTFx |
| Zipformer CR-CTC greedy | 10,135 RTFx | 9,848 RTFx | 7,109 RTFx |
| Zipformer transducer beam 6 | 7,244 RTFx | 7,550 RTFx | 4,716 RTFx |
| Zipformer CR-CTC transducer beam 6 | 7,275 RTFx | 7,542 RTFx | 4,651 RTFx |

Beam search keeps candidate scoring, selection, hypothesis merging, and state
updates on the GPU. Its token output exactly matches the previous host-side
search on the tested real-audio batch. The published leaderboard WERs use
modified beam search, while the faster greedy and CTC modes require separate
accuracy evaluation before making Pareto-front claims.

### Parakeet H100 Results

The Parakeet measurements use a batch-128 engine optimized for 15-second
audio with a 19-second maximum, plus a batch-32 fallback engine with a
40-second maximum. The seven cleaned public datasets contain 580,586 seconds
of audio. Timing covers input batching, host-to-device transfer, feature
extraction, FastConformer encoding, TDT greedy decoding, and tokenization; it
does not include WAV loading or WER report generation.

| Model | Runtime | Audio | RTFx | Leaderboard RTFx | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parakeet TDT 0.6B v2 | 81.28 s | 580,586 s | 7,143 | 6,038.1 | +18.3% |
| Parakeet TDT 0.6B v3 | 82.05 s | 580,586 s | 7,076 | 6,098.2 | +16.0% |

The exported v2 model reproduces the current leaderboard WERs within
0.08 absolute WER on every public cleaned dataset:

| Dataset | v2 leaderboard | v2 TensorRT | v3 leaderboard | v3 TensorRT |
| --- | ---: | ---: | ---: | ---: |
| AMI-Cleaned | 9.10 | 9.02 | 9.41 | 9.31 |
| Earnings22 | 10.78 | 10.71 | 10.77 | 9.87 |
| GigaSpeech-Cleaned | 8.15 | 8.18 | 8.00 | 8.00 |
| LibriSpeech clean | 1.27 | 1.29 | 1.51 | 1.51 |
| LibriSpeech other | 2.73 | 2.70 | 3.12 | 3.13 |
| SPGISpeech | 1.94 | 1.91 | 3.63 | 3.63 |
| VoxPopuli-AA-Cleaned | 3.78 | 3.85 | 3.19 | 3.07 |

Parakeet v3 is exported from the original `.nemo` archive. Production bundles
can use the TDT greedy or modified beam-search runtime.

## Development

```bash
uv sync --extra cuda12 --extra export --extra dev
uv run pytest
uv run ruff check .
```

Model checkpoints retain the licenses stated in their source repositories.
Fast GPU ASR runtime and export code is Apache-2.0.
