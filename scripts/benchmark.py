#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Benchmark one packaged TensorRT ASR model with repeated PCM16 audio.

The command repeats one waveform across a full or partial engine batch and
reports synchronized wall-clock timings. Component measurements isolate the
encoder, decoder, and CPU postprocessor; end-to-end latency is measured in a
separate pass through :class:`~fast_gpu_asr.asr.ASR`.
"""

import argparse
import json
import logging
import wave
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np

from fast_gpu_asr import ASR

logger = logging.getLogger(__name__)


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    Parameters
    ----------
    value : str
        Command-line value to parse.

    Returns
    -------
    int
        Parsed positive integer.

    Raises
    ------
    ValueError
        Raised when ``value`` is not an integer.
    argparse.ArgumentTypeError
        Raised when the parsed integer is less than one.
    """

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")

    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a nonnegative command-line integer.

    Parameters
    ----------
    value : str
        Command-line value to parse.

    Returns
    -------
    int
        Parsed nonnegative integer.

    Raises
    ------
    ValueError
        Raised when ``value`` is not an integer.
    argparse.ArgumentTypeError
        Raised when the parsed integer is negative.
    """

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value}")

    return parsed


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed benchmark arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a packaged TensorRT ASR model by repeating one mono "
            "PCM16 WAV across a batch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing the packaged TensorRT model.",
    )
    parser.add_argument(
        "--wav",
        required=True,
        type=Path,
        help="Mono PCM16 WAV used for every utterance in the batch.",
    )
    parser.add_argument(
        "--device-id",
        default=0,
        type=nonnegative_integer,
        help="CUDA device ordinal.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        help="Actual batch size; the engine capacity is used when omitted.",
    )
    parser.add_argument(
        "--warmups",
        default=3,
        type=nonnegative_integer,
        help="Number of end-to-end warmup calls.",
    )
    parser.add_argument(
        "--runs",
        default=10,
        type=positive_integer,
        help="Number of measured component and end-to-end calls.",
    )

    return parser.parse_args()


def read_pcm16(path: Path) -> tuple[np.typing.NDArray[np.float32], int]:
    """Read a mono PCM16 WAV file as normalized float32 audio.

    Parameters
    ----------
    path : Path
        WAV file to read.

    Returns
    -------
    tuple[np.typing.NDArray[np.float32], int]
        Waveform normalized to ``[-1.0, 1.0)`` and its sampling rate in hertz.

    Raises
    ------
    ValueError
        Raised when the WAV container is malformed, the file is not mono
        PCM16, is empty, or contains a truncated or partial PCM frame.
    """

    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Expected a mono PCM16 WAV file.")

            sample_rate = wav_file.getframerate()
            # Request one extra frame so a data chunk ending in one stray byte is
            # returned and rejected by the exact payload-size check below.
            audio_bytes = wav_file.readframes(wav_file.getnframes() + 1)
            expected_bytes = wav_file.getnframes() * 2
            if len(audio_bytes) != expected_bytes:
                raise ValueError(f"Truncated PCM16 WAV file {path}.")

            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio /= 2**15
            if audio.size == 0:
                raise ValueError(f"Expected nonempty PCM16 WAV file {path}.")

    except (EOFError, wave.Error) as error:
        detail = str(error) or type(error).__name__
        raise ValueError(f"Invalid WAV file {path}: {detail}.") from error

    return audio, sample_rate


def main() -> None:
    """Run repeated batched inference and log benchmark measurements.

    The logged JSON object contains median ``encoder_sec``, ``decoder_sec``,
    ``postprocess_sec``, and independently measured ``total_sec`` durations. It
    also reports the batch's ``audio_seconds``, end-to-end ``rtfx``, and
    ``cupy_memory_pool_bytes`` retained by CuPy's default memory pool.

    Notes
    -----
    GPU component intervals end with an explicit stream synchronization, so
    they include asynchronous CUDA execution and required data transfers.
    ``total_sec`` is measured independently and is therefore not expected to
    equal the sum of the three component medians.
    """

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )

    args = parse_args()
    audio, sample_rate = read_pcm16(args.wav)
    model = ASR(args.model_dir, device_id=args.device_id)

    if sample_rate != model.encoder.sample_rate:
        logger.error(
            "Expected %d Hz benchmark audio, got %d Hz",
            model.encoder.sample_rate,
            sample_rate,
        )
        raise SystemExit(2)

    batch_size = (
        model.encoder.batch_size if args.batch_size is None else args.batch_size
    )
    if not 1 <= batch_size <= model.encoder.batch_size:
        logger.error(
            "batch-size must be in [1, %d], got %d",
            model.encoder.batch_size,
            batch_size,
        )
        raise SystemExit(2)

    audios = [audio] * batch_size

    with cp.cuda.Device(args.device_id):
        memory_pool = cp.get_default_memory_pool()
        memory_pool.free_all_blocks()

        for _ in range(args.warmups):
            model(audios)
        model.stream.synchronize()

        component_results: dict[str, list[float]] = {
            "encoder_sec": [],
            "decoder_sec": [],
            "postprocess_sec": [],
        }
        for _ in range(args.runs):
            timer = perf_counter()
            encoder_output, encoder_output_lengths = model.encoder(audios)
            model.stream.synchronize()
            encoder_sec = perf_counter() - timer

            timer = perf_counter()
            token_ids, timestamps = model.decoder(
                encoder_output, encoder_output_lengths
            )
            model.stream.synchronize()
            decoder_sec = perf_counter() - timer

            timer = perf_counter()
            model.postprocessor(audios, token_ids, timestamps)
            postprocess_sec = perf_counter() - timer

            component_results["encoder_sec"].append(encoder_sec)
            component_results["decoder_sec"].append(decoder_sec)
            component_results["postprocess_sec"].append(postprocess_sec)

        total_results = []
        for _ in range(args.runs):
            timer = perf_counter()
            model(audios)
            model.stream.synchronize()
            total_results.append(perf_counter() - timer)

        medians = {
            name: float(np.median(values)) for name, values in component_results.items()
        }
        medians["total_sec"] = float(np.median(total_results))
        medians["audio_seconds"] = audio.size * batch_size / sample_rate
        medians["rtfx"] = medians["audio_seconds"] / medians["total_sec"]
        medians["cupy_memory_pool_bytes"] = memory_pool.total_bytes()

    logger.info("%s", json.dumps(medians, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
