#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the packaged-model benchmark command line."""

import importlib.util
import json
import logging
import runpy
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "fast_gpu_asr_test_benchmark",
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py",
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None

benchmark_module = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(benchmark_module)

parse_args = benchmark_module.parse_args
read_pcm16 = benchmark_module.read_pcm16

BASE_ARGUMENTS = ("benchmark.py", "--model-dir", "model", "--wav", "audio.wav")


def write_wav(
    path: Path,
    samples: np.typing.NDArray[np.int16],
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> None:
    """Write a small PCM WAV fixture.

    Parameters
    ----------
    path : Path
        Destination WAV path.
    samples : np.typing.NDArray[np.int16]
        Interleaved signed PCM16 samples. Other sample widths use zero bytes.
    channels : int
        Number of channels declared in the WAV header.
    sample_width : int
        Number of bytes declared for each sample.
    sample_rate : int
        Sampling rate declared in the WAV header.
    """

    frame_bytes = (
        samples.astype(np.int16, copy=False).tobytes()
        if sample_width == 2
        else bytes(samples.size * sample_width)
    )
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frame_bytes)


def test_read_pcm16_normalizes_mono_waveform(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    write_wav(wav_path, samples, sample_rate=8000)

    audio, sample_rate = read_pcm16(wav_path)

    assert sample_rate == 8000
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, samples.astype(np.float32) / 32768.0)


@pytest.mark.parametrize(
    ("channels", "sample_width"),
    ((2, 2), (1, 1)),
    ids=("stereo", "pcm8"),
)
def test_read_pcm16_rejects_incompatible_wav(
    tmp_path: Path,
    channels: int,
    sample_width: int,
) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(
        wav_path,
        np.zeros(4 * channels, dtype=np.int16),
        channels=channels,
        sample_width=sample_width,
    )

    with pytest.raises(ValueError, match="Expected a mono PCM16 WAV file"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_truncated_payload(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.arange(8, dtype=np.int16))
    wav_path.write_bytes(wav_path.read_bytes()[:-2])

    with pytest.raises(ValueError, match="Truncated PCM16 WAV file"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_partial_pcm_frame(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.array([1], dtype=np.int16))
    wav_bytes = bytearray(wav_path.read_bytes())
    data_offset = wav_bytes.index(b"data")
    # The data chunk has an incomplete sample plus a separate RIFF padding byte.
    struct.pack_into("<I", wav_bytes, data_offset + 4, 3)
    wav_bytes.extend(b"\0\0")
    struct.pack_into("<I", wav_bytes, 4, len(wav_bytes) - 8)
    wav_path.write_bytes(wav_bytes)

    with pytest.raises(ValueError, match="Truncated PCM16 WAV file"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_empty_waveform(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.empty(0, dtype=np.int16))

    with pytest.raises(ValueError, match="Expected nonempty PCM16 WAV file"):
        read_pcm16(wav_path)


@pytest.mark.parametrize("payload", (b"", b"not a wave file"))
def test_read_pcm16_wraps_malformed_wav_errors(tmp_path: Path, payload: bytes) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(payload)

    with pytest.raises(ValueError, match="Invalid WAV file") as error:
        read_pcm16(wav_path)

    assert isinstance(error.value.__cause__, (EOFError, wave.Error))


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", list(BASE_ARGUMENTS))

    assert vars(parse_args()) == {
        "model_dir": Path("model"),
        "wav": Path("audio.wav"),
        "device_id": 0,
        "batch_size": None,
        "warmups": 3,
        "runs": 10,
    }


def test_parse_args_accepts_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            *BASE_ARGUMENTS,
            "--device-id",
            "0",
            "--batch-size",
            "1",
            "--warmups",
            "0",
            "--runs",
            "1",
        ],
    )

    args = parse_args()
    assert (args.device_id, args.batch_size, args.warmups, args.runs) == (0, 1, 0, 1)


@pytest.mark.parametrize(
    ("arguments", "missing"),
    ((("--wav", "audio.wav"), "--model-dir"), (("--model-dir", "model"), "--wav")),
    ids=("model-dir", "wav"),
)
def test_parse_args_requires_model_and_wav(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, str],
    missing: str,
) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark.py", *arguments])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 2
    assert f"the following arguments are required: {missing}" in capsys.readouterr().err


def test_script_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark.py", "--help"])

    with pytest.raises(SystemExit) as error:
        runpy.run_path(benchmark_module.__file__, run_name="__main__")

    assert error.value.code == 0
    assert "--model-dir" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--batch-size", "0", "expected a positive integer"),
        ("--runs", "0", "expected a positive integer"),
        ("--device-id", "-1", "expected a nonnegative integer"),
        ("--warmups", "-1", "expected a nonnegative integer"),
        ("--batch-size", "1.5", "invalid positive_integer value"),
        ("--warmups", "many", "invalid nonnegative_integer value"),
    ),
)
def test_parse_args_rejects_invalid_counts(
    option: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", [*BASE_ARGUMENTS, option, value])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("batch_arguments", "batch_size", "warmups", "sample_rate"),
    (((), 2, 2, 8000), (("--batch-size", "1"), 1, 0, 16000)),
    ids=("engine-batch", "partial-batch"),
)
def test_main_reports_synchronized_median_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    batch_arguments: tuple[str, ...],
    batch_size: int,
    warmups: int,
    sample_rate: int,
) -> None:
    model_path = tmp_path / "model"
    wav_path = tmp_path / "audio.wav"
    samples = np.arange(160, dtype=np.int16)
    write_wav(wav_path, samples, sample_rate=sample_rate)

    clock = [0.0]

    class Stream:
        """Accumulate simulated asynchronous GPU execution time."""

        def __init__(self) -> None:
            self.pending_seconds = 0.0
            self.synchronizations = 0

        def synchronize(self) -> None:
            """Advance the wall clock by all queued GPU work."""

            clock[0] += self.pending_seconds
            self.pending_seconds = 0.0
            self.synchronizations += 1

    stream = Stream()
    encoder_output = np.zeros((batch_size, 1, 2), dtype=np.float32)
    encoder_output_lengths = np.ones(batch_size, dtype=np.int32)
    token_ids = [[1] for _ in range(batch_size)]
    timestamps = [[0.0] for _ in range(batch_size)]
    encoder_durations = iter((1.0, 2.0, 100.0))
    decoder_durations = iter((100.0, 3.0, 2.0))
    postprocess_durations = iter((0.1, 0.2, 10.0))
    total_durations = iter([0.5] * warmups + [4.0, 2.0, 9.0])

    def encode(_audios):
        """Queue encoder work and return fixed device outputs."""

        assert stream.pending_seconds == 0.0
        stream.pending_seconds += next(encoder_durations)
        return encoder_output, encoder_output_lengths

    def decode(_encoder_output, _encoder_output_lengths):
        """Queue decoder work and return fixed token sequences."""

        stream.pending_seconds += next(decoder_durations)
        return token_ids, timestamps

    def postprocess(_audios, _token_ids, _timestamps):
        """Advance CPU time and return placeholder transcriptions."""

        clock[0] += next(postprocess_durations)
        return ["word"] * batch_size, [[] for _ in range(batch_size)]

    def transcribe(_audios):
        """Queue one end-to-end model invocation."""

        stream.pending_seconds += next(total_durations)
        return ["word"] * batch_size, [[] for _ in range(batch_size)]

    encoder = Mock(side_effect=encode)
    encoder.batch_size = 2
    encoder.sample_rate = sample_rate
    decoder = Mock(side_effect=decode)
    postprocessor = Mock(side_effect=postprocess)
    model = Mock(side_effect=transcribe)
    model.encoder = encoder
    model.decoder = decoder
    model.postprocessor = postprocessor
    model.stream = stream
    model_factory = Mock(return_value=model)
    memory_pool = Mock()
    memory_pool.total_bytes.return_value = 4096
    device = MagicMock()
    device_factory = Mock(return_value=device)

    monkeypatch.setattr(benchmark_module, "ASR", model_factory)
    monkeypatch.setattr(benchmark_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        benchmark_module,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=device_factory),
            get_default_memory_pool=Mock(return_value=memory_pool),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--model-dir",
            str(model_path),
            "--wav",
            str(wav_path),
            "--device-id",
            "1",
            *batch_arguments,
            "--warmups",
            str(warmups),
            "--runs",
            "3",
        ],
    )

    with caplog.at_level(logging.INFO, logger=benchmark_module.__name__):
        benchmark_module.main()

    records = [
        record for record in caplog.records if record.name == benchmark_module.__name__
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    result = json.loads(records[0].getMessage())
    audio_seconds = samples.size * batch_size / sample_rate
    assert result == pytest.approx(
        {
            "encoder_sec": 2.0,
            "decoder_sec": 3.0,
            "postprocess_sec": 0.2,
            "total_sec": 4.0,
            "audio_seconds": audio_seconds,
            "rtfx": audio_seconds / 4.0,
            "cupy_memory_pool_bytes": 4096,
        }
    )

    model_factory.assert_called_once_with(model_path, device_id=1)
    device_factory.assert_called_once_with(1)
    device.__enter__.assert_called_once_with()
    device.__exit__.assert_called_once_with(None, None, None)
    memory_pool.free_all_blocks.assert_called_once_with()
    memory_pool.total_bytes.assert_called_once_with()
    assert stream.synchronizations == 10
    assert encoder.call_count == decoder.call_count == postprocessor.call_count == 3
    assert model.call_count == warmups + 3

    audios = encoder.call_args_list[0].args[0]
    assert len(audios) == batch_size
    expected_audio = samples.astype(np.float32) / np.float32(2**15)
    for audio in audios:
        np.testing.assert_array_equal(audio, expected_audio)
    assert all(
        call.args[0] is audios for call in encoder.call_args_list + model.call_args_list
    )
    assert all(
        call.args[0] is encoder_output and call.args[1] is encoder_output_lengths
        for call in decoder.call_args_list
    )
    assert all(
        call.args[0] is audios
        and call.args[1] is token_ids
        and call.args[2] is timestamps
        for call in postprocessor.call_args_list
    )


def test_main_reads_wav_before_initializing_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.zeros(8, dtype=np.int16), channels=2)
    model_factory = Mock()
    monkeypatch.setattr(benchmark_module, "ASR", model_factory)
    monkeypatch.setattr("sys.argv", [*BASE_ARGUMENTS[:-1], str(wav_path)])

    with pytest.raises(ValueError, match="Expected a mono PCM16 WAV file"):
        benchmark_module.main()

    model_factory.assert_not_called()


@pytest.mark.parametrize(
    ("model_sample_rate", "batch_arguments", "message"),
    (
        (8000, (), "Expected 8000 Hz benchmark audio, got 16000 Hz"),
        (16000, ("--batch-size", "3"), "batch-size must be in [1, 2], got 3"),
    ),
    ids=("sample-rate", "batch-capacity"),
)
def test_main_rejects_incompatible_input_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    model_sample_rate: int,
    batch_arguments: tuple[str, ...],
    message: str,
) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.zeros(160, dtype=np.int16))
    model = SimpleNamespace(
        encoder=SimpleNamespace(batch_size=2, sample_rate=model_sample_rate)
    )
    model_factory = Mock(return_value=model)
    device_factory = Mock()
    memory_pool_factory = Mock()
    monkeypatch.setattr(benchmark_module, "ASR", model_factory)
    monkeypatch.setattr(
        benchmark_module,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=device_factory),
            get_default_memory_pool=memory_pool_factory,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--model-dir",
            "model",
            "--wav",
            str(wav_path),
            *batch_arguments,
        ],
    )

    with (
        caplog.at_level(logging.ERROR, logger=benchmark_module.__name__),
        pytest.raises(SystemExit) as error,
    ):
        benchmark_module.main()

    assert error.value.code == 2
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == benchmark_module.__name__
    ] == [message]
    model_factory.assert_called_once_with(Path("model"), device_id=0)
    device_factory.assert_not_called()
    memory_pool_factory.assert_not_called()
