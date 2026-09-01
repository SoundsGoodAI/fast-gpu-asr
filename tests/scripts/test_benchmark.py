#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the packaged-model benchmark command line."""

import importlib.util
import json
import logging
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "fast_gpu_asr_test_benchmark", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
benchmark_module = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(benchmark_module)
parse_args = benchmark_module.parse_args
read_pcm16 = benchmark_module.read_pcm16


def write_wav(
    path: Path,
    samples: np.typing.NDArray[np.int16],
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> None:
    """Write a small PCM WAV fixture using only the standard library."""

    if sample_width == 2:
        frame_bytes = samples.astype("<i2", copy=False).tobytes()
    else:
        # Incompatible-format tests still need a payload consistent with the
        # declared sample width so they exercise only the intended validation.
        frame_bytes = bytes(samples.size * sample_width)

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
    ((2, 2), (1, 1), (1, 3), (1, 4)),
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

    with pytest.raises(ValueError, match="mono PCM16"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_truncated_payload(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.arange(8, dtype=np.int16))
    wav_path.write_bytes(wav_path.read_bytes()[:-2])

    with pytest.raises(ValueError, match=rf"Truncated PCM16 WAV file {wav_path}"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_partial_pcm_frame(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.array([1], dtype=np.int16))
    wav_bytes = bytearray(wav_path.read_bytes())
    data_offset = wav_bytes.index(b"data")
    struct.pack_into("<I", wav_bytes, data_offset + 4, 3)
    # Add one partial-sample byte and its RIFF padding byte. The container is
    # structurally complete, but its PCM16 data chunk is not frame-aligned.
    wav_bytes.extend(b"\0\0")
    struct.pack_into("<I", wav_bytes, 4, len(wav_bytes) - 8)
    wav_path.write_bytes(wav_bytes)

    with pytest.raises(ValueError, match=rf"Truncated PCM16 WAV file {wav_path}"):
        read_pcm16(wav_path)


def test_read_pcm16_rejects_empty_waveform(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.empty(0, dtype=np.int16))

    with pytest.raises(ValueError, match="nonempty PCM16"):
        read_pcm16(wav_path)


def test_benchmark_parser_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark.py", "--model-dir", "model", "--wav", "audio.wav"],
    )
    args = parse_args()

    assert args.model_dir == Path("model")
    assert args.wav == Path("audio.wav")
    assert args.device_id == 0
    assert args.batch_size is None
    assert args.warmups == 3
    assert args.runs == 10


def test_benchmark_parser_accepts_count_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--model-dir",
            "model",
            "--wav",
            "audio.wav",
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

    assert args.device_id == 0
    assert args.batch_size == 1
    assert args.warmups == 0
    assert args.runs == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ("--batch-size", "0"),
            "argument --batch-size: expected a positive integer, got 0",
        ),
        (
            ("--warmups", "-1"),
            "argument --warmups: expected a nonnegative integer, got -1",
        ),
        (("--runs", "0"), "argument --runs: expected a positive integer, got 0"),
        (("--runs", "-1"), "argument --runs: expected a positive integer, got -1"),
        (
            ("--device-id", "-1"),
            "argument --device-id: expected a nonnegative integer, got -1",
        ),
    ),
)
def test_benchmark_parser_rejects_invalid_counts(
    arguments: tuple[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark.py",
            "--model-dir",
            "model",
            "--wav",
            "audio.wav",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse_args()

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert message in captured.err


@pytest.mark.parametrize(
    ("batch_arguments", "expected_batch_size", "warmups"),
    (((), 2, 2), (("--batch-size", "1"), 1, 0)),
    ids=("engine-capacity-with-warmups", "partial-batch-without-warmups"),
)
def test_benchmark_main_runs_synchronized_measurements_and_reports_medians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    batch_arguments: tuple[str, ...],
    expected_batch_size: int,
    warmups: int,
) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.zeros(160, dtype=np.int16))

    class FakeStream:
        def __init__(self) -> None:
            self.pending_seconds = 0.0

        def synchronize(self) -> None:
            assert device.active
            events.append("stream:synchronize")
            clock[0] += self.pending_seconds
            self.pending_seconds = 0.0

    class FakeEncoder:
        batch_size = 2
        sample_rate = 16000

        def __call__(self, audios):
            assert device.active
            assert_batch(audios)
            events.append("encoder")
            stream.pending_seconds += next(encoder_durations)
            encoder_output = np.zeros((len(audios), 1, 2))
            encoder_output_lengths = np.ones(len(audios), np.int32)
            data_flow["audios"] = audios
            data_flow["encoder_output"] = encoder_output
            data_flow["encoder_output_lengths"] = encoder_output_lengths
            return encoder_output, encoder_output_lengths

    class FakeDecoder:
        def __call__(self, encoder_output, encoder_output_lengths):
            assert device.active
            assert encoder_output is data_flow["encoder_output"]
            assert encoder_output_lengths is data_flow["encoder_output_lengths"]
            events.append("decoder")
            stream.pending_seconds += next(decoder_durations)
            batch_size = len(encoder_output_lengths)
            token_ids = [[1] for _ in range(batch_size)]
            timestamps = [[0.0] for _ in range(batch_size)]
            data_flow["token_ids"] = token_ids
            data_flow["timestamps"] = timestamps
            return token_ids, timestamps

    class FakePostProcessor:
        def __call__(self, audios, token_ids, timestamps):
            assert device.active
            assert audios is data_flow["audios"]
            assert token_ids is data_flow["token_ids"]
            assert timestamps is data_flow["timestamps"]
            events.append("postprocessor")
            clock[0] += next(postprocess_durations)
            return ["word"] * len(audios), [[("word", 0.0, 0.01)]] * len(audios)

    class FakeASR:
        def __init__(self, model_dir: Path, device_id: int) -> None:
            assert model_dir == Path("model")
            assert device_id == 1
            self.encoder = FakeEncoder()
            self.decoder = FakeDecoder()
            self.postprocessor = FakePostProcessor()
            self.stream = stream

        def __call__(self, audios):
            assert device.active
            assert memory_pool.freed
            assert_batch(audios)
            events.append("model")
            stream.pending_seconds += next(model_durations)
            return ["word"] * len(audios), [[] for _ in audios]

    class FakeMemoryPool:
        def __init__(self) -> None:
            self.freed = False

        def free_all_blocks(self) -> None:
            assert device.active
            events.append("pool:free")
            self.freed = True

        def total_bytes(self) -> int:
            assert device.active
            events.append("pool:total_bytes")
            return 4096

    memory_pool = FakeMemoryPool()
    stream = FakeStream()
    events: list[str] = []
    data_flow: dict[str, object] = {}
    clock = [0.0]
    model_durations = iter((*((0.5,) * warmups), 4.0, 2.0, 3.0))
    encoder_durations = iter((1.0, 2.0, 100.0))
    decoder_durations = iter((100.0, 3.0, 2.0))
    postprocess_durations = iter((0.1, 0.2, 0.3))

    class FakeDevice:
        active = False

        def __init__(self, device_id: int) -> None:
            assert device_id == 1

        def __enter__(self) -> None:
            assert not self.active
            self.active = True
            events.append("device:enter")
            return None

        def __exit__(self, *args: object) -> None:
            assert self.active
            events.append("device:exit")
            self.active = False
            return None

    device = FakeDevice(1)

    def assert_batch(audios: object) -> None:
        assert isinstance(audios, list)
        assert len(audios) == expected_batch_size
        for audio in audios:
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            np.testing.assert_array_equal(audio, np.zeros(160, dtype=np.float32))

    def get_memory_pool() -> FakeMemoryPool:
        assert device.active
        events.append("pool:get")
        return memory_pool

    def get_device(device_id: int) -> FakeDevice:
        assert device_id == 1
        return device

    monkeypatch.setattr(benchmark_module, "ASR", FakeASR)
    monkeypatch.setattr(benchmark_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        benchmark_module,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(Device=get_device),
            get_default_memory_pool=get_memory_pool,
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
    assert memory_pool.freed
    assert not device.active
    expected_events = (
        ["device:enter", "pool:get", "pool:free"]
        + ["model"] * warmups
        + ["stream:synchronize"]
        + [
            "encoder",
            "stream:synchronize",
            "decoder",
            "stream:synchronize",
            "postprocessor",
        ]
        * 3
        + ["model", "stream:synchronize"] * 3
        + ["pool:total_bytes", "device:exit"]
    )
    assert events == expected_events
    assert result["encoder_sec"] == pytest.approx(2.0)
    assert result["decoder_sec"] == pytest.approx(3.0)
    assert result["postprocess_sec"] == pytest.approx(0.2)
    assert result["total_sec"] == pytest.approx(3.0)
    audio_seconds = 0.01 * expected_batch_size
    assert result["audio_seconds"] == pytest.approx(audio_seconds)
    assert result["rtfx"] == pytest.approx(audio_seconds / 3.0)
    assert result["cupy_memory_pool_bytes"] == 4096
    assert set(result) == {
        "audio_seconds",
        "cupy_memory_pool_bytes",
        "decoder_sec",
        "encoder_sec",
        "postprocess_sec",
        "rtfx",
        "total_sec",
    }


def test_benchmark_main_rejects_invalid_wav_before_model_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav_path = tmp_path / "stereo.wav"
    write_wav(wav_path, np.zeros(8, dtype=np.int16), channels=2)

    monkeypatch.setattr(
        benchmark_module,
        "ASR",
        lambda *_args, **_kwargs: pytest.fail(
            "Invalid benchmark audio initialized the model."
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark.py", "--model-dir", "model", "--wav", str(wav_path)],
    )

    with pytest.raises(ValueError, match="mono PCM16"):
        benchmark_module.main()


@pytest.mark.parametrize(
    ("model_sample_rate", "batch_size", "message"),
    (
        (8_000, None, "Expected 8000 Hz benchmark audio, got 16000 Hz"),
        (16_000, 3, "batch-size must be in [1, 2], got 3"),
    ),
)
def test_benchmark_main_rejects_incompatible_input_before_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_sample_rate: int,
    batch_size: int | None,
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, np.zeros(160, dtype=np.int16))

    class FakeASR:
        def __init__(self, *_: object, **__: object) -> None:
            self.encoder = type(
                "FakeEncoder",
                (),
                {"batch_size": 2, "sample_rate": model_sample_rate},
            )()

        def __call__(self, _: object) -> None:
            pytest.fail("Invalid benchmark input reached warmup inference.")

    monkeypatch.setattr(benchmark_module, "ASR", FakeASR)
    monkeypatch.setattr(
        benchmark_module,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=lambda *_args, **_kwargs: pytest.fail(
                    "Invalid benchmark input entered a CUDA device context."
                )
            ),
            get_default_memory_pool=lambda: pytest.fail(
                "Invalid benchmark input touched the memory pool."
            ),
        ),
    )
    arguments = ["--model-dir", "model", "--wav", str(wav_path)]
    if batch_size is not None:
        arguments.extend(("--batch-size", str(batch_size)))
    monkeypatch.setattr("sys.argv", ["benchmark.py", *arguments])

    with (
        caplog.at_level(logging.ERROR, logger=benchmark_module.__name__),
        pytest.raises(SystemExit) as error,
    ):
        benchmark_module.main()

    assert error.value.code == 2
    records = [
        record for record in caplog.records if record.name == benchmark_module.__name__
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].getMessage() == message
