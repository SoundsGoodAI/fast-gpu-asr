#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Composition tests for the public batched ASR pipeline."""

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from yaml import YAMLError

import fast_gpu_asr.asr as asr_module
from fast_gpu_asr import ASR
from fast_gpu_asr.utils import ASRInitializationError


class FakeStream:
    """Stand in for the shared nonblocking CuPy stream."""


type RuntimeArgument = Path | int | float | tuple[int, ...] | FakeStream
type RuntimeCall = tuple[RuntimeArgument, ...] | tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class PatchedRuntime:
    """Collect calls and CUDA-scope events from patched runtime components."""

    calls: dict[str, list[RuntimeCall]]
    stream: FakeStream
    events: list[tuple[str, int | None]]


class RecordingLock:
    """Record lock scope and expose whether pipeline stages are protected."""

    def __init__(self, events: list[str]) -> None:
        """Initialize an unlocked scope backed by an event log.

        Parameters
        ----------
        events : list[str]
            Mutable log receiving lock entry and exit events.
        """

        self.events = events
        self.active = False

    def __enter__(self) -> "RecordingLock":
        """Enter the lock scope.

        Returns
        -------
        RecordingLock
            Active lock instance.
        """

        assert not self.active
        self.active = True
        self.events.append("enter_lock")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the lock scope and preserve any propagated exception.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class leaving the scope, when present.
        exc_value : BaseException | None
            Exception instance leaving the scope, when present.
        traceback : TracebackType | None
            Exception traceback leaving the scope, when present.
        """

        del exc_type, exc_value, traceback
        assert self.active
        self.events.append("exit_lock")
        self.active = False


def make_config(model_type: str, decoder_type: str) -> DictConfig:
    """Return minimal routing metadata consumed directly by ``ASR``.

    Parameters
    ----------
    model_type : str
        Zipformer or Parakeet runtime model identifier.
    decoder_type : str
        CTC, greedy transducer, or modified-beam-search decoder identifier.

    Returns
    -------
    DictConfig
        Minimal model configuration for constructor tests.
    """

    zipformer = model_type == "zipformer_asr"
    audio_encoder_params = {
        "frame_shift_ms": 10,
        "subsampling_factor": 4 if zipformer else 8,
    }
    if zipformer:
        audio_encoder_params["right_padding_samples"] = 200

    decoder_params: dict[str, float | int | list[int]] = {"blank_penalty": 0.25}
    if model_type == "zipformer_asr" and decoder_type != "ctc_greedy_search":
        decoder_params["context_size"] = 2
    elif model_type == "parakeet_asr":
        decoder_params.update(
            {
                "max_symbols_per_timestep": 10,
                "tdt_durations": [0, 1, 2, 3, 4],
            }
        )

    return OmegaConf.create(
        {
            "model_type": model_type,
            "decoder_type": decoder_type,
            "model_samplerate": 16000,
            "vocab_size": 32,
            "blank_id": 0 if zipformer else 32,
            "audio_encoder_params": audio_encoder_params,
            "decoder_params": decoder_params,
        }
    )


def write_config(
    model_dir: Path,
    model_type: str,
    decoder_type: str,
) -> None:
    """Write one minimal model configuration used by constructor tests.

    Parameters
    ----------
    model_dir : Path
        Directory receiving ``model_config.yaml``.
    model_type : str
        Zipformer or Parakeet runtime model identifier.
    decoder_type : str
        Decoder identifier stored in the configuration.
    """

    OmegaConf.save(
        make_config(model_type, decoder_type),
        model_dir / "model_config.yaml",
    )


def patch_runtime_components(
    monkeypatch: pytest.MonkeyPatch,
) -> PatchedRuntime:
    """Install runtime fakes that record construction and CUDA scope.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace CUDA and pipeline constructors.

    Returns
    -------
    PatchedRuntime
        Constructor calls, shared stream, and ordered scope events.
    """

    calls: dict[str, list[RuntimeCall]] = {
        "stream": [],
        "encoder": [],
        "ctc": [],
        "zipformer": [],
        "parakeet": [],
        "postprocessor": [],
    }
    stream = FakeStream()
    events: list[tuple[str, int | None]] = []
    active_device: int | None = None

    class FakeDevice:
        """Track entry and exit for one requested CUDA device."""

        def __init__(self, device_id: int) -> None:
            """Store the requested CUDA device identifier."""

            self.device_id = device_id

        def __enter__(self) -> "FakeDevice":
            """Enter and record the selected CUDA device."""

            nonlocal active_device
            assert active_device is None
            active_device = self.device_id
            events.append(("enter_device", self.device_id))
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            """Exit and record the selected CUDA device."""

            del exc_type, exc_value, traceback
            nonlocal active_device
            events.append(("exit_device", self.device_id))
            active_device = None

    class RecordingComponent:
        """Record construction of one patched pipeline component."""

        component_name = ""

        def __init__(self, *args: RuntimeArgument) -> None:
            """Record constructor arguments and the active CUDA device."""

            calls[self.component_name].append(args)
            events.append((self.component_name, active_device))

    class FakeEncoder(RecordingComponent):
        """Record encoder construction."""

        component_name = "encoder"
        batch_size = 7

    class FakeCTCDecoder(RecordingComponent):
        """Record CTC decoder construction."""

        component_name = "ctc"

    class FakeZipformerDecoder(RecordingComponent):
        """Record Zipformer transducer decoder construction."""

        component_name = "zipformer"

    class FakeParakeetDecoder(RecordingComponent):
        """Record Parakeet decoder construction."""

        component_name = "parakeet"

    class FakePostProcessor(RecordingComponent):
        """Record postprocessor construction."""

        component_name = "postprocessor"

    monkeypatch.setattr(asr_module.cp.cuda, "Device", FakeDevice)

    def make_stream(**kwargs: bool) -> FakeStream:
        """Record stream options and return the shared fake stream."""

        calls["stream"].append(tuple(sorted(kwargs.items())))
        events.append(("stream", active_device))
        return stream

    monkeypatch.setattr(asr_module.cp.cuda, "Stream", make_stream)
    monkeypatch.setattr(asr_module, "Encoder", FakeEncoder)
    monkeypatch.setattr(asr_module, "CTCGreedyDecoder", FakeCTCDecoder)
    monkeypatch.setattr(
        asr_module,
        "ZipformerModifiedBeamSearchDecoder",
        FakeZipformerDecoder,
    )
    monkeypatch.setattr(
        asr_module,
        "ParakeetModifiedBeamSearchDecoder",
        FakeParakeetDecoder,
    )
    monkeypatch.setattr(asr_module, "PostProcessor", FakePostProcessor)
    return PatchedRuntime(calls, stream, events)


def test_asr_propagates_missing_model_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asr_module.cp.cuda,
        "Device",
        lambda _device_id: pytest.fail("CUDA initialized before loading the config."),
    )

    with pytest.raises(FileNotFoundError, match="model_config.yaml"):
        ASR(tmp_path)


def test_asr_propagates_malformed_model_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "model_config.yaml").write_text("model_type: [\n", encoding="utf8")
    monkeypatch.setattr(
        asr_module.cp.cuda,
        "Device",
        lambda _device_id: pytest.fail("CUDA initialized before parsing the config."),
    )

    with pytest.raises(YAMLError):
        ASR(tmp_path)


def test_asr_propagates_interpolation_failure_in_routing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config("zipformer_asr", "ctc_greedy_search")
    config["model_type"] = "${missing_model_type}"
    OmegaConf.save(config, tmp_path / "model_config.yaml")
    runtime = patch_runtime_components(monkeypatch)

    with pytest.raises(InterpolationKeyError, match="missing_model_type"):
        ASR(tmp_path, validate=False)

    assert not any(runtime.calls.values())
    assert runtime.events == [("enter_device", 0), ("exit_device", 0)]


def test_asr_propagates_cuda_device_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    failure = RuntimeError("invalid device ordinal")

    class FailingDevice:
        """Fail on device entry and reject any subsequent exit attempt."""

        def __init__(self, device_id: int) -> None:
            """Check the device ordinal requested by ASR."""

            assert device_id == 7

        def __enter__(self) -> None:
            """Raise the configured device-entry failure."""

            raise failure

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            """Reject an exit call after unsuccessful context entry."""

            del exc_type, exc_value, traceback
            pytest.fail("A failed device context must not be exited.")

    monkeypatch.setattr(asr_module.cp.cuda, "Device", FailingDevice)
    monkeypatch.setattr(
        asr_module,
        "Encoder",
        lambda *_: pytest.fail("Encoder construction must not be reached."),
    )

    with pytest.raises(RuntimeError) as error:
        ASR(tmp_path, device_id=7, validate=False)

    assert error.value is failure


@pytest.mark.parametrize(
    (
        "model_type",
        "decoder_type",
        "encoder_filename",
        "decoder_call",
        "decoder_filename",
        "frame_shift_sec",
    ),
    (
        (
            "zipformer_asr",
            "ctc_greedy_search",
            "zipformer.trt",
            "ctc",
            None,
            0.04,
        ),
        (
            "zipformer_asr",
            "transducer_greedy_search",
            "zipformer.trt",
            "zipformer",
            "decoder.trt",
            0.04,
        ),
        (
            "zipformer_asr",
            "transducer_modified_beam_search",
            "zipformer.trt",
            "zipformer",
            "decoder.trt",
            0.04,
        ),
        (
            "parakeet_asr",
            "transducer_greedy_search",
            "parakeet.trt",
            "parakeet",
            "tdt_decoder.trt",
            0.08,
        ),
        (
            "parakeet_asr",
            "transducer_modified_beam_search",
            "parakeet.trt",
            "parakeet",
            "tdt_decoder.trt",
            0.08,
        ),
    ),
)
def test_asr_routes_model_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    decoder_type: str,
    encoder_filename: str,
    decoder_call: str,
    decoder_filename: str | None,
    frame_shift_sec: float,
) -> None:
    write_config(tmp_path, model_type, decoder_type)
    runtime = patch_runtime_components(monkeypatch)
    monkeypatch.setattr(
        asr_module,
        "validate_model",
        lambda *_: pytest.fail("validate=False unexpectedly validated the model."),
    )

    model = ASR(str(tmp_path), device_id=3, validate=False)

    assert model.call_lock.acquire(blocking=False)
    assert not model.call_lock.acquire(blocking=False)
    model.call_lock.release()

    right_padding_samples = 200 if model_type == "zipformer_asr" else 0
    assert runtime.calls["encoder"] == [
        (
            tmp_path / encoder_filename,
            16000,
            3,
            runtime.stream,
            right_padding_samples,
        )
    ]
    assert type(model.encoder) is asr_module.Encoder
    assert model.encoder.batch_size == 7
    assert model.stream is runtime.stream
    assert runtime.calls["stream"] == [
        (("non_blocking", True), ("null", False), ("ptds", False))
    ]
    assert runtime.calls["postprocessor"] == [(tmp_path / "bpe.model", 16000)]
    assert type(model.postprocessor) is asr_module.PostProcessor

    if decoder_filename is None:
        expected_decoder_args = (0, frame_shift_sec, 0.25, 3, runtime.stream)
    elif decoder_call == "zipformer":
        expected_decoder_args = (
            tmp_path / decoder_filename,
            7,
            2,
            32,
            0,
            frame_shift_sec,
            0.25,
            3,
            runtime.stream,
        )
    else:
        expected_decoder_args = (
            tmp_path / decoder_filename,
            7,
            32,
            (0, 1, 2, 3, 4),
            10,
            frame_shift_sec,
            0.25,
            3,
            runtime.stream,
        )

    decoder_classes = {
        "ctc": asr_module.CTCGreedyDecoder,
        "zipformer": asr_module.ZipformerModifiedBeamSearchDecoder,
        "parakeet": asr_module.ParakeetModifiedBeamSearchDecoder,
    }
    assert runtime.calls[decoder_call] == [expected_decoder_args]
    assert all(
        runtime.calls[name] == [] for name in decoder_classes if name != decoder_call
    )
    assert type(model.decoder) is decoder_classes[decoder_call]
    assert runtime.events == [
        ("enter_device", 3),
        ("stream", 3),
        ("encoder", 3),
        (decoder_call, 3),
        ("exit_device", 3),
        ("postprocessor", None),
    ]


def test_asr_validates_by_default_on_requested_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    runtime = patch_runtime_components(monkeypatch)

    def validate_model(model_dir: Path, model_config: DictConfig) -> None:
        """Record validation while the requested CUDA device is active."""

        assert model_dir == tmp_path
        assert model_config.model_type == "zipformer_asr"
        assert runtime.events == [("enter_device", 4)]
        runtime.events.append(("validate", 4))

    monkeypatch.setattr(asr_module, "validate_model", validate_model)

    ASR(tmp_path, device_id=4)

    assert runtime.events == [
        ("enter_device", 4),
        ("validate", 4),
        ("stream", 4),
        ("encoder", 4),
        ("ctc", 4),
        ("exit_device", 4),
        ("postprocessor", None),
    ]


def test_asr_validation_failure_stops_component_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    OmegaConf.save({"invalid": True}, tmp_path / "model_config.yaml")
    runtime = patch_runtime_components(monkeypatch)
    error = ASRInitializationError("invalid bundle")

    def fail_validation(model_dir: Path, model_config: DictConfig) -> None:
        """Reject the bundle before ASR reads its routing metadata."""

        assert model_dir == tmp_path
        assert model_config.invalid is True
        raise error

    monkeypatch.setattr(asr_module, "validate_model", fail_validation)

    with pytest.raises(ASRInitializationError) as raised:
        ASR(tmp_path)

    assert raised.value is error
    assert not any(runtime.calls.values())
    assert runtime.events == [("enter_device", 0), ("exit_device", 0)]


def test_asr_stream_initialization_failure_stops_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    runtime = patch_runtime_components(monkeypatch)
    failure = RuntimeError("stream unavailable")

    def fail_stream(**kwargs: bool) -> None:
        """Check stream options and raise the configured creation failure."""

        assert kwargs == {"null": False, "non_blocking": True, "ptds": False}
        raise failure

    monkeypatch.setattr(asr_module.cp.cuda, "Stream", fail_stream)

    with pytest.raises(RuntimeError) as error:
        ASR(tmp_path, validate=False)

    assert error.value is failure
    assert not any(runtime.calls.values())
    assert runtime.events == [("enter_device", 0), ("exit_device", 0)]


@pytest.mark.parametrize(
    ("model_type", "decoder_type", "component_name", "completed_components"),
    (
        ("zipformer_asr", "ctc_greedy_search", "Encoder", ()),
        (
            "zipformer_asr",
            "ctc_greedy_search",
            "CTCGreedyDecoder",
            ("encoder",),
        ),
        (
            "zipformer_asr",
            "transducer_modified_beam_search",
            "ZipformerModifiedBeamSearchDecoder",
            ("encoder",),
        ),
        (
            "parakeet_asr",
            "transducer_modified_beam_search",
            "ParakeetModifiedBeamSearchDecoder",
            ("encoder",),
        ),
        (
            "zipformer_asr",
            "ctc_greedy_search",
            "PostProcessor",
            ("encoder", "ctc"),
        ),
    ),
    ids=("encoder", "ctc", "zipformer", "parakeet", "postprocessor"),
)
def test_asr_propagates_component_initialization_failure_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    decoder_type: str,
    component_name: str,
    completed_components: tuple[str, ...],
) -> None:
    write_config(tmp_path, model_type, decoder_type)
    runtime = patch_runtime_components(monkeypatch)
    failure = RuntimeError(component_name)
    failed_calls: list[RuntimeCall] = []

    def fail(*args: RuntimeArgument) -> None:
        """Record a failed constructor and verify the enclosing device scope."""

        failed_calls.append(args)
        expected_events = [
            ("enter_device", 0),
            ("stream", 0),
            *((name, 0) for name in completed_components),
        ]
        if component_name == "PostProcessor":
            expected_events.append(("exit_device", 0))
        assert runtime.events == expected_events
        raise failure

    monkeypatch.setattr(asr_module, component_name, fail)

    with pytest.raises(RuntimeError) as error:
        ASR(tmp_path, validate=False)

    assert error.value is failure
    assert len(failed_calls) == 1
    component_call_counts = {
        name: len(calls) for name, calls in runtime.calls.items() if name != "stream"
    }
    assert component_call_counts == {
        name: int(name in completed_components) for name in component_call_counts
    }
    assert runtime.events == [
        ("enter_device", 0),
        ("stream", 0),
        *((name, 0) for name in completed_components),
        ("exit_device", 0),
    ]


@pytest.mark.parametrize("failing_stage", (None, "encoder", "decoder", "postprocessor"))
def test_asr_call_forwards_outputs_and_recovers_from_failure(
    failing_stage: str | None,
) -> None:
    model = ASR.__new__(ASR)
    audios = [np.zeros(3, dtype=np.float32)]
    encoder_output = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    encoder_lengths = np.array([2], dtype=np.int32)
    token_ids = [[1, 2]]
    timestamps = [[0.0, 0.04]]
    result = (["text"], [[("text", 0.0, 0.1)]])
    events: list[str] = []
    call_lock = RecordingLock(events)
    failure = RuntimeError(failing_stage)

    def record_stage(name: str) -> None:
        """Record a protected pipeline stage and inject its configured failure."""

        assert call_lock.active
        events.append(name)
        if name == failing_stage:
            raise failure

    def encoder(
        received_audios: list[np.typing.NDArray[np.float32]],
    ) -> tuple[
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.int32],
    ]:
        """Check waveform identity and return the fixed encoder outputs."""

        assert received_audios is audios
        record_stage("encoder")
        return encoder_output, encoder_lengths

    def decoder(
        received_output: np.typing.NDArray[np.float32],
        received_lengths: np.typing.NDArray[np.int32],
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Check encoder output identities and return fixed tokens and times."""

        assert received_output is encoder_output
        assert received_lengths is encoder_lengths
        record_stage("decoder")
        return token_ids, timestamps

    def postprocessor(
        received_audios: list[np.typing.NDArray[np.float32]],
        received_tokens: list[list[int]],
        received_timestamps: list[list[float]],
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Check stage-output identities and return fixed transcription results."""

        assert received_audios is audios
        assert received_tokens is token_ids
        assert received_timestamps is timestamps
        record_stage("postprocessor")
        return result

    model.encoder = encoder  # type: ignore[assignment]
    model.decoder = decoder  # type: ignore[assignment]
    model.postprocessor = postprocessor  # type: ignore[assignment]
    model.call_lock = call_lock  # type: ignore[assignment]

    if failing_stage is not None:
        with pytest.raises(RuntimeError) as raised:
            model(audios)
        assert raised.value is failure
        stages = ["encoder", "decoder", "postprocessor"]
        assert events == [
            "enter_lock",
            *stages[: stages.index(failing_stage) + 1],
            "exit_lock",
        ]
        assert not call_lock.active
        events.clear()
        failing_stage = None

    actual_texts, actual_word_timestamps = model(audios)

    assert actual_texts is result[0]
    assert actual_word_timestamps is result[1]
    assert events == [
        "enter_lock",
        "encoder",
        "decoder",
        "postprocessor",
        "exit_lock",
    ]
    assert not call_lock.active
