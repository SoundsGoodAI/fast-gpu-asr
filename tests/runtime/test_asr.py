#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Composition tests for the public batched ASR pipeline."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError
from yaml import YAMLError

import fast_gpu_asr.asr as asr_module
from fast_gpu_asr import ASR
from fast_gpu_asr.utils import ASRInitializationError


class FakeStream:
    """Stand in for the shared nonblocking CuPy stream."""


class FakeDevice:
    """Record entry and exit from one requested CUDA device context."""

    events: list[tuple[str, int]] = []

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id

    def __enter__(self) -> "FakeDevice":
        self.events.append(("enter_device", self.device_id))
        return self

    def __exit__(self, *args: object) -> None:
        self.events.append(("exit_device", self.device_id))


class RecordingLock:
    """Record lock scope and expose whether pipeline stages are protected."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active = False

    def __enter__(self) -> "RecordingLock":
        assert not self.active
        self.active = True
        self.events.append("enter_lock")
        return self

    def __exit__(self, *args: object) -> None:
        assert self.active
        self.events.append("exit_lock")
        self.active = False


def make_config(model_type: str, decoder_type: str) -> dict[str, object]:
    """Return metadata containing exactly the fields consumed by ``ASR``."""

    zipformer = model_type == "zipformer_asr"
    audio_encoder_params: dict[str, object] = {
        "frame_shift_ms": 10,
        "subsampling_factor": 4 if zipformer else 8,
    }
    if zipformer:
        audio_encoder_params["right_padding_samples"] = 200

    decoder_params: dict[str, object] = {
        "beam": 1,
        "blank_penalty": 0.25,
    }
    if model_type == "zipformer_asr" and decoder_type != "ctc_greedy_search":
        decoder_params["context_size"] = 2
    elif model_type == "parakeet_asr":
        decoder_params.update(
            {
                "max_symbols_per_timestep": 10,
                "tdt_durations": [0, 1, 2, 3, 4],
            }
        )

    return {
        "model_type": model_type,
        "decoder_type": decoder_type,
        "model_samplerate": 16000,
        "vocab_size": 32,
        "blank_id": 0 if zipformer else 32,
        "audio_encoder_params": audio_encoder_params,
        "decoder_params": decoder_params,
    }


def write_config(
    model_dir: Path,
    model_type: str,
    decoder_type: str,
) -> None:
    """Write one minimal model configuration used by constructor tests."""

    OmegaConf.save(
        make_config(model_type, decoder_type),
        model_dir / "model_config.yaml",
    )


def patch_runtime_components(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[tuple[object, ...]]], FakeStream]:
    """Install recording runtime components and return their constructor calls."""

    FakeDevice.events = []
    calls: dict[str, list[tuple[object, ...]]] = {
        "stream": [],
        "encoder": [],
        "ctc": [],
        "zipformer": [],
        "parakeet": [],
        "postprocessor": [],
    }
    stream = FakeStream()

    class FakeEncoder:
        batch_size = 7

        def __init__(self, *args: object) -> None:
            calls["encoder"].append(args)

    class FakeCTCDecoder:
        def __init__(self, *args: object) -> None:
            calls["ctc"].append(args)

    class FakeZipformerDecoder:
        def __init__(self, *args: object) -> None:
            calls["zipformer"].append(args)

    class FakeParakeetDecoder:
        def __init__(self, *args: object) -> None:
            calls["parakeet"].append(args)

    class FakePostProcessor:
        def __init__(self, *args: object) -> None:
            calls["postprocessor"].append(args)

    monkeypatch.setattr(asr_module.cp.cuda, "Device", FakeDevice)

    def make_stream(**kwargs: object) -> FakeStream:
        calls["stream"].append(tuple(sorted(kwargs.items())))
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
    return calls, stream


def test_asr_propagates_missing_model_config_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model_config.yaml"):
        ASR(tmp_path)


def test_asr_propagates_malformed_model_config_error(tmp_path: Path) -> None:
    (tmp_path / "model_config.yaml").write_text("model_type: [\n", encoding="utf8")

    with pytest.raises(YAMLError):
        ASR(tmp_path)


def test_asr_propagates_non_utf8_model_config_error(tmp_path: Path) -> None:
    (tmp_path / "model_config.yaml").write_bytes(b"model_type: \xff")

    with pytest.raises(UnicodeDecodeError):
        ASR(tmp_path)


def test_asr_propagates_interpolation_failure_in_routing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config("zipformer_asr", "ctc_greedy_search")
    config["model_type"] = "${missing_model_type}"
    OmegaConf.save(config, tmp_path / "model_config.yaml")
    calls, _ = patch_runtime_components(monkeypatch)

    with pytest.raises(InterpolationKeyError, match="missing_model_type"):
        ASR(tmp_path, validate=False)

    assert not any(calls.values())
    assert FakeDevice.events == [("enter_device", 0), ("exit_device", 0)]


def test_asr_propagates_cuda_device_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    failure = RuntimeError("invalid device ordinal")

    class FailingDevice:
        def __init__(self, device_id: int) -> None:
            assert device_id == 7

        def __enter__(self) -> None:
            raise failure

        def __exit__(self, *args: object) -> None:
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
    calls, stream = patch_runtime_components(monkeypatch)
    monkeypatch.setattr(
        asr_module,
        "validate_model",
        lambda *_: pytest.fail("validate=False unexpectedly validated the model."),
    )

    model = ASR(tmp_path, device_id=3, validate=False)

    right_padding_samples = 200 if model_type == "zipformer_asr" else 0
    assert calls["encoder"] == [
        (
            tmp_path / encoder_filename,
            16000,
            3,
            stream,
            right_padding_samples,
        )
    ]
    assert model.encoder.batch_size == 7
    assert model.stream is stream
    assert model.call_lock.acquire(blocking=False)
    model.call_lock.release()
    assert FakeDevice.events == [("enter_device", 3), ("exit_device", 3)]
    assert calls["stream"] == [
        (("non_blocking", True), ("null", False), ("ptds", False))
    ]
    assert calls["postprocessor"] == [(tmp_path / "bpe.model", 16000)]
    assert sum(bool(calls[name]) for name in ("ctc", "zipformer", "parakeet")) == 1

    decoder_args = calls[decoder_call][0]
    if decoder_filename is None:
        assert decoder_args == (0, frame_shift_sec, 0.25, 3, stream)
    elif decoder_call == "zipformer":
        assert decoder_args == (
            tmp_path / decoder_filename,
            7,
            2,
            32,
            0,
            frame_shift_sec,
            0.25,
            3,
            stream,
        )
    else:
        assert decoder_args == (
            tmp_path / decoder_filename,
            7,
            32,
            (0, 1, 2, 3, 4),
            10,
            frame_shift_sec,
            0.25,
            3,
            stream,
        )


def test_asr_validates_by_default_on_requested_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    calls, _ = patch_runtime_components(monkeypatch)
    events: list[tuple[str, object]] = []

    def validate_model(model_dir: Path, model_config: object) -> None:
        events.append(("validate", model_dir))
        assert model_config.model_type == "zipformer_asr"  # type: ignore[attr-defined]
        assert FakeDevice.events == [("enter_device", 4)]

    monkeypatch.setattr(asr_module, "validate_model", validate_model)

    ASR(tmp_path, device_id=4)

    assert events == [("validate", tmp_path)]
    assert FakeDevice.events == [("enter_device", 4), ("exit_device", 4)]
    assert len(calls["encoder"]) == 1


def test_asr_initializes_gpu_components_inside_requested_device_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep validation and stream creation on the selected CUDA device."""

    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    patch_runtime_components(monkeypatch)
    events: list[tuple[str, int]] = []
    active_device: int | None = None

    class OrderedDevice:
        def __init__(self, device_id: int) -> None:
            self.device_id = device_id

        def __enter__(self) -> "OrderedDevice":
            nonlocal active_device
            active_device = self.device_id
            events.append(("enter", self.device_id))
            return self

        def __exit__(self, *args: object) -> None:
            nonlocal active_device
            events.append(("exit", self.device_id))
            active_device = None

    def validate_model(*args: object) -> None:
        del args
        assert active_device == 4
        events.append(("validate", active_device))

    def make_stream(**kwargs: object) -> FakeStream:
        assert kwargs == {"null": False, "non_blocking": True, "ptds": False}
        assert active_device == 4
        events.append(("stream", active_device))
        return FakeStream()

    class OrderedEncoder:
        batch_size = 7

        def __init__(self, *args: object) -> None:
            del args
            assert active_device == 4
            events.append(("encoder", active_device))

    class OrderedDecoder:
        def __init__(self, *args: object) -> None:
            del args
            assert active_device == 4
            events.append(("decoder", active_device))

    monkeypatch.setattr(asr_module.cp.cuda, "Device", OrderedDevice)
    monkeypatch.setattr(asr_module.cp.cuda, "Stream", make_stream)
    monkeypatch.setattr(asr_module, "validate_model", validate_model)
    monkeypatch.setattr(asr_module, "Encoder", OrderedEncoder)
    monkeypatch.setattr(asr_module, "CTCGreedyDecoder", OrderedDecoder)

    ASR(tmp_path, device_id=4)

    assert events == [
        ("enter", 4),
        ("validate", 4),
        ("stream", 4),
        ("encoder", 4),
        ("decoder", 4),
        ("exit", 4),
    ]


def test_asr_validation_failure_stops_component_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    calls, _ = patch_runtime_components(monkeypatch)
    error = RuntimeError("invalid bundle")

    def fail_validation(*args: object) -> None:
        raise error

    monkeypatch.setattr(asr_module, "validate_model", fail_validation)

    with pytest.raises(RuntimeError) as raised:
        ASR(tmp_path)

    assert raised.value is error
    assert not any(calls.values())
    assert FakeDevice.events == [("enter_device", 0), ("exit_device", 0)]


def test_asr_preserves_validation_initialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    calls, _ = patch_runtime_components(monkeypatch)
    failure = ASRInitializationError("invalid model bundle")

    def fail_validation(*args: object) -> None:
        raise failure

    monkeypatch.setattr(asr_module, "validate_model", fail_validation)

    with pytest.raises(ASRInitializationError) as error:
        ASR(tmp_path)

    assert error.value is failure
    assert not any(calls.values())
    assert FakeDevice.events == [("enter_device", 0), ("exit_device", 0)]


@pytest.mark.parametrize(
    ("model_type", "decoder_type", "component_name"),
    (
        ("zipformer_asr", "ctc_greedy_search", "Encoder"),
        ("zipformer_asr", "ctc_greedy_search", "CTCGreedyDecoder"),
        (
            "zipformer_asr",
            "transducer_modified_beam_search",
            "ZipformerModifiedBeamSearchDecoder",
        ),
        (
            "parakeet_asr",
            "transducer_modified_beam_search",
            "ParakeetModifiedBeamSearchDecoder",
        ),
        ("zipformer_asr", "ctc_greedy_search", "PostProcessor"),
    ),
)
def test_asr_preserves_component_initialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    decoder_type: str,
    component_name: str,
) -> None:
    """Do not rewrap initialization errors raised by runtime components."""

    write_config(tmp_path, model_type, decoder_type)
    patch_runtime_components(monkeypatch)
    failure = ASRInitializationError(component_name)

    def fail(*args: object) -> None:
        del args
        raise failure

    monkeypatch.setattr(asr_module, component_name, fail)

    with pytest.raises(ASRInitializationError) as error:
        ASR(str(tmp_path), validate=False)

    assert error.value is failure
    assert FakeDevice.events == [("enter_device", 0), ("exit_device", 0)]


@pytest.mark.parametrize("failing_component", ("encoder", "ctc", "postprocessor"))
def test_asr_propagates_component_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_component: str,
) -> None:
    write_config(tmp_path, "zipformer_asr", "ctc_greedy_search")
    calls, _ = patch_runtime_components(monkeypatch)
    failure = RuntimeError(failing_component)
    component_names = {
        "encoder": "Encoder",
        "ctc": "CTCGreedyDecoder",
        "postprocessor": "PostProcessor",
    }

    def fail(*args: object) -> None:
        raise failure

    monkeypatch.setattr(asr_module, component_names[failing_component], fail)

    with pytest.raises(RuntimeError) as error:
        ASR(tmp_path, validate=False)

    assert error.value is failure
    assert FakeDevice.events == [("enter_device", 0), ("exit_device", 0)]
    if failing_component == "encoder":
        assert calls["ctc"] == calls["postprocessor"] == []
    elif failing_component == "ctc":
        assert len(calls["encoder"]) == 1
        assert calls["postprocessor"] == []
    else:
        assert len(calls["encoder"]) == len(calls["ctc"]) == 1


def test_asr_call_forwards_exact_stage_outputs() -> None:
    model = ASR.__new__(ASR)
    audios = [np.zeros(3, dtype=np.float32)]
    encoder_output = object()
    encoder_lengths = object()
    token_ids = object()
    timestamps = object()
    result = (["text"], [[("text", 0.0, 0.1)]])
    events: list[str] = []
    call_lock = RecordingLock(events)

    def encoder(received_audios: object) -> tuple[object, object]:
        assert received_audios is audios
        assert call_lock.active
        events.append("encoder")
        return encoder_output, encoder_lengths

    def decoder(
        received_output: object,
        received_lengths: object,
    ) -> tuple[object, object]:
        assert received_output is encoder_output
        assert received_lengths is encoder_lengths
        assert call_lock.active
        events.append("decoder")
        return token_ids, timestamps

    def postprocessor(
        received_audios: object,
        received_tokens: object,
        received_timestamps: object,
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        assert received_audios is audios
        assert received_tokens is token_ids
        assert received_timestamps is timestamps
        assert call_lock.active
        events.append("postprocessor")
        return result

    model.encoder = encoder  # type: ignore[assignment]
    model.decoder = decoder  # type: ignore[assignment]
    model.postprocessor = postprocessor  # type: ignore[assignment]
    model.call_lock = call_lock  # type: ignore[assignment]

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


@pytest.mark.parametrize("failing_stage", ("encoder", "decoder", "postprocessor"))
def test_asr_call_propagates_failure_and_stops(
    failing_stage: str,
) -> None:
    model = ASR.__new__(ASR)
    error = RuntimeError(failing_stage)
    events: list[str] = []
    call_lock = RecordingLock(events)

    def stage(name: str, result: object) -> Callable[..., object]:
        def call(*args: object) -> object:
            assert call_lock.active
            events.append(name)
            if name == failing_stage:
                raise error
            return result

        return call

    model.encoder = stage("encoder", (object(), object()))  # type: ignore[assignment]
    model.decoder = stage("decoder", (object(), object()))  # type: ignore[assignment]
    model.postprocessor = stage("postprocessor", ([], []))  # type: ignore[assignment]
    model.call_lock = call_lock  # type: ignore[assignment]

    with pytest.raises(RuntimeError) as raised:
        model([np.zeros(1, dtype=np.float32)])

    assert raised.value is error
    expected_events = ["encoder", "decoder", "postprocessor"]
    assert events == [
        "enter_lock",
        *expected_events[: expected_events.index(failing_stage) + 1],
        "exit_lock",
    ]
    assert not call_lock.active
