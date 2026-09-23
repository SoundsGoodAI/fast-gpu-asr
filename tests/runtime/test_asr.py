#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Composition tests for the public batched ASR pipeline."""

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from unittest.mock import Mock

import cupy as cp
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from sentencepiece import sentencepiece_model_pb2
from yaml import YAMLError

import fast_gpu_asr.asr as asr_module
from fast_gpu_asr import ASR
from fast_gpu_asr.export import export_parakeet, export_zipformer
from fast_gpu_asr.export.export_utils import build_tensorrt_engine
from fast_gpu_asr.export.model.parakeet.decoder import Decoder as TDTDecoder
from fast_gpu_asr.export.model.parakeet.parakeet import ParakeetEncoder
from fast_gpu_asr.export.model.zipformer.decoder import Decoder as RNNTDecoder
from fast_gpu_asr.export.model.zipformer.decoder import Joiner
from fast_gpu_asr.export.model.zipformer.subsampling import BiasNorm
from fast_gpu_asr.export.model.zipformer.zipformer import SimpleDownsample, Zipformer2
from fast_gpu_asr.utils import ASRInitializationError
from tests.export.test_export_models import (
    add_zipformer_right_context,
    make_parakeet_encoder,
)


class FakeStream:
    """Stand in for the shared nonblocking CuPy stream."""


type RuntimeArgument = (
    Path | int | float | tuple[int, ...] | cp.cuda.Device | FakeStream
)
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
    elif model_type == "parakeet_asr" and decoder_type != "ctc_greedy_search":
        decoder_params.update(
            {"max_symbols_per_timestep": 10, "tdt_durations": [0, 1, 2, 3, 4]}
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


def write_config(model_dir: Path, model_type: str, decoder_type: str) -> None:
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
        make_config(model_type, decoder_type), model_dir / "model_config.yaml"
    )


def patch_runtime_components(monkeypatch: pytest.MonkeyPatch) -> PatchedRuntime:
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

        def __init__(self, *args: RuntimeArgument) -> None:
            """Record construction and retain the encoder's device and sample rate."""

            super().__init__(*args)
            self.sample_rate = args[1]
            self.device = args[2]

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
        asr_module, "ZipformerModifiedBeamSearchDecoder", FakeZipformerDecoder
    )
    monkeypatch.setattr(
        asr_module, "ParakeetModifiedBeamSearchDecoder", FakeParakeetDecoder
    )
    monkeypatch.setattr(asr_module, "PostProcessor", FakePostProcessor)
    return PatchedRuntime(calls, stream, events)


def test_asr_propagates_missing_model_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        asr_module.cp.cuda,
        "Device",
        lambda device_id: pytest.fail("CUDA initialized before loading the config."),
    )

    with pytest.raises(FileNotFoundError, match="model_config.yaml"):
        ASR(tmp_path)


def test_asr_propagates_malformed_model_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model_config.yaml").write_text("model_type: [\n", encoding="utf8")
    monkeypatch.setattr(
        asr_module.cp.cuda,
        "Device",
        lambda device_id: pytest.fail("CUDA initialized before parsing the config."),
    )

    with pytest.raises(YAMLError):
        ASR(tmp_path)


def test_asr_propagates_interpolation_failure_in_routing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        ("zipformer_asr", "ctc_greedy_search", "zipformer.trt", "ctc", None, 0.04),
        ("parakeet_asr", "ctc_greedy_search", "parakeet.trt", "ctc", None, 0.08),
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
            model.device,
            runtime.stream,
            right_padding_samples,
        )
    ]
    assert isinstance(model.encoder, asr_module.Encoder)
    assert model.device.device_id == 3
    assert model.encoder.batch_size == 7
    assert model.encoder.sample_rate == 16000
    assert model.encoder.device is model.device
    assert model.stream is runtime.stream
    assert runtime.calls["stream"] == [
        (("non_blocking", True), ("null", False), ("ptds", False))
    ]
    assert runtime.calls["postprocessor"] == [(tmp_path / "bpe.model",)]
    assert isinstance(model.postprocessor, asr_module.PostProcessor)

    if decoder_filename is None:
        expected_decoder_args = (
            32 if model_type == "parakeet_asr" else 0,
            frame_shift_sec,
            0.25,
            model.device,
            runtime.stream,
        )
    elif decoder_call == "zipformer":
        expected_decoder_args = (
            tmp_path / decoder_filename,
            7,
            2,
            32,
            0,
            frame_shift_sec,
            0.25,
            model.device,
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
            model.device,
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
    assert isinstance(model.decoder, decoder_classes[decoder_call])
    assert runtime.events == [
        ("enter_device", 3),
        ("stream", 3),
        ("encoder", 3),
        (decoder_call, 3),
        ("exit_device", 3),
        ("postprocessor", None),
    ]


def test_asr_validates_by_default_on_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        ("zipformer_asr", "ctc_greedy_search", "CTCGreedyDecoder", ("encoder",)),
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
        ("zipformer_asr", "ctc_greedy_search", "PostProcessor", ("encoder", "ctc")),
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
    audios = [np.zeros(1600, dtype=np.float32)]
    encoder_output = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    encoder_lengths = np.array([2], dtype=np.int32)
    token_ids = [[1, 2]]
    timestamps = [[0.0, 0.04]]
    result = (["1 2"], [[("1", 0.0, 0.04), ("2", 0.04, 0.2)]])
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
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.int32]]:
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
        audio_durations: list[float],
        received_tokens: list[list[int]],
        received_timestamps: list[list[float]],
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Check stage-output identities and return fixed transcription results."""

        assert audio_durations == [0.2]
        assert received_tokens is token_ids
        assert received_timestamps is timestamps
        record_stage("postprocessor")
        return result

    model.encoder = Mock(side_effect=encoder, sample_rate=8000)
    model.decoder = Mock(side_effect=decoder)
    model.postprocessor = Mock(side_effect=postprocessor)
    model.call_lock = call_lock  # type: ignore[assignment]

    stages = ["encoder", "decoder", "postprocessor"]
    if failing_stage is not None:
        with pytest.raises(RuntimeError) as raised:
            model(audios)
        assert raised.value is failure
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
    assert events == ["enter_lock", *stages, "exit_lock"]
    assert not call_lock.active


def build_bundle(
    directory: Path, family: str, dtype: torch.dtype, ctc: bool
) -> Zipformer2 | ParakeetEncoder:
    """Export tiny real engines and metadata for full-pipeline GPU tests.

    Parameters
    ----------
    directory : pathlib.Path
        Existing directory receiving the tokenizer, configuration, and engines.
    family : str
        ``zipformer`` or ``parakeet``.
    dtype : torch.dtype
        Encoder and transducer decoder precision.
    ctc : bool
        Select CTC instead of transducer decoding.

    Returns
    -------
    Zipformer2 | ParakeetEncoder
        Seeded eager encoder for numerical comparison. Output heads favor
        ``alpha``. Engines use optimization level zero to keep test builds short.
    """

    zipformer = family == "zipformer"
    vocab_size = 10 if zipformer else 5
    proto = sentencepiece_model_pb2.ModelProto()
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = proto.trainer_spec.eos_id = -1
    proto.trainer_spec.vocab_size = vocab_size
    proto.normalizer_spec.name = "identity"
    pieces = (
        "<unk>",
        "<blk>",
        "\u2581alpha",
        "\u2581beta",
        "\u2581",
        "a",
        "b",
        "l",
        "p",
        "h",
    )
    for index, piece in enumerate(pieces[:vocab_size]):
        token = proto.pieces.add(piece=piece, score=-float(index))
        if index == 0:
            token.type = token.UNKNOWN
        elif index == 1:
            token.type = token.CONTROL
    (directory / "bpe.model").write_bytes(proto.SerializeToString())

    args = Namespace(
        output_dir=directory,
        batch_size=2,
        beam=1 if ctc else 2,
        decoder_type="ctc_greedy_search" if ctc else "transducer_modified_beam_search",
        opt_audio_seconds=0.2,
        debug=False,
    )
    encoder_params = {
        "feature_dim": 16,
        "frame_shift_ms": 10,
        "pos_emb_max_len": 64,
        "subsampling_factor": 4 if zipformer else 8,
        "min_audio_seconds": 0.1,
        "opt_audio_seconds": 0.2,
        "max_audio_seconds": 0.4,
    }
    decoder_params = {"beam": args.beam, "blank_penalty": 0.25}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        if zipformer:
            encoder_dims = (16, 24, 32, 40, 32, 24)
            encoder = Zipformer2(
                samp_freq=16000,
                frame_shift_ms=10,
                frame_length_ms=25,
                feature_dim=16,
                preemph=0.97,
                low_freq=20,
                high_freq=7600,
                min_frames=9,
                subsample_output_dim=encoder_dims[0],
                subsample_layer1_channels=2,
                subsample_layer2_channels=4,
                subsample_layer3_channels=8,
                subsampling_batch_partitions=1,
                encoder_dims=list(encoder_dims),
                num_encoder_layers=[1] * 6,
                downsampling_factors=[1, 2, 4, 8, 4, 2],
                bypass_scales=[torch.ones(dimension) for dimension in encoder_dims],
                num_heads=[2] * 6,
                feedforward_dims=[16, 20, 24, 28, 24, 20],
                cnn_module_kernels=[3] * 6,
                query_head_dim=4,
                pos_head_dim=4,
                value_head_dim=4,
                pos_dim=4,
                pos_max_len=64,
                output_dim=10,
                use_ctc=ctc,
                dtype=dtype,
            ).eval()
            with torch.no_grad():
                for module in encoder.modules():
                    if isinstance(module, BiasNorm):
                        module.scale.fill_(1.0)
                    elif isinstance(module, SimpleDownsample):
                        module.weights.fill_(1.0 / module.weights.size(0))
            encoder_params.update(
                output_dim=10,
                right_padding_samples=200,
                use_ctc=ctc,
                encoder_dims=list(encoder_dims),
                num_encoder_layers=[1] * 6,
                downsampling_factors=[1, 2, 4, 8, 4, 2],
                feedforward_dims=[16, 20, 24, 28, 24, 20],
            )
            decoder = None if ctc else RNNTDecoder(vocab_size, 8, 10, 2, dtype).eval()
            joiner = None if ctc else Joiner(10, vocab_size, dtype).eval()
            head = encoder.projection_output if ctc else joiner.output_proj
            if not ctc:
                decoder_params.update(context_size=2, decoder_dim=8, joiner_dim=10)
        else:
            encoder = make_parakeet_encoder(dtype=dtype, use_ctc=ctc)
            encoder_params.update(model_dim=16, n_layers=1)
            decoder = (
                None if ctc else TDTDecoder(vocab_size, 16, 8, 8, 2, 2, dtype).eval()
            )
            head = encoder.projection_output if ctc else decoder.output_proj
            if not ctc:
                decoder_params.update(
                    encoder_dim=16,
                    decoder_dim=8,
                    joiner_dim=8,
                    pred_rnn_layers=2,
                    num_extra_outputs=2,
                    max_symbols_per_timestep=3,
                    tdt_durations=[0, 1],
                )
        with torch.no_grad():
            head.weight.mul_(0.01)
            head.bias.fill_(-5.0)
            head.bias[2:4] = torch.tensor([3.0, 2.0], dtype=head.bias.dtype)
            if not zipformer and not ctc:
                head.bias[-1] = 5.0
        if zipformer:
            source = OmegaConf.create(
                {
                    "feature_opts": {
                        "frame_opts": {"samp_freq": 16000, "frame_length_ms": 25}
                    },
                    "model_params": {"joiner_dim": 10},
                }
            )
            encoder_path, decoder_path = export_zipformer.export_model_to_onnx(
                encoder, decoder, joiner, source, args
            )
        else:
            source = OmegaConf.create(
                {
                    "sample_rate": 16000,
                    "decoder": {"prednet": {"pred_rnn_layers": 2, "pred_hidden": 8}},
                    "joint": {"jointnet": {"encoder_hidden": 16}},
                }
            )
            encoder_path, decoder_path = export_parakeet.export_model_to_onnx(
                encoder, decoder, source, args
            )
    padding = 200 if zipformer else 0
    build_tensorrt_engine(
        encoder_path,
        encoder_path.with_suffix(".trt"),
        {"audio": tuple((2, length + padding) for length in (1600, 3200, 6400))},
        optimization_level=0,
    )
    if decoder_path is not None:
        build_tensorrt_engine(decoder_path, decoder_path.with_suffix(".trt"), {}, 0)
    config = OmegaConf.create(
        {
            "model_type": f"{family}_asr",
            "decoder_type": args.decoder_type,
            "model_samplerate": 16000,
            "vocab_size": vocab_size,
            "blank_id": 1 if zipformer else vocab_size,
            "audio_encoder_params": encoder_params,
            "decoder_params": decoder_params,
        }
    )
    OmegaConf.save(config, directory / "model_config.yaml")
    return encoder


@pytest.mark.cuda
@pytest.mark.parametrize("family", ("zipformer", "parakeet"))
@pytest.mark.parametrize("ctc", (False, True), ids=("transducer", "ctc"))
@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float16,
        pytest.param(torch.bfloat16, marks=pytest.mark.sm80),
    ),
    ids=("fp32", "fp16", "bf16"),
)
def test_asr_with_real_engines(
    tmp_path: Path, family: str, dtype: torch.dtype, ctc: bool
) -> None:
    reference = build_bundle(tmp_path, family, dtype, ctc)
    model = ASR(tmp_path)
    assert model.encoder.device is model.decoder.device is model.device
    waveforms = [
        0.15 * np.sin(np.arange(length, dtype=np.float32) * 0.07)
        for length in (2400, 3200, 5600)
    ]
    for audios in (waveforms[:2], waveforms[2:], waveforms[:1]):
        result = model(audios)
        texts, word_metadata = result
        for audio, text, words in zip(audios, texts, word_metadata, strict=True):
            assert set(text.split()) == {"alpha"}
            assert words
            assert all(
                0 <= start <= end <= len(audio) / model.encoder.sample_rate
                for _, start, end in words
            )

        lengths = torch.tensor([len(audio) for audio in audios], dtype=torch.int64)
        padded = torch.zeros(len(audios), max(map(len, audios)))
        for row, audio in enumerate(audios):
            padded[row, : len(audio)] = torch.from_numpy(audio)
        if family == "zipformer":
            padded = add_zipformer_right_context(padded, lengths)
        with torch.inference_mode():
            embeddings, valid_lengths = reference(padded, lengths)
        with model.encoder.device, model.stream:
            encoded, encoded_lengths = model.encoder(audios)
            np.testing.assert_array_equal(encoded_lengths.get(), valid_lengths.numpy())
            encoded_host = encoded.astype(cp.float32).get()
        for row, length in enumerate(valid_lengths.tolist()):
            expected = embeddings[row, :length].float().numpy()
            relative_rmse = np.sqrt(
                np.mean((encoded_host[row, :length] - expected) ** 2)
            ) / max(np.sqrt(np.mean(expected**2)), 1e-8)
            assert relative_rmse < (0.02 if dtype == torch.bfloat16 else 0.003)

        assert model(audios) == result
