#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for model-bundle, tokenizer, and TensorRT engine validation."""

from pathlib import Path

import pytest
import tensorrt as trt
import torch
from omegaconf import DictConfig, OmegaConf

import fast_gpu_asr.utils as utils_module
from fast_gpu_asr.utils import (
    ASRInferenceError,
    ASRInitializationError,
    get_engine,
    get_names,
    validate_decoder_engine,
    validate_encoder_engine,
    validate_model,
    validate_model_config,
    validate_tokenizer,
)


def test_utils_defines_exception_classes() -> None:
    """Keep the public exception classes canonical in the utilities module."""

    assert ASRInitializationError.__module__ == "fast_gpu_asr.utils"
    assert ASRInferenceError.__module__ == "fast_gpu_asr.utils"


class FakeEngine:
    """Expose the TensorRT metadata queried by runtime validation."""

    def __init__(
        self,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, trt.DataType],
        profiles: dict[str, tuple[tuple[int, ...], ...]] | None = None,
    ) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.names = input_names + output_names
        self.shapes = shapes
        self.dtypes = dtypes
        self.profiles = profiles or {}
        self.locations = {name: trt.TensorLocation.DEVICE for name in self.names}
        self.formats = {name: trt.TensorFormat.LINEAR for name in self.names}
        self.num_io_tensors = len(self.names)

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_mode(self, name: str) -> trt.TensorIOMode:
        if name in self.input_names:
            return trt.TensorIOMode.INPUT
        return trt.TensorIOMode.OUTPUT

    def get_tensor_profile_shape(
        self,
        name: str,
        profile_index: int,
    ) -> tuple[tuple[int, ...], ...]:
        assert profile_index == 0
        return self.profiles[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> trt.TensorFormat:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> trt.TensorLocation:
        return self.locations[name]


def install_fake_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_path: Path,
    vocab_size: int,
    unk_id: int,
    blank_id: int = 0,
) -> None:
    """Install a minimal SentencePiece fake for tokenizer-validation tests."""

    class FakeTokenizer:
        def __init__(self, model_file: str) -> None:
            assert model_file == str(tokenizer_path)

        def vocab_size(self) -> int:
            return vocab_size

        def unk_id(self) -> int:
            return unk_id

        def piece_to_id(self, piece: str) -> int:
            if piece == "<blk>":
                return blank_id
            return unk_id

        def id_to_piece(self, token_id: int) -> str:
            if token_id == blank_id:
                return "<blk>"
            return "<unk>"

    monkeypatch.setattr(
        utils_module.spm,
        "SentencePieceProcessor",
        FakeTokenizer,
    )


def make_zipformer_config(
    decoder_type: str = "transducer_modified_beam_search",
) -> DictConfig:
    """Return a compact valid Zipformer runtime configuration."""

    ctc = decoder_type == "ctc_greedy_search"
    return OmegaConf.create(
        {
            "model_type": "zipformer_asr",
            "decoder_type": decoder_type,
            "model_samplerate": 16000,
            "vocab_size": 4,
            "blank_id": 0,
            "audio_encoder_params": {
                "feature_dim": 80,
                "output_dim": 4 if ctc else 3,
                "encoder_dims": [16, 24, 32, 40, 32, 24],
                "num_encoder_layers": [1, 1, 1, 1, 1, 1],
                "downsampling_factors": [1, 2, 4, 8, 4, 2],
                "feedforward_dims": [32, 48, 64, 80, 64, 48],
                "frame_shift_ms": 10,
                "min_audio_seconds": 0.1,
                "opt_audio_seconds": 1.0,
                "max_audio_seconds": 2.0,
                "pos_emb_max_len": 128,
                "right_padding_samples": 20,
                "subsampling_factor": 4,
                "use_ctc": ctc,
            },
            "decoder_params": {
                "beam": 1 if "greedy" in decoder_type else 2,
                "blank_penalty": 0.0,
                "context_size": 2,
                "decoder_dim": 3,
                "joiner_dim": 3,
            },
        }
    )


def make_parakeet_config() -> DictConfig:
    """Return a compact valid Parakeet runtime configuration."""

    return OmegaConf.create(
        {
            "model_type": "parakeet_asr",
            "decoder_type": "transducer_modified_beam_search",
            "model_samplerate": 16000,
            "vocab_size": 4,
            "blank_id": 4,
            "audio_encoder_params": {
                "feature_dim": 8,
                "frame_shift_ms": 10,
                "min_audio_seconds": 0.1,
                "opt_audio_seconds": 1.0,
                "max_audio_seconds": 2.0,
                "model_dim": 6,
                "n_layers": 2,
                "pos_emb_max_len": 128,
                "subsampling_factor": 8,
            },
            "decoder_params": {
                "beam": 2,
                "blank_penalty": 0.0,
                "decoder_dim": 5,
                "encoder_dim": 6,
                "joiner_dim": 5,
                "max_symbols_per_timestep": 4,
                "num_extra_outputs": 3,
                "pred_rnn_layers": 2,
                "tdt_durations": [0, 1, 2],
            },
        }
    )


def make_encoder_engine(
    model_config: DictConfig,
    output_dtype: trt.DataType = trt.float16,
    batch_size: int = 2,
) -> FakeEngine:
    """Build matching encoder metadata for either supported architecture."""

    right_padding_samples = (
        model_config.audio_encoder_params.right_padding_samples
        if model_config.model_type == "zipformer_asr"
        else 0
    )
    audio_profile = tuple(
        (
            batch_size,
            round(seconds * model_config.model_samplerate) + right_padding_samples,
        )
        for seconds in (
            model_config.audio_encoder_params.min_audio_seconds,
            model_config.audio_encoder_params.opt_audio_seconds,
            model_config.audio_encoder_params.max_audio_seconds,
        )
    )
    output_dim = (
        model_config.audio_encoder_params.output_dim
        if model_config.model_type == "zipformer_asr"
        else model_config.audio_encoder_params.model_dim
    )
    return FakeEngine(
        ("audio", "audio_lengths"),
        ("encoder_output", "encoder_output_lengths"),
        {
            "audio": (batch_size, -1),
            "audio_lengths": (batch_size,),
            "encoder_output": (batch_size, -1, output_dim),
            "encoder_output_lengths": (batch_size,),
        },
        {
            "audio": trt.float32,
            "audio_lengths": trt.int64,
            "encoder_output": output_dtype,
            "encoder_output_lengths": trt.int32,
        },
        {
            "audio": audio_profile,
            "audio_lengths": ((batch_size,),) * 3,
        },
    )


def make_decoder_engine(
    model_config: DictConfig,
    floating_dtype: trt.DataType = trt.float16,
    batch_size: int = 2,
) -> FakeEngine:
    """Build matching transducer-decoder metadata."""

    capacity = batch_size * model_config.decoder_params.beam
    if model_config.model_type == "zipformer_asr":
        joiner_dim = model_config.decoder_params.joiner_dim
        return FakeEngine(
            ("decoder_input", "encoder_output"),
            ("tokens_log_prob",),
            {
                "decoder_input": (capacity, joiner_dim),
                "encoder_output": (capacity, joiner_dim),
                "tokens_log_prob": (capacity, model_config.vocab_size),
            },
            {
                "decoder_input": floating_dtype,
                "encoder_output": floating_dtype,
                "tokens_log_prob": trt.float32,
            },
        )

    decoder_dim = model_config.decoder_params.decoder_dim
    state_shape = (
        model_config.decoder_params.pred_rnn_layers,
        capacity,
        decoder_dim,
    )
    return FakeEngine(
        ("encoder_output", "targets", "input_states_1", "input_states_2"),
        (
            "token_log_probs",
            "duration_log_probs",
            "output_states_1",
            "output_states_2",
        ),
        {
            "encoder_output": (capacity, model_config.decoder_params.encoder_dim),
            "targets": (capacity, 1),
            "input_states_1": state_shape,
            "input_states_2": state_shape,
            "token_log_probs": (capacity, model_config.vocab_size + 1),
            "duration_log_probs": (
                capacity,
                model_config.decoder_params.num_extra_outputs,
            ),
            "output_states_1": state_shape,
            "output_states_2": state_shape,
        },
        {
            "encoder_output": floating_dtype,
            "targets": trt.int32,
            "input_states_1": floating_dtype,
            "input_states_2": floating_dtype,
            "token_log_probs": trt.float32,
            "duration_log_probs": trt.float32,
            "output_states_1": floating_dtype,
            "output_states_2": floating_dtype,
        },
    )


COMMON_REQUIRED_FIELDS = (
    "model_type",
    "decoder_type",
    "model_samplerate",
    "vocab_size",
    "blank_id",
    "audio_encoder_params.feature_dim",
    "audio_encoder_params.frame_shift_ms",
    "audio_encoder_params.min_audio_seconds",
    "audio_encoder_params.opt_audio_seconds",
    "audio_encoder_params.max_audio_seconds",
    "audio_encoder_params.pos_emb_max_len",
    "audio_encoder_params.subsampling_factor",
    "decoder_params.beam",
    "decoder_params.blank_penalty",
)
PARAKEET_REQUIRED_FIELDS = (
    "audio_encoder_params.model_dim",
    "audio_encoder_params.n_layers",
    "decoder_params.decoder_dim",
    "decoder_params.encoder_dim",
    "decoder_params.joiner_dim",
    "decoder_params.max_symbols_per_timestep",
    "decoder_params.num_extra_outputs",
    "decoder_params.pred_rnn_layers",
    "decoder_params.tdt_durations",
)
ZIPFORMER_REQUIRED_FIELDS = (
    "audio_encoder_params.downsampling_factors",
    "audio_encoder_params.encoder_dims",
    "audio_encoder_params.feedforward_dims",
    "audio_encoder_params.num_encoder_layers",
    "audio_encoder_params.output_dim",
    "audio_encoder_params.right_padding_samples",
    "audio_encoder_params.use_ctc",
)
ZIPFORMER_TRANSDUCER_REQUIRED_FIELDS = (
    "decoder_params.context_size",
    "decoder_params.decoder_dim",
    "decoder_params.joiner_dim",
)


@pytest.mark.parametrize(
    ("architecture", "field"),
    (
        *(("zipformer", field) for field in COMMON_REQUIRED_FIELDS),
        *(("parakeet", field) for field in COMMON_REQUIRED_FIELDS),
        *(("parakeet", field) for field in PARAKEET_REQUIRED_FIELDS),
        *(("zipformer", field) for field in ZIPFORMER_REQUIRED_FIELDS),
        *(("zipformer", field) for field in ZIPFORMER_TRANSDUCER_REQUIRED_FIELDS),
    ),
    ids=lambda value: str(value).replace(".", "-"),
)
def test_validate_model_config_reports_missing_required_field(
    architecture: str, field: str
) -> None:
    model_config = (
        make_parakeet_config()
        if architecture == "parakeet"
        else make_zipformer_config()
    )
    parent_path, _, field_name = field.rpartition(".")
    parent = (
        OmegaConf.select(model_config, parent_path) if parent_path else model_config
    )
    del parent[field_name]

    with pytest.raises(ASRInitializationError) as error:
        validate_model_config(model_config)
    assert str(error.value) == f"Missing required model configuration field {field}."


def test_validate_model_config_reports_mandatory_missing_value() -> None:
    model_config = make_parakeet_config()
    model_config.decoder_params.joiner_dim = "???"

    with pytest.raises(
        ASRInitializationError,
        match=r"Missing required model configuration field decoder_params\.joiner_dim",
    ):
        validate_model_config(model_config)


def test_validate_model_config_accepts_ctc_without_transducer_decoder_fields() -> None:
    model_config = make_zipformer_config("ctc_greedy_search")
    for field in ZIPFORMER_TRANSDUCER_REQUIRED_FIELDS:
        field_name = field.removeprefix("decoder_params.")
        del model_config.decoder_params[field_name]

    validate_model_config(model_config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_type", "${missing_model_type}"),
        ("decoder_params.joiner_dim", "${missing_joiner_dim}"),
    ),
)
def test_validate_model_config_wraps_unresolved_interpolation(
    field: str,
    value: str,
) -> None:
    model_config = make_parakeet_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(
        ASRInitializationError,
        match="Failed to resolve model configuration",
    ) as error:
        validate_model_config(model_config)

    assert error.value.__cause__ is not None


def test_validate_model_config_wraps_interpolation_cycle() -> None:
    model_config = make_parakeet_config()
    model_config.cycle_a = "${cycle_b}"
    model_config.cycle_b = "${cycle_a}"

    with pytest.raises(
        ASRInitializationError,
        match="Failed to resolve model configuration",
    ) as error:
        validate_model_config(model_config)

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "blank_penalty",
    (
        -float(torch.finfo(torch.float32).max),
        -1.0,
        0.0,
        1.0,
        float(torch.finfo(torch.float32).max),
    ),
)
def test_validate_model_config_accepts_float32_blank_penalty(
    blank_penalty: float,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.blank_penalty = blank_penalty

    validate_model_config(model_config)


@pytest.mark.parametrize(
    "blank_penalty",
    (
        -float(torch.finfo(torch.float32).max) * 2.0,
        float(torch.finfo(torch.float32).max) * 2.0,
    ),
)
def test_validate_model_config_rejects_blank_penalty_outside_float32(
    blank_penalty: float,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.blank_penalty = blank_penalty

    with pytest.raises(ASRInitializationError, match="finite float32 value"):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    "blank_penalty",
    (float("-inf"), float("inf"), float("nan"), 0, "0.0"),
)
def test_validate_model_config_rejects_non_float32_blank_penalty(
    blank_penalty: object,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.blank_penalty = blank_penalty

    with pytest.raises(ASRInitializationError, match="finite float32 value"):
        validate_model_config(model_config)


@pytest.mark.parametrize("model_config", ({}, [], None))
def test_validate_model_config_rejects_non_dictconfig(model_config: object) -> None:
    with pytest.raises(ASRInitializationError, match="Expected model_config"):
        validate_model_config(model_config)  # type: ignore[arg-type]


def test_get_engine_loads_native_encoder_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "zipformer.trt"
    engine_path.write_bytes(b"engine")
    engine = object()
    plugin_calls = 0

    class FakeRuntime:
        def __init__(self, logger: object) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> object:
            assert serialized_engine == b"engine"
            return engine

    def load_plugins() -> None:
        nonlocal plugin_calls
        plugin_calls += 1

    monkeypatch.setattr(utils_module, "load_tensorrt_plugins", load_plugins)
    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: True)
    monkeypatch.setattr(utils_module.trt, "Runtime", FakeRuntime)

    assert get_engine(engine_path) is engine
    assert plugin_calls == 1


def test_get_engine_does_not_load_native_plugins_for_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"engine")
    monkeypatch.setattr(
        utils_module,
        "load_tensorrt_plugins",
        lambda: pytest.fail("Decoder engine unexpectedly loaded encoder plugins."),
    )
    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: True)
    monkeypatch.setattr(
        utils_module.trt,
        "Runtime",
        lambda _: type(
            "FakeRuntime",
            (),
            {"deserialize_cuda_engine": lambda self, data: object()},
        )(),
    )

    assert get_engine(engine_path) is not None


@pytest.mark.parametrize("deserialize_result", (None, RuntimeError("invalid plan")))
def test_get_engine_reports_deserialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deserialize_result: None | RuntimeError,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"engine")

    class FakeRuntime:
        def __init__(self, logger: object) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> None:
            if isinstance(deserialize_result, RuntimeError):
                raise deserialize_result
            return None

    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: True)
    monkeypatch.setattr(utils_module.trt, "Runtime", FakeRuntime)

    with pytest.raises(ASRInitializationError, match="Failed to deserialize"):
        get_engine(engine_path)


def test_get_engine_wraps_runtime_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"engine")
    failure = RuntimeError("runtime unavailable")

    def make_runtime(_: object) -> None:
        raise failure

    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: True)
    monkeypatch.setattr(utils_module.trt, "Runtime", make_runtime)

    with pytest.raises(ASRInitializationError, match="Failed to deserialize") as error:
        get_engine(engine_path)

    assert error.value.__cause__ is failure


def test_get_engine_wraps_file_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: True)
    monkeypatch.setattr(
        utils_module.trt,
        "Runtime",
        lambda _: type(
            "FakeRuntime",
            (),
            {"deserialize_cuda_engine": lambda self, data: object()},
        )(),
    )

    with pytest.raises(ASRInitializationError, match="Failed to deserialize") as error:
        get_engine(engine_path)

    assert isinstance(error.value.__cause__, OSError)


def test_get_engine_reports_plugin_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.touch()
    monkeypatch.setattr(utils_module.trt, "init_libnvinfer_plugins", lambda *_: False)

    with pytest.raises(ASRInitializationError, match="initialize TensorRT plugins"):
        get_engine(engine_path)


def test_get_names_preserves_engine_order() -> None:
    engine = FakeEngine(
        ("second_input", "first_input"),
        ("second_output", "first_output"),
        {},
        {},
    )

    assert get_names(engine) == (
        ("second_input", "first_input"),
        ("second_output", "first_output"),
    )


def test_get_names_rejects_unsupported_tensor_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine(("audio",), (), {"audio": (1, -1)}, {"audio": trt.float32})
    monkeypatch.setattr(engine, "get_tensor_mode", lambda _: trt.TensorIOMode.NONE)

    with pytest.raises(ASRInitializationError, match="unsupported I/O mode"):
        get_names(engine)


def test_validate_tokenizer_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ASRInitializationError, match="Missing SentencePiece tokenizer"):
        validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_parakeet_tokenizer_uses_full_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    install_fake_tokenizer(monkeypatch, tokenizer_path, vocab_size=4, unk_id=3)

    validate_tokenizer(tmp_path, make_parakeet_config())


def test_validate_tokenizer_reports_vocabulary_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    install_fake_tokenizer(monkeypatch, tokenizer_path, vocab_size=3, unk_id=2)

    with pytest.raises(ASRInitializationError, match="tokenizer vocabulary size 4"):
        validate_tokenizer(tmp_path, make_parakeet_config())


@pytest.mark.parametrize(
    ("tokenizer_vocab_size", "unk_id"),
    (
        (5, 4),
        (4, 1),
    ),
)
def test_validate_zipformer_tokenizer_accepts_supported_vocab_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_vocab_size: int,
    unk_id: int,
) -> None:
    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    install_fake_tokenizer(
        monkeypatch,
        tokenizer_path,
        vocab_size=tokenizer_vocab_size,
        unk_id=unk_id,
    )

    validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_zipformer_tokenizer_requires_blank_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    install_fake_tokenizer(
        monkeypatch,
        tokenizer_path,
        vocab_size=5,
        unk_id=4,
        blank_id=4,
    )

    with pytest.raises(ASRInitializationError, match="in-vocabulary <blk> token"):
        validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_tokenizer_reports_loader_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "bpe.model").touch()

    def fail_to_load(*args: object, **kwargs: object) -> None:
        raise RuntimeError("invalid tokenizer")

    monkeypatch.setattr(
        utils_module.spm,
        "SentencePieceProcessor",
        fail_to_load,
    )

    with pytest.raises(ASRInitializationError, match="Failed to load SentencePiece"):
        validate_tokenizer(tmp_path, make_parakeet_config())


@pytest.mark.parametrize("output_dtype", (trt.float32, trt.float16, trt.bfloat16))
@pytest.mark.parametrize(
    "model_config",
    (make_zipformer_config(), make_parakeet_config()),
)
def test_validate_encoder_engine_accepts_supported_output_precision(
    model_config: DictConfig,
    output_dtype: trt.DataType,
) -> None:
    assert (
        validate_encoder_engine(
            make_encoder_engine(model_config, output_dtype),
            model_config,
        )
        == 2
    )


@pytest.mark.parametrize(
    ("tensor_name", "dtype", "message"),
    (
        ("audio", trt.float16, "encoder audio dtype"),
        ("audio_lengths", trt.int32, "encoder audio_lengths dtype"),
        ("encoder_output", trt.int32, "encoder_output dtype"),
        (
            "encoder_output_lengths",
            trt.int64,
            "encoder_output_lengths dtype",
        ),
    ),
)
def test_validate_encoder_engine_rejects_invalid_dtype(
    tensor_name: str,
    dtype: trt.DataType,
    message: str,
) -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    engine.dtypes[tensor_name] = dtype

    with pytest.raises(ASRInitializationError, match=message):
        validate_encoder_engine(engine, model_config)


def test_validate_encoder_engine_rejects_missing_io_name() -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    engine.input_names = ("audio",)
    engine.names = engine.input_names + engine.output_names
    engine.num_io_tensors = len(engine.names)

    with pytest.raises(ASRInitializationError, match="Expected encoder inputs"):
        validate_encoder_engine(engine, model_config)


def test_validate_encoder_engine_rejects_audio_profile_mismatch() -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    min_shape, opt_shape, max_shape = engine.profiles["audio"]
    engine.profiles["audio"] = (
        (min_shape[0], min_shape[1] + 1),
        opt_shape,
        max_shape,
    )

    with pytest.raises(ASRInitializationError, match="encoder audio profile"):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize(
    ("metadata", "value", "message"),
    (
        ("locations", trt.TensorLocation.HOST, "reside on the GPU"),
        ("formats", trt.TensorFormat.CHW32, "linear storage"),
    ),
)
def test_validate_encoder_engine_rejects_incompatible_io_layout(
    metadata: str,
    value: trt.TensorLocation | trt.TensorFormat,
    message: str,
) -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    getattr(engine, metadata)["encoder_output"] = value

    with pytest.raises(ASRInitializationError, match=message):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize(
    "model_config",
    (make_zipformer_config(), make_parakeet_config()),
)
@pytest.mark.parametrize("floating_dtype", (trt.float32, trt.float16, trt.bfloat16))
def test_validate_decoder_engine_accepts_supported_precision(
    model_config: DictConfig,
    floating_dtype: trt.DataType,
) -> None:
    validate_decoder_engine(
        make_decoder_engine(model_config, floating_dtype),
        model_config,
        2,
    )


def test_validate_parakeet_decoder_rejects_mixed_floating_precision() -> None:
    model_config = make_parakeet_config()
    engine = make_decoder_engine(model_config)
    engine.dtypes["output_states_2"] = trt.float32

    with pytest.raises(ASRInitializationError, match="share an FP32, FP16, or BF16"):
        validate_decoder_engine(engine, model_config, 2)


def test_validate_zipformer_decoder_rejects_mixed_floating_precision() -> None:
    model_config = make_zipformer_config()
    engine = make_decoder_engine(model_config)
    engine.dtypes["encoder_output"] = trt.float32

    with pytest.raises(ASRInitializationError, match="share an FP32, FP16, or BF16"):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize(
    ("tensor_name", "dtype", "message"),
    (
        ("targets", trt.int64, "targets dtype"),
        ("token_log_probs", trt.float16, "token_log_probs dtype"),
        ("duration_log_probs", trt.float16, "duration_log_probs dtype"),
    ),
)
def test_validate_parakeet_decoder_rejects_boundary_dtype(
    tensor_name: str,
    dtype: trt.DataType,
    message: str,
) -> None:
    model_config = make_parakeet_config()
    engine = make_decoder_engine(model_config)
    engine.dtypes[tensor_name] = dtype

    with pytest.raises(ASRInitializationError, match=message):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize(
    ("tensor_name", "dtype", "message"),
    (
        ("decoder_input", trt.int32, "share an FP32, FP16, or BF16"),
        ("tokens_log_prob", trt.float16, "tokens_log_prob dtype"),
    ),
)
def test_validate_zipformer_decoder_rejects_boundary_dtype(
    tensor_name: str,
    dtype: trt.DataType,
    message: str,
) -> None:
    model_config = make_zipformer_config()
    engine = make_decoder_engine(model_config)
    engine.dtypes[tensor_name] = dtype

    with pytest.raises(ASRInitializationError, match=message):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize("batch_size", (0, -1, 1.5))
def test_validate_decoder_engine_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    model_config = make_zipformer_config()

    with pytest.raises(ASRInitializationError, match="decoder batch size"):
        validate_decoder_engine(
            make_decoder_engine(model_config),
            model_config,
            batch_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "model_config",
    (make_zipformer_config(), make_parakeet_config()),
)
def test_validate_decoder_engine_rejects_missing_output_name(
    model_config: DictConfig,
) -> None:
    engine = make_decoder_engine(model_config)
    engine.output_names = ()
    engine.names = engine.input_names
    engine.num_io_tensors = len(engine.names)

    with pytest.raises(ASRInitializationError, match="Expected decoder outputs"):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize(
    ("model_config", "tensor_name"),
    (
        (make_zipformer_config(), "tokens_log_prob"),
        (make_parakeet_config(), "token_log_probs"),
    ),
)
def test_validate_decoder_engine_rejects_shape_mismatch(
    model_config: DictConfig,
    tensor_name: str,
) -> None:
    engine = make_decoder_engine(model_config)
    expected_shape = engine.shapes[tensor_name]
    engine.shapes[tensor_name] = (expected_shape[0] + 1, *expected_shape[1:])

    with pytest.raises(ASRInitializationError, match=f"{tensor_name} shape"):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize(
    ("model_config", "encoder_filename", "decoder_filename"),
    (
        (make_zipformer_config(), "zipformer.trt", "decoder.trt"),
        (make_parakeet_config(), "parakeet.trt", "tdt_decoder.trt"),
    ),
)
def test_validate_model_reports_missing_transducer_decoder_before_loading_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_config: DictConfig,
    encoder_filename: str,
    decoder_filename: str,
) -> None:
    (tmp_path / encoder_filename).touch()
    monkeypatch.setattr(utils_module, "validate_tokenizer", lambda *_: None)
    monkeypatch.setattr(
        utils_module,
        "get_engine",
        lambda _: pytest.fail("Engine loaded before artifact preflight completed."),
    )

    with pytest.raises(ASRInitializationError, match=decoder_filename):
        validate_model(tmp_path, model_config)


def test_validate_model_accepts_complete_zipformer_ctc_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_zipformer_config("ctc_greedy_search")
    (tmp_path / "zipformer.trt").touch()
    encoder = make_encoder_engine(model_config)
    validated: list[str] = []
    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        utils_module,
        "validate_tokenizer",
        lambda *_: validated.append("tokenizer"),
    )

    def load_engine(engine_path: Path) -> FakeEngine:
        loaded_paths.append(engine_path)
        return encoder

    monkeypatch.setattr(utils_module, "get_engine", load_engine)

    validate_model(tmp_path, model_config)

    assert validated == ["tokenizer"]
    assert loaded_paths == [tmp_path / "zipformer.trt"]


def test_validate_model_accepts_complete_zipformer_transducer_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_zipformer_config()
    (tmp_path / "zipformer.trt").touch()
    (tmp_path / "decoder.trt").touch()
    engines = {
        tmp_path / "zipformer.trt": make_encoder_engine(model_config),
        tmp_path / "decoder.trt": make_decoder_engine(model_config),
    }
    loaded_paths: list[Path] = []
    context_validated = False
    monkeypatch.setattr(utils_module, "validate_tokenizer", lambda *_: None)

    def load_engine(engine_path: Path) -> FakeEngine:
        loaded_paths.append(engine_path)
        return engines[engine_path]

    monkeypatch.setattr(utils_module, "get_engine", load_engine)

    def validate_context(*args: object) -> None:
        nonlocal context_validated
        context_validated = True

    monkeypatch.setattr(
        utils_module,
        "validate_zipformer_context_lookup",
        validate_context,
    )

    validate_model(tmp_path, model_config)

    assert loaded_paths == [tmp_path / "zipformer.trt", tmp_path / "decoder.trt"]
    assert context_validated


def test_validate_model_accepts_complete_parakeet_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_parakeet_config()
    (tmp_path / "parakeet.trt").touch()
    (tmp_path / "tdt_decoder.trt").touch()
    engines = {
        tmp_path / "parakeet.trt": make_encoder_engine(model_config),
        tmp_path / "tdt_decoder.trt": make_decoder_engine(model_config),
    }
    loaded_paths: list[Path] = []
    monkeypatch.setattr(utils_module, "validate_tokenizer", lambda *_: None)

    def load_engine(engine_path: Path) -> FakeEngine:
        loaded_paths.append(engine_path)
        return engines[engine_path]

    monkeypatch.setattr(utils_module, "get_engine", load_engine)
    monkeypatch.setattr(
        utils_module,
        "validate_zipformer_context_lookup",
        lambda *_: pytest.fail("Parakeet unexpectedly validated a context lookup."),
    )

    validate_model(tmp_path, model_config)

    assert loaded_paths == [tmp_path / "parakeet.trt", tmp_path / "tdt_decoder.trt"]
