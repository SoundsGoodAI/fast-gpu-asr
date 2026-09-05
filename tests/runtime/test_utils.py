#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for model-bundle, tokenizer, and TensorRT engine validation."""

from contextlib import nullcontext
from pathlib import Path
from pickle import UnpicklingError
from typing import cast

import pytest
import tensorrt as trt
import torch
from omegaconf import DictConfig, OmegaConf

import fast_gpu_asr.utils as utils_module
from fast_gpu_asr.constants import INT32_MAX, ZIPFORMER_DECODER_CONTEXTS_FILE
from fast_gpu_asr.utils import (
    ASRInitializationError,
    get_engine,
    get_names,
    validate_decoder_engine,
    validate_encoder_engine,
    validate_model,
    validate_model_config,
    validate_tokenizer,
    validate_zipformer_context_lookup,
)


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
        """Initialize configurable TensorRT engine metadata.

        Parameters
        ----------
        input_names : tuple[str, ...]
            Tensor names reported as engine inputs.
        output_names : tuple[str, ...]
            Tensor names reported as engine outputs.
        shapes : dict[str, tuple[int, ...]]
            Static or dynamic engine shapes indexed by tensor name.
        dtypes : dict[str, trt.DataType]
            TensorRT dtypes indexed by tensor name.
        profiles : dict[str, tuple[tuple[int, ...], ...]] | None
            Optional minimum, optimum, and maximum profile shapes indexed by
            dynamic input name.
        """

        self.input_names = input_names
        self.output_names = output_names
        self.names = input_names + output_names
        self.shapes = shapes
        self.dtypes = dtypes
        self.profiles = profiles or {}
        self.num_io_tensors = len(self.names)

    def get_tensor_name(self, index: int) -> str:
        """Return the tensor name at one engine I/O index.

        Parameters
        ----------
        index : int
            Zero-based engine I/O index.

        Returns
        -------
        str
            Tensor name stored at ``index``.
        """

        return self.names[index]

    def get_tensor_mode(self, name: str) -> trt.TensorIOMode:
        """Return whether a named tensor is an input or output.

        Parameters
        ----------
        name : str
            Engine tensor name.

        Returns
        -------
        trt.TensorIOMode
            ``INPUT`` for configured inputs and ``OUTPUT`` otherwise.
        """

        if name in self.input_names:
            return trt.TensorIOMode.INPUT
        return trt.TensorIOMode.OUTPUT

    def get_tensor_profile_shape(
        self,
        name: str,
        profile_index: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return dynamic profile shapes for a named input.

        Parameters
        ----------
        name : str
            Dynamic engine input name.
        profile_index : int
            Optimization-profile index; this fake supports only profile zero.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Minimum, optimum, and maximum shapes configured for ``name``.
        """

        assert profile_index == 0
        return self.profiles[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return the configured engine shape for a tensor.

        Parameters
        ----------
        name : str
            Engine tensor name.

        Returns
        -------
        tuple[int, ...]
            Shape associated with ``name``.
        """

        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        """Return the configured TensorRT dtype for a tensor.

        Parameters
        ----------
        name : str
            Engine tensor name.

        Returns
        -------
        trt.DataType
            Dtype associated with ``name``.
        """

        return self.dtypes[name]


def install_fake_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: Path,
    vocab_size: int,
    unk_id: int,
    blank_id: int = 0,
    blank_piece: str = "<blk>",
    standalone_id: int = 2,
    standalone_piece: str = "▁",
) -> None:
    """Install a configurable SentencePiece fake for tokenizer validation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the SentencePiece processor constructor.
    model_dir : Path
        Bundle directory in which a placeholder tokenizer is created.
    vocab_size : int
        Vocabulary size reported by the fake tokenizer.
    unk_id : int
        Unknown-token ID reported by the fake tokenizer.
    blank_id : int
        Token ID returned for ``<blk>``.
    blank_piece : str
        Piece surface returned for ``blank_id``.
    standalone_id : int
        Token ID returned for the standalone metaspace marker.
    standalone_piece : str
        Piece surface returned for ``standalone_id``.
    """

    tokenizer_path = model_dir / "bpe.model"
    tokenizer_path.touch()

    class FakeTokenizer:
        """Expose tokenizer metadata captured from the enclosing helper."""

        def __init__(self, model_file: str) -> None:
            """Validate the tokenizer path supplied by production code.

            Parameters
            ----------
            model_file : str
                SentencePiece model path requested by ``validate_tokenizer``.
            """

            assert model_file == str(tokenizer_path)

        def vocab_size(self) -> int:
            """Return the configured vocabulary size.

            Returns
            -------
            int
                Number of tokenizer pieces.
            """

            return vocab_size

        def unk_id(self) -> int:
            """Return the configured unknown-token ID.

            Returns
            -------
            int
                Unknown-token ID.
            """

            return unk_id

        def piece_to_id(self, piece: str) -> int:
            """Map validation-sensitive pieces to configured token IDs.

            Parameters
            ----------
            piece : str
                SentencePiece surface to resolve.

            Returns
            -------
            int
                Blank, standalone-marker, or unknown-token ID.
            """

            if piece == "<blk>":
                return blank_id
            if piece == "▁":
                return standalone_id
            return unk_id

        def id_to_piece(self, token_id: int) -> str:
            """Map validation-sensitive token IDs to configured pieces.

            Parameters
            ----------
            token_id : int
                Token ID to resolve.

            Returns
            -------
            str
                Blank, standalone-marker, or unknown-token surface.
            """

            if token_id == blank_id:
                return blank_piece
            if token_id == standalone_id:
                return standalone_piece
            return "<unk>"

    monkeypatch.setattr(
        utils_module.spm,
        "SentencePieceProcessor",
        FakeTokenizer,
    )


def make_zipformer_config(
    decoder_type: str = "transducer_modified_beam_search",
) -> DictConfig:
    """Return a compact valid Zipformer runtime configuration.

    Parameters
    ----------
    decoder_type : str
        Decoder mode stored in the generated model metadata.

    Returns
    -------
    DictConfig
        Valid Zipformer bundle configuration for runtime validation tests.
    """

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
    """Return a compact valid Parakeet runtime configuration.

    Returns
    -------
    DictConfig
        Valid Parakeet bundle configuration for runtime validation tests.
    """

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


def make_model_config(architecture: str) -> DictConfig:
    """Return a valid configuration for the requested architecture.

    Parameters
    ----------
    architecture : str
        Test architecture selector; ``parakeet`` selects Parakeet and every
        other value selects Zipformer.

    Returns
    -------
    DictConfig
        Valid runtime configuration for the selected architecture.
    """

    if architecture == "parakeet":
        return make_parakeet_config()
    return make_zipformer_config()


def make_encoder_engine(
    model_config: DictConfig,
    output_dtype: trt.DataType = trt.float16,
    batch_size: int = 2,
) -> FakeEngine:
    """Build matching encoder metadata for either supported architecture.

    Parameters
    ----------
    model_config : DictConfig
        Zipformer or Parakeet runtime configuration defining tensor dimensions
        and the audio profile.
    output_dtype : trt.DataType
        TensorRT dtype reported for ``encoder_output``.
    batch_size : int
        Fixed batch dimension used by engine tensors and profiles.

    Returns
    -------
    FakeEngine
        Encoder metadata satisfying the supplied model configuration.
    """

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
    """Build matching transducer-decoder metadata.

    Parameters
    ----------
    model_config : DictConfig
        Zipformer or Parakeet runtime configuration defining decoder tensors.
    floating_dtype : trt.DataType
        TensorRT dtype reported for model-precision decoder tensors.
    batch_size : int
        Fixed encoder batch size used to derive decoder capacity.

    Returns
    -------
    FakeEngine
        Decoder metadata satisfying the supplied model configuration.
    """

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
TRANSDUCER_MODEL_FILES = {
    "zipformer": ("zipformer.trt", "decoder.trt"),
    "parakeet": ("parakeet.trt", "tdt_decoder.trt"),
}
DECODER_TENSOR_RANKS = {
    "zipformer": {
        "decoder_input": 2,
        "encoder_output": 2,
        "tokens_log_prob": 2,
    },
    "parakeet": {
        "encoder_output": 2,
        "targets": 2,
        "input_states_1": 3,
        "input_states_2": 3,
        "token_log_probs": 2,
        "duration_log_probs": 2,
        "output_states_1": 3,
        "output_states_2": 3,
    },
}
DECODER_PRECISION_TENSORS = {
    "zipformer": ("decoder_input", "encoder_output"),
    "parakeet": (
        "encoder_output",
        "input_states_1",
        "input_states_2",
        "output_states_1",
        "output_states_2",
    ),
}
COMMON_POSITIVE_INTEGER_FIELDS = (
    "model_samplerate",
    "vocab_size",
    "audio_encoder_params.frame_shift_ms",
    "audio_encoder_params.pos_emb_max_len",
    "audio_encoder_params.subsampling_factor",
    "decoder_params.beam",
)
POSITIVE_INTEGER_FIELDS = {
    "zipformer": (
        *COMMON_POSITIVE_INTEGER_FIELDS,
        "audio_encoder_params.feature_dim",
        "audio_encoder_params.output_dim",
        "decoder_params.context_size",
        "decoder_params.decoder_dim",
        "decoder_params.joiner_dim",
        *(
            f"audio_encoder_params.{field}.{index}"
            for field in (
                "encoder_dims",
                "num_encoder_layers",
                "downsampling_factors",
                "feedforward_dims",
            )
            for index in range(6)
        ),
    ),
    "parakeet": (
        *COMMON_POSITIVE_INTEGER_FIELDS,
        "audio_encoder_params.feature_dim",
        "audio_encoder_params.model_dim",
        "audio_encoder_params.n_layers",
        "decoder_params.decoder_dim",
        "decoder_params.encoder_dim",
        "decoder_params.joiner_dim",
        "decoder_params.max_symbols_per_timestep",
        "decoder_params.num_extra_outputs",
        "decoder_params.pred_rnn_layers",
    ),
}


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
    model_config = make_model_config(architecture)
    parent_path, _, field_name = field.rpartition(".")
    parent = (
        OmegaConf.select(model_config, parent_path) if parent_path else model_config
    )
    del parent[field_name]

    with pytest.raises(ASRInitializationError) as error:
        validate_model_config(model_config)
    assert str(error.value) == f"Missing required model configuration field {field}."


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_type", None),
        ("decoder_params.joiner_dim", None),
        ("decoder_params.joiner_dim", "???"),
    ),
)
def test_validate_model_config_reports_mandatory_missing_value(
    field: str,
    value: str | None,
) -> None:
    model_config = make_parakeet_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ASRInitializationError) as error:
        validate_model_config(model_config)

    assert str(error.value) == f"Missing required model configuration field {field}."


@pytest.mark.parametrize(
    ("architecture", "decoder_type"),
    (
        ("zipformer", "transducer_modified_beam_search"),
        ("zipformer", "transducer_greedy_search"),
        ("zipformer", "ctc_greedy_search"),
        ("parakeet", "transducer_modified_beam_search"),
        ("parakeet", "transducer_greedy_search"),
    ),
)
def test_validate_model_config_accepts_supported_decoder_modes(
    architecture: str,
    decoder_type: str,
) -> None:
    if architecture == "zipformer":
        model_config = make_zipformer_config(decoder_type)
    else:
        model_config = make_parakeet_config()
        model_config.decoder_type = decoder_type
        model_config.decoder_params.beam = (
            1 if decoder_type == "transducer_greedy_search" else 2
        )

    validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "updates"),
    (
        pytest.param(
            "zipformer",
            {"audio_encoder_params.encoder_dims": [16] * 6},
            id="equal-zipformer-stack-dimensions",
        ),
        pytest.param(
            "zipformer",
            {
                "audio_encoder_params.min_audio_seconds": 1.0,
                "audio_encoder_params.opt_audio_seconds": 1.0,
                "audio_encoder_params.max_audio_seconds": 1.0,
            },
            id="equal-zipformer-audio-profile",
        ),
        pytest.param(
            "parakeet",
            {
                "audio_encoder_params.min_audio_seconds": 1.0,
                "audio_encoder_params.opt_audio_seconds": 1.0,
                "audio_encoder_params.max_audio_seconds": 1.0,
            },
            id="equal-parakeet-audio-profile",
        ),
        pytest.param(
            "zipformer",
            {"decoder_params.beam": 4},
            id="zipformer-beam-equals-vocabulary",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.beam": 4},
            id="parakeet-beam-equals-vocabulary",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.right_padding_samples": 0},
            id="zero-right-padding",
        ),
    ),
)
def test_validate_model_config_accepts_inclusive_boundaries(
    architecture: str,
    updates: dict[str, int | float | list[int]],
) -> None:
    model_config = make_model_config(architecture)
    for field, value in updates.items():
        OmegaConf.update(model_config, field, value)

    validate_model_config(model_config)


def test_validate_model_config_accepts_ctc_without_transducer_decoder_fields() -> None:
    model_config = make_zipformer_config("ctc_greedy_search")
    for field in ZIPFORMER_TRANSDUCER_REQUIRED_FIELDS:
        field_name = field.removeprefix("decoder_params.")
        del model_config.decoder_params[field_name]

    validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "field", "value", "message"),
    (
        ("zipformer", "model_type", "unknown", "Unsupported model_type"),
        (
            "zipformer",
            "decoder_type",
            "unknown",
            "Expected decoder_type to be one of",
        ),
        (
            "parakeet",
            "decoder_type",
            "ctc_greedy_search",
            "Parakeet TDT models do not contain a CTC head",
        ),
    ),
)
def test_validate_model_config_rejects_unsupported_modes(
    architecture: str,
    field: str,
    value: str,
    message: str,
) -> None:
    model_config = make_model_config(architecture)
    model_config[field] = value

    with pytest.raises(ASRInitializationError, match=message):
        validate_model_config(model_config)


def test_validate_model_config_wraps_unresolved_interpolation() -> None:
    model_config = make_parakeet_config()
    model_config.model_type = "${missing_model_type}"

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
        0.0,
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
    (float("inf"), float("nan"), 0),
)
def test_validate_model_config_rejects_non_float32_blank_penalty(
    blank_penalty: float | int,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.blank_penalty = blank_penalty

    with pytest.raises(ASRInitializationError, match="finite float32 value"):
        validate_model_config(model_config)


def test_validate_model_config_rejects_non_dictconfig() -> None:
    with pytest.raises(ASRInitializationError, match="Expected model_config"):
        validate_model_config(cast(DictConfig, {}))


@pytest.mark.parametrize(
    ("architecture", "field"),
    tuple(
        (architecture, field)
        for architecture, fields in POSITIVE_INTEGER_FIELDS.items()
        for field in fields
    ),
    ids=lambda value: str(value).replace(".", "-"),
)
def test_validate_model_config_rejects_zero_for_every_positive_integer(
    architecture: str,
    field: str,
) -> None:
    model_config = make_model_config(architecture)
    OmegaConf.update(model_config, field, 0)

    with pytest.raises(ASRInitializationError, match="positive integer"):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "field", "value"),
    (
        ("parakeet", "audio_encoder_params.n_layers", INT32_MAX + 1),
        ("zipformer", "audio_encoder_params.feedforward_dims.5", -1),
        ("parakeet", "audio_encoder_params.n_layers", 1.5),
        ("zipformer", "audio_encoder_params.feedforward_dims.5", 1.5),
    ),
)
def test_validate_model_config_rejects_invalid_integer_values(
    architecture: str,
    field: str,
    value: int | float,
) -> None:
    model_config = make_model_config(architecture)
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ASRInitializationError, match="positive integer"):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    "field",
    (
        "encoder_dims",
        "num_encoder_layers",
        "downsampling_factors",
        "feedforward_dims",
    ),
)
def test_validate_model_config_requires_six_zipformer_stack_values(
    field: str,
) -> None:
    model_config = make_zipformer_config()
    model_config.audio_encoder_params[field] = [1] * 5

    with pytest.raises(
        ASRInitializationError,
        match=rf"audio_encoder_params\.{field}.*six positive integers",
    ):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "field"),
    (
        ("zipformer", "audio_encoder_params.encoder_dims"),
        ("zipformer", "audio_encoder_params.num_encoder_layers"),
        ("zipformer", "audio_encoder_params.downsampling_factors"),
        ("zipformer", "audio_encoder_params.feedforward_dims"),
        ("parakeet", "decoder_params.tdt_durations"),
    ),
)
def test_validate_model_config_propagates_non_iterable_metadata(
    architecture: str,
    field: str,
) -> None:
    model_config = make_model_config(architecture)
    OmegaConf.update(model_config, field, 1)

    with pytest.raises(TypeError):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "updates", "message"),
    (
        pytest.param(
            "zipformer",
            {"audio_encoder_params.encoder_dims": [16, 24, 32, 24, 20, 16]},
            "nondecreasing",
            id="zipformer-ascending-dimension-order",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.encoder_dims": [16, 24, 32, 40, 24, 32]},
            "nondecreasing",
            id="zipformer-descending-dimension-order",
        ),
        pytest.param(
            "zipformer",
            {
                "decoder_type": "transducer_greedy_search",
                "decoder_params.beam": 2,
            },
            "Expected beam=1",
            id="greedy-beam",
        ),
        pytest.param(
            "zipformer",
            {
                "decoder_type": "ctc_greedy_search",
                "decoder_params.beam": 2,
                "audio_encoder_params.output_dim": 4,
                "audio_encoder_params.use_ctc": True,
            },
            "Expected beam=1",
            id="ctc-greedy-beam",
        ),
        pytest.param(
            "zipformer",
            {"decoder_params.beam": 5},
            "beam <= vocab_size",
            id="beam-above-vocabulary",
        ),
        pytest.param(
            "zipformer",
            {"blank_id": -1},
            "Expected Zipformer blank_id",
            id="zipformer-negative-blank-id",
        ),
        pytest.param(
            "zipformer",
            {"blank_id": 4},
            "Expected Zipformer blank_id",
            id="zipformer-blank-id-at-vocabulary-size",
        ),
        pytest.param(
            "parakeet",
            {"blank_id": 3},
            "Expected blank_id=4",
            id="parakeet-blank-id-mismatch",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.use_ctc": True},
            "use_ctc=False",
            id="zipformer-ctc-metadata",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.use_ctc": 0},
            "use_ctc=False",
            id="zipformer-ctc-type",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.output_dim": 4},
            "output_dim=3",
            id="zipformer-output-dimension",
        ),
        pytest.param(
            "zipformer",
            {
                "decoder_type": "ctc_greedy_search",
                "decoder_params.beam": 1,
                "audio_encoder_params.use_ctc": True,
            },
            "output_dim=4",
            id="zipformer-ctc-output-dimension",
        ),
        pytest.param(
            "zipformer",
            {"decoder_params.context_size": 3},
            "context_size at most 2",
            id="zipformer-context-size",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.encoder_dim": 7},
            "model_dim and decoder_params.encoder_dim must match",
            id="parakeet-encoder-dimension",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": []},
            "non-negative signed 32-bit integers",
            id="parakeet-empty-durations",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [0, -1, 2]},
            "non-negative signed 32-bit integers",
            id="parakeet-negative-duration",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [0, 1.0, 2]},
            "non-negative signed 32-bit integers",
            id="parakeet-float-duration",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [0, 1, 1]},
            "must contain unique values",
            id="parakeet-duplicate-durations",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [0, 1, INT32_MAX + 1]},
            "non-negative signed 32-bit integers",
            id="parakeet-duration-overflow",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [0, 1]},
            "must match decoder_params.num_extra_outputs",
            id="parakeet-duration-count",
        ),
        pytest.param(
            "parakeet",
            {"decoder_params.tdt_durations": [1, 2, 3]},
            "must contain zero and at least one positive duration",
            id="parakeet-missing-zero-duration",
        ),
        pytest.param(
            "parakeet",
            {
                "decoder_params.num_extra_outputs": 1,
                "decoder_params.tdt_durations": [0],
            },
            "must contain zero and at least one positive duration",
            id="parakeet-missing-positive-duration",
        ),
        pytest.param(
            "parakeet",
            {
                "vocab_size": 50_000,
                "blank_id": 50_000,
                "decoder_params.beam": 50_000,
            },
            "Parakeet per-utterance search table",
            id="parakeet-search-overflow",
        ),
        pytest.param(
            "zipformer",
            {"vocab_size": 50_000, "decoder_params.beam": 50_000},
            "Zipformer per-utterance search table",
            id="zipformer-search-overflow",
        ),
        pytest.param(
            "zipformer",
            {
                "vocab_size": 50_000,
                "audio_encoder_params.output_dim": 1,
                "decoder_params.beam": 1,
                "decoder_params.joiner_dim": 1,
            },
            "predictor context cache exceeds signed 32-bit",
            id="zipformer-context-cache-overflow",
        ),
    ),
)
def test_validate_model_config_rejects_inconsistent_metadata(
    architecture: str,
    updates: dict[str, int | bool | str | list[int | float]],
    message: str,
) -> None:
    model_config = make_model_config(architecture)
    for field, value in updates.items():
        OmegaConf.update(model_config, field, value)

    with pytest.raises(ASRInitializationError, match=message):
        validate_model_config(model_config)


@pytest.mark.parametrize(
    ("architecture", "updates", "message"),
    (
        pytest.param(
            "zipformer",
            {"audio_encoder_params.min_audio_seconds": 1},
            "finite floats",
            id="non-float-duration",
        ),
        pytest.param(
            "parakeet",
            {"audio_encoder_params.min_audio_seconds": 0.0},
            "0 < min_audio_seconds",
            id="zero-minimum-duration",
        ),
        pytest.param(
            "parakeet",
            {"audio_encoder_params.opt_audio_seconds": 0.05},
            "min_audio_seconds <= opt_audio_seconds",
            id="minimum-above-optimum",
        ),
        pytest.param(
            "parakeet",
            {"audio_encoder_params.max_audio_seconds": 0.5},
            "opt_audio_seconds <= max_audio_seconds",
            id="optimum-above-maximum",
        ),
        pytest.param(
            "parakeet",
            {"audio_encoder_params.max_audio_seconds": float("inf")},
            "exceeds signed 32-bit sample indexing",
            id="sample-index-overflow",
        ),
        pytest.param(
            "parakeet",
            {"audio_encoder_params.min_audio_seconds": 1e-8},
            "between 1 and",
            id="zero-sample-minimum",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.right_padding_samples": -1},
            "non-negative signed-32-bit integer",
            id="negative-right-padding",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.right_padding_samples": INT32_MAX + 1},
            "non-negative signed-32-bit integer",
            id="oversized-right-padding",
        ),
        pytest.param(
            "zipformer",
            {"audio_encoder_params.min_audio_seconds": 1e-8},
            "fit inside the minimum audio profile",
            id="padding-only-minimum",
        ),
    ),
)
def test_validate_model_config_rejects_invalid_audio_profile(
    architecture: str,
    updates: dict[str, int | float],
    message: str,
) -> None:
    model_config = make_model_config(architecture)
    for field, value in updates.items():
        OmegaConf.update(model_config, field, value)

    with pytest.raises(ASRInitializationError, match=message):
        validate_model_config(model_config)


@pytest.mark.parametrize("engine_name", ("zipformer.trt", "parakeet.trt"))
def test_get_engine_loads_native_encoder_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine_name: str,
) -> None:
    engine_path = tmp_path / engine_name
    engine_path.write_bytes(b"engine")
    engine = FakeEngine((), (), {}, {})
    events: list[str] = []
    initializer_calls: list[tuple[trt.ILogger, str]] = []
    runtime_loggers: list[trt.ILogger] = []

    class FakeRuntime:
        """Record TensorRT runtime construction and engine deserialization."""

        def __init__(self, logger: trt.ILogger) -> None:
            """Record the logger used to construct the runtime.

            Parameters
            ----------
            logger : trt.ILogger
                TensorRT logger passed by ``get_engine``.
            """

            events.append("construct-runtime")
            runtime_loggers.append(logger)

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> FakeEngine:
            """Validate serialized bytes and return the configured fake engine.

            Parameters
            ----------
            serialized_engine : bytes
                Engine plan read from the test bundle.

            Returns
            -------
            FakeEngine
                Engine instance configured by the enclosing test.
            """

            events.append("deserialize-engine")
            assert serialized_engine == b"engine"
            return engine

    def load_plugins() -> None:
        """Record native custom-plugin loading."""

        events.append("load-native-plugins")

    def initialize_plugins(logger: trt.ILogger, namespace: str) -> bool:
        """Record standard TensorRT plugin initialization.

        Parameters
        ----------
        logger : trt.ILogger
            Logger supplied to TensorRT plugin initialization.
        namespace : str
            Plugin namespace requested by ``get_engine``.

        Returns
        -------
        bool
            ``True`` to emulate successful initialization.
        """

        events.append("initialize-tensorrt-plugins")
        initializer_calls.append((logger, namespace))
        return True

    monkeypatch.setattr(utils_module, "load_tensorrt_plugins", load_plugins)
    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        initialize_plugins,
    )
    monkeypatch.setattr(utils_module.trt, "Runtime", FakeRuntime)

    assert get_engine(engine_path) is engine
    assert events == [
        "load-native-plugins",
        "initialize-tensorrt-plugins",
        "construct-runtime",
        "deserialize-engine",
    ]
    assert len(runtime_loggers) == 1
    assert initializer_calls == [(runtime_loggers[0], "")]


@pytest.mark.parametrize("error_type", (OSError, RuntimeError))
def test_get_engine_wraps_native_plugin_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError] | type[RuntimeError],
) -> None:
    engine_path = tmp_path / "zipformer.trt"
    engine_path.write_bytes(b"engine")
    failure = error_type("plugin unavailable")

    def fail_to_load_plugins() -> None:
        """Raise the configured native-plugin loading failure.

        Raises
        ------
        OSError | RuntimeError
            Failure instance configured by the parametrized test.
        """

        raise failure

    def fail_to_initialize(_logger: trt.ILogger, _namespace: str) -> bool:
        """Fail if TensorRT initialization continues after plugin loading fails.

        Parameters
        ----------
        _logger : trt.ILogger
            Unused TensorRT logger.
        _namespace : str
            Unused plugin namespace.

        Returns
        -------
        bool
            This callback never returns because reaching it fails the test.
        """

        pytest.fail("TensorRT initialized after custom-plugin loading failed.")

    monkeypatch.setattr(utils_module, "load_tensorrt_plugins", fail_to_load_plugins)
    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        fail_to_initialize,
    )

    with pytest.raises(
        ASRInitializationError,
        match="Failed to load TensorRT plugins",
    ) as error:
        get_engine(engine_path)

    assert error.value.__cause__ is failure


@pytest.mark.parametrize("engine_name", ("decoder.trt", "tdt_decoder.trt"))
def test_get_engine_does_not_load_native_plugins_for_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine_name: str,
) -> None:
    engine_path = tmp_path / engine_name
    engine_path.write_bytes(b"engine")
    engine = FakeEngine((), (), {}, {})
    events: list[str] = []

    class FakeRuntime:
        """Record decoder-engine deserialization without native plugin loading."""

        def __init__(self, logger: trt.ILogger) -> None:
            """Record runtime construction.

            Parameters
            ----------
            logger : trt.ILogger
                TensorRT logger supplied by ``get_engine``.
            """

            events.append("construct-runtime")

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> FakeEngine:
            """Validate serialized bytes and return the configured fake engine.

            Parameters
            ----------
            serialized_engine : bytes
                Decoder engine plan read from the test bundle.

            Returns
            -------
            FakeEngine
                Engine instance configured by the enclosing test.
            """

            events.append("deserialize-engine")
            assert serialized_engine == b"engine"
            return engine

    def initialize_plugins(_logger: trt.ILogger, _namespace: str) -> bool:
        """Record successful standard TensorRT plugin initialization.

        Parameters
        ----------
        _logger : trt.ILogger
            TensorRT logger supplied by ``get_engine``.
        _namespace : str
            Plugin namespace supplied by ``get_engine``.

        Returns
        -------
        bool
            ``True`` to emulate successful initialization.
        """

        events.append("initialize-tensorrt-plugins")
        return True

    monkeypatch.setattr(
        utils_module,
        "load_tensorrt_plugins",
        lambda: pytest.fail("Decoder engine unexpectedly loaded encoder plugins."),
    )
    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        initialize_plugins,
    )
    monkeypatch.setattr(utils_module.trt, "Runtime", FakeRuntime)

    assert get_engine(engine_path) is engine
    assert events == [
        "initialize-tensorrt-plugins",
        "construct-runtime",
        "deserialize-engine",
    ]


@pytest.mark.parametrize("deserialize_result", (None, RuntimeError("invalid plan")))
def test_get_engine_reports_deserialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deserialize_result: None | RuntimeError,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"engine")

    class FakeRuntime:
        """Emulate TensorRT deserialization failure modes."""

        def __init__(self, logger: trt.ILogger) -> None:
            """Retain the TensorRT logger supplied at construction.

            Parameters
            ----------
            logger : trt.ILogger
                Logger passed by ``get_engine``.
            """

            self.logger = logger

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> None:
            """Return no engine or raise the configured deserialization failure.

            Parameters
            ----------
            serialized_engine : bytes
                Serialized engine plan supplied by ``get_engine``.

            Returns
            -------
            None
                Returned when TensorRT deserialization produces no engine.

            Raises
            ------
            RuntimeError
                Raised when the parametrized test supplies a runtime failure.
            """

            if isinstance(deserialize_result, RuntimeError):
                raise deserialize_result
            return None

    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        lambda _logger, _namespace: True,
    )
    monkeypatch.setattr(utils_module.trt, "Runtime", FakeRuntime)

    with pytest.raises(ASRInitializationError, match="Failed to deserialize") as error:
        get_engine(engine_path)

    if deserialize_result is None:
        assert error.value.__cause__ is None
    else:
        assert error.value.__cause__ is deserialize_result


def test_get_engine_wraps_runtime_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"engine")
    failure = RuntimeError("runtime unavailable")

    def make_runtime(_logger: trt.ILogger) -> None:
        """Raise the configured TensorRT runtime-construction failure.

        Parameters
        ----------
        _logger : trt.ILogger
            Logger supplied to the runtime constructor.

        Raises
        ------
        RuntimeError
            Always raised to emulate unavailable TensorRT runtime state.
        """

        raise failure

    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        lambda _logger, _namespace: True,
    )
    monkeypatch.setattr(utils_module.trt, "Runtime", make_runtime)

    with pytest.raises(ASRInitializationError, match="Failed to deserialize") as error:
        get_engine(engine_path)

    assert error.value.__cause__ is failure


def test_get_engine_wraps_file_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"

    class FakeRuntime:
        """Reject deserialization after the expected engine read failure."""

        def __init__(self, logger: trt.ILogger) -> None:
            """Retain the TensorRT logger supplied at construction.

            Parameters
            ----------
            logger : trt.ILogger
                Logger passed by ``get_engine``.
            """

            self.logger = logger

        def deserialize_cuda_engine(self, serialized_engine: bytes) -> FakeEngine:
            """Fail if deserialization is reached with unexpected bytes.

            Parameters
            ----------
            serialized_engine : bytes
                Unexpected serialized engine data.

            Returns
            -------
            FakeEngine
                This method never returns because reaching it fails the test.
            """

            pytest.fail(f"Unexpected engine bytes: {serialized_engine!r}")

    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        lambda _logger, _namespace: True,
    )
    monkeypatch.setattr(
        utils_module.trt,
        "Runtime",
        FakeRuntime,
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

    def fail_to_make_runtime(_logger: trt.ILogger) -> None:
        """Fail if runtime construction follows failed plugin initialization.

        Parameters
        ----------
        _logger : trt.ILogger
            Unused TensorRT logger.
        """

        pytest.fail("Runtime constructed after TensorRT plugin initialization failed.")

    monkeypatch.setattr(
        utils_module.trt,
        "init_libnvinfer_plugins",
        lambda _logger, _namespace: False,
    )
    monkeypatch.setattr(utils_module.trt, "Runtime", fail_to_make_runtime)

    with pytest.raises(ASRInitializationError, match="initialize TensorRT plugins"):
        get_engine(engine_path)


def test_get_names_preserves_engine_order() -> None:
    engine = FakeEngine(
        ("second_input", "first_input"),
        ("second_output", "first_output"),
        {},
        {},
    )
    engine.names = (
        "first_output",
        "second_input",
        "first_input",
        "second_output",
    )

    assert get_names(engine) == (
        ("second_input", "first_input"),
        ("first_output", "second_output"),
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
    install_fake_tokenizer(monkeypatch, tmp_path, vocab_size=4, unk_id=3)

    validate_tokenizer(tmp_path, make_parakeet_config())


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_tokenizer_requires_standalone_word_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    zipformer = architecture == "zipformer"
    unk_id = 4 if zipformer else 3
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=5 if zipformer else 4,
        unk_id=unk_id,
        standalone_id=unk_id,
        standalone_piece="<unk>",
    )
    model_config = make_zipformer_config() if zipformer else make_parakeet_config()

    with pytest.raises(ASRInitializationError, match="standalone SentencePiece"):
        validate_tokenizer(tmp_path, model_config)


def test_validate_tokenizer_reports_vocabulary_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tokenizer(monkeypatch, tmp_path, vocab_size=3, unk_id=2)

    with pytest.raises(ASRInitializationError, match="tokenizer vocabulary size 4"):
        validate_tokenizer(tmp_path, make_parakeet_config())


@pytest.mark.parametrize(
    ("tokenizer_vocab_size", "unk_id"),
    ((6, 5), (5, 1)),
    ids=("trailing-unknown", "in-vocabulary-unknown"),
)
def test_validate_zipformer_tokenizer_reports_effective_vocabulary_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_vocab_size: int,
    unk_id: int,
) -> None:
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=tokenizer_vocab_size,
        unk_id=unk_id,
    )

    with pytest.raises(ASRInitializationError, match="tokenizer vocabulary size 4"):
        validate_tokenizer(tmp_path, make_zipformer_config())


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
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=tokenizer_vocab_size,
        unk_id=unk_id,
    )

    validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_zipformer_tokenizer_requires_blank_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=5,
        unk_id=4,
        blank_id=4,
    )

    with pytest.raises(ASRInitializationError, match="in-vocabulary <blk> token"):
        validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_zipformer_tokenizer_rejects_invalid_blank_piece_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=5,
        unk_id=4,
        blank_id=1,
        blank_piece="not-blank",
    )

    with pytest.raises(ASRInitializationError, match="in-vocabulary <blk> token"):
        validate_tokenizer(tmp_path, make_zipformer_config())


def test_validate_zipformer_tokenizer_rejects_configured_blank_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tokenizer(
        monkeypatch,
        tmp_path,
        vocab_size=5,
        unk_id=4,
        blank_id=1,
    )
    model_config = make_zipformer_config()
    model_config.blank_id = 0

    with pytest.raises(ASRInitializationError, match="Zipformer blank_id 1"):
        validate_tokenizer(tmp_path, model_config)


@pytest.mark.parametrize("error_type", (OSError, RuntimeError))
def test_validate_tokenizer_reports_loader_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError] | type[RuntimeError],
) -> None:
    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    failure = error_type("invalid tokenizer")

    def fail_to_load(model_file: str) -> None:
        """Validate the path and raise the configured tokenizer load failure.

        Parameters
        ----------
        model_file : str
            SentencePiece model path supplied by ``validate_tokenizer``.

        Raises
        ------
        OSError | RuntimeError
            Failure instance configured by the parametrized test.
        """

        assert model_file == str(tokenizer_path)
        raise failure

    monkeypatch.setattr(
        utils_module.spm,
        "SentencePieceProcessor",
        fail_to_load,
    )

    with pytest.raises(
        ASRInitializationError, match="Failed to load SentencePiece"
    ) as error:
        validate_tokenizer(tmp_path, make_parakeet_config())

    assert error.value.__cause__ is failure


@pytest.mark.parametrize("output_dtype", (trt.float32, trt.float16, trt.bfloat16))
@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_encoder_engine_accepts_supported_output_precision(
    architecture: str,
    output_dtype: trt.DataType,
) -> None:
    model_config = make_model_config(architecture)
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


@pytest.mark.parametrize("io_kind", ("input", "output"))
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_validate_encoder_engine_rejects_invalid_io_signature(
    io_kind: str,
    mutation: str,
) -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    names = engine.input_names if io_kind == "input" else engine.output_names
    if mutation == "missing":
        names = names[:-1]
    else:
        names += ("unexpected_tensor",)
    if io_kind == "input":
        engine.input_names = names
    else:
        engine.output_names = names
    engine.names = engine.input_names + engine.output_names
    engine.num_io_tensors = len(engine.names)

    with pytest.raises(ASRInitializationError, match=f"Expected encoder {io_kind}s"):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_encoder_engine_rejects_audio_profile_mismatch(
    architecture: str,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_encoder_engine(model_config)
    profile = list(engine.profiles["audio"])
    profile[1] = (profile[1][0], profile[1][1] + 1)
    engine.profiles["audio"] = tuple(profile)

    with pytest.raises(ASRInitializationError, match="encoder audio profile"):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_shape",
        "rank_one",
        "zero_batch",
        "oversized_batch",
        "variable_batch",
    ),
)
def test_validate_encoder_engine_rejects_malformed_audio_profile(
    malformation: str,
) -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    min_shape, opt_shape, max_shape = engine.profiles["audio"]
    if malformation == "missing_shape":
        engine.profiles["audio"] = (min_shape, opt_shape)
        message = "three rank-2 encoder audio profile shapes"
    elif malformation == "rank_one":
        profile = [min_shape, opt_shape, max_shape]
        profile[1] = (profile[1][0],)
        engine.profiles["audio"] = tuple(profile)
        message = "three rank-2 encoder audio profile shapes"
    elif malformation == "zero_batch":
        engine.profiles["audio"] = (
            (0, min_shape[1]),
            (0, opt_shape[1]),
            (0, max_shape[1]),
        )
        message = "fixed positive signed-32-bit encoder batch size"
    elif malformation == "oversized_batch":
        engine.profiles["audio"] = (
            (INT32_MAX + 1, min_shape[1]),
            (INT32_MAX + 1, opt_shape[1]),
            (INT32_MAX + 1, max_shape[1]),
        )
        message = "fixed positive signed-32-bit encoder batch size"
    else:
        engine.profiles["audio"] = (
            min_shape,
            (min_shape[0] + 1, opt_shape[1]),
            max_shape,
        )
        message = "fixed positive signed-32-bit encoder batch size"

    with pytest.raises(ASRInitializationError, match=message):
        validate_encoder_engine(engine, model_config)


def test_validate_encoder_engine_rejects_lengths_profile_mismatch() -> None:
    model_config = make_zipformer_config()
    engine = make_encoder_engine(model_config)
    profile = list(engine.profiles["audio_lengths"])
    profile[1] = (3,)
    engine.profiles["audio_lengths"] = tuple(profile)

    with pytest.raises(ASRInitializationError, match="audio_lengths profile"):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
@pytest.mark.parametrize(
    ("tensor_name", "dimension"),
    (
        ("audio", 0),
        ("audio", 1),
        ("audio_lengths", 0),
        ("encoder_output", 0),
        ("encoder_output", 1),
        ("encoder_output", 2),
        ("encoder_output_lengths", 0),
    ),
    ids=lambda value: str(value),
)
def test_validate_encoder_engine_rejects_tensor_shape_mismatch(
    architecture: str,
    tensor_name: str,
    dimension: int,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_encoder_engine(model_config)
    shape = list(engine.shapes[tensor_name])
    shape[dimension] += 1
    engine.shapes[tensor_name] = tuple(shape)

    with pytest.raises(
        ASRInitializationError, match=f"encoder tensor {tensor_name} shape"
    ):
        validate_encoder_engine(engine, model_config)


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
@pytest.mark.parametrize("floating_dtype", (trt.float32, trt.float16, trt.bfloat16))
def test_validate_decoder_engine_accepts_supported_precision(
    architecture: str,
    floating_dtype: trt.DataType,
) -> None:
    model_config = make_model_config(architecture)
    validate_decoder_engine(
        make_decoder_engine(model_config, floating_dtype),
        model_config,
        2,
    )


@pytest.mark.parametrize(
    ("architecture", "tensor_name"),
    tuple(
        (architecture, tensor_name)
        for architecture, tensor_names in DECODER_PRECISION_TENSORS.items()
        for tensor_name in tensor_names
    ),
    ids=lambda value: str(value),
)
def test_validate_decoder_engine_rejects_mixed_floating_precision(
    architecture: str,
    tensor_name: str,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_decoder_engine(model_config)
    engine.dtypes[tensor_name] = trt.float32

    with pytest.raises(ASRInitializationError, match="share an FP32, FP16, or BF16"):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_decoder_engine_rejects_uniform_unsupported_precision(
    architecture: str,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_decoder_engine(model_config)
    for tensor_name in DECODER_PRECISION_TENSORS[architecture]:
        engine.dtypes[tensor_name] = trt.int32

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


@pytest.mark.parametrize("batch_size", (0, INT32_MAX + 1, 1.5))
def test_validate_decoder_engine_rejects_invalid_batch_size(
    batch_size: int | float,
) -> None:
    model_config = make_zipformer_config()

    with pytest.raises(ASRInitializationError, match="decoder batch size"):
        validate_decoder_engine(
            make_decoder_engine(model_config),
            model_config,
            cast(int, batch_size),
        )


@pytest.mark.parametrize(
    "architecture",
    ("zipformer", "parakeet"),
)
@pytest.mark.parametrize("io_kind", ("input", "output"))
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_validate_decoder_engine_rejects_invalid_io_signature(
    architecture: str,
    io_kind: str,
    mutation: str,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_decoder_engine(model_config)
    names = engine.input_names if io_kind == "input" else engine.output_names
    if mutation == "missing":
        names = names[:-1]
    else:
        names += ("unexpected_tensor",)
    if io_kind == "input":
        engine.input_names = names
    else:
        engine.output_names = names
    engine.names = engine.input_names + engine.output_names
    engine.num_io_tensors = len(engine.names)

    with pytest.raises(ASRInitializationError, match=f"Expected decoder {io_kind}s"):
        validate_decoder_engine(engine, model_config, 2)


@pytest.mark.parametrize(
    ("architecture", "tensor_name", "dimension"),
    tuple(
        (architecture, tensor_name, dimension)
        for architecture, tensor_ranks in DECODER_TENSOR_RANKS.items()
        for tensor_name, rank in tensor_ranks.items()
        for dimension in range(rank)
    ),
    ids=lambda value: str(value),
)
def test_validate_decoder_engine_rejects_shape_mismatch(
    architecture: str,
    tensor_name: str,
    dimension: int,
) -> None:
    model_config = make_model_config(architecture)
    engine = make_decoder_engine(model_config)
    shape = list(engine.shapes[tensor_name])
    shape[dimension] += 1
    engine.shapes[tensor_name] = tuple(shape)

    with pytest.raises(ASRInitializationError, match=f"{tensor_name} shape"):
        validate_decoder_engine(engine, model_config, 2)


def test_validate_decoder_engine_rejects_capacity_overflow() -> None:
    model_config = make_zipformer_config()
    batch_size = INT32_MAX // model_config.decoder_params.beam + 1

    with pytest.raises(ASRInitializationError, match="decoder capacity exceeds"):
        validate_decoder_engine(
            make_decoder_engine(model_config),
            model_config,
            batch_size,
        )


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_decoder_engine_rejects_tensor_element_overflow(
    architecture: str,
) -> None:
    if architecture == "parakeet":
        model_config = make_parakeet_config()
        model_config.decoder_params.decoder_dim = INT32_MAX
        tensor_name = "input_states_1"
    else:
        model_config = make_zipformer_config()
        model_config.decoder_params.joiner_dim = INT32_MAX
        tensor_name = "decoder_input"
    engine = make_decoder_engine(model_config)

    with pytest.raises(
        ASRInitializationError,
        match=f"Decoder tensor {tensor_name} exceeds signed 32-bit",
    ):
        validate_decoder_engine(engine, model_config, 2)


def test_validate_zipformer_context_lookup_reports_missing_cache(
    tmp_path: Path,
) -> None:
    with pytest.raises(ASRInitializationError, match="Missing predictor context cache"):
        validate_zipformer_context_lookup(tmp_path, make_zipformer_config())


@pytest.mark.parametrize(
    "error_type",
    (EOFError, OSError, RuntimeError, UnpicklingError),
)
def test_validate_zipformer_context_lookup_wraps_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    context_lookup_path = tmp_path / ZIPFORMER_DECODER_CONTEXTS_FILE
    context_lookup_path.touch()
    failure = error_type("invalid cache")

    def fail_to_load(
        path: Path,
        map_location: str,
        weights_only: bool,
    ) -> None:
        """Validate cache-loading options and raise the configured failure.

        Parameters
        ----------
        path : Path
            Predictor context-cache path passed to ``torch.load``.
        map_location : str
            Device mapping requested while loading the cache.
        weights_only : bool
            Whether ``torch.load`` restricts deserialization to tensor data.

        Raises
        ------
        Exception
            Failure instance configured by the parametrized test.
        """

        assert path == context_lookup_path
        assert map_location == "cpu"
        assert weights_only is True
        raise failure

    monkeypatch.setattr(utils_module.torch, "load", fail_to_load)

    with pytest.raises(
        ASRInitializationError,
        match="Failed to load predictor context cache",
    ) as error:
        validate_zipformer_context_lookup(tmp_path, make_zipformer_config())

    assert error.value.__cause__ is failure


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.bfloat16))
@pytest.mark.parametrize("context_size", (1, 2))
def test_validate_zipformer_context_lookup_accepts_supported_dtype(
    tmp_path: Path,
    dtype: torch.dtype,
    context_size: int,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.context_size = context_size
    expected_shape = (
        (model_config.vocab_size + 1) ** model_config.decoder_params.context_size,
        model_config.decoder_params.joiner_dim,
    )
    torch.save(
        torch.zeros(expected_shape, dtype=dtype),
        tmp_path / ZIPFORMER_DECODER_CONTEXTS_FILE,
    )

    validate_zipformer_context_lookup(tmp_path, model_config)


@pytest.mark.parametrize(
    ("malformation", "message"),
    (
        ("non_tensor", "contain one tensor"),
        ("sparse", "contiguous dense CPU tensor"),
        ("noncontiguous", "contiguous dense CPU tensor"),
        ("shape_rows", "context lookup shape"),
        ("shape_columns", "context lookup shape"),
        ("dtype", "FP16, FP32, or BF16"),
        ("nan", "every predictor context lookup value to be finite"),
    ),
)
def test_validate_zipformer_context_lookup_rejects_invalid_content(
    tmp_path: Path,
    malformation: str,
    message: str,
) -> None:
    model_config = make_zipformer_config()
    expected_shape = (
        (model_config.vocab_size + 1) ** model_config.decoder_params.context_size,
        model_config.decoder_params.joiner_dim,
    )
    if malformation == "non_tensor":
        payload = {"context_lookup": torch.zeros(expected_shape)}
    elif malformation == "sparse":
        payload = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.int64),
            torch.empty(0),
            size=expected_shape,
            check_invariants=True,
        )
    elif malformation == "noncontiguous":
        payload = torch.zeros(expected_shape[::-1]).T
        assert not payload.is_contiguous()
    elif malformation == "shape_rows":
        payload = torch.zeros(expected_shape[0] - 1, expected_shape[1])
    elif malformation == "shape_columns":
        payload = torch.zeros(expected_shape[0], expected_shape[1] - 1)
    elif malformation == "dtype":
        payload = torch.zeros(expected_shape, dtype=torch.int32)
    else:
        payload = torch.zeros(expected_shape)
        payload[0, 0] = float("nan")

    torch.save(payload, tmp_path / ZIPFORMER_DECODER_CONTEXTS_FILE)

    expected_warning = (
        pytest.warns(UserWarning, match="Validating sparse tensor invariants")
        if malformation == "sparse"
        else nullcontext()
    )
    with expected_warning, pytest.raises(ASRInitializationError, match=message):
        validate_zipformer_context_lookup(tmp_path, model_config)


def test_validate_model_rejects_config_before_reading_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_zipformer_config()
    model_config.decoder_params.beam = 0
    monkeypatch.setattr(
        utils_module,
        "validate_tokenizer",
        lambda _model_dir, _model_config: pytest.fail(
            "Tokenizer loaded before model metadata was validated."
        ),
    )

    with pytest.raises(ASRInitializationError, match="positive integer"):
        validate_model(tmp_path, model_config)


def test_validate_model_stops_after_tokenizer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = ASRInitializationError("invalid tokenizer")

    def reject_tokenizer(_model_dir: Path, _model_config: DictConfig) -> None:
        """Raise the configured tokenizer-validation failure.

        Parameters
        ----------
        _model_dir : Path
            Unused model bundle directory.
        _model_config : DictConfig
            Unused runtime model configuration.

        Raises
        ------
        ASRInitializationError
            Always raised to stop model validation at the tokenizer stage.
        """

        raise failure

    monkeypatch.setattr(utils_module, "validate_tokenizer", reject_tokenizer)
    monkeypatch.setattr(
        utils_module,
        "get_engine",
        lambda _path: pytest.fail("Engine loaded after tokenizer validation failed."),
    )

    with pytest.raises(ASRInitializationError) as error:
        validate_model(tmp_path, make_zipformer_config())

    assert error.value is failure


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_model_reports_missing_transducer_decoder_before_loading_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    model_config = make_model_config(architecture)
    encoder_filename, decoder_filename = TRANSDUCER_MODEL_FILES[architecture]
    (tmp_path / encoder_filename).touch()
    tokenizer_calls: list[tuple[Path, DictConfig]] = []

    def validate_tokenizer(model_dir: Path, config: DictConfig) -> None:
        """Record tokenizer validation before artifact preflight fails.

        Parameters
        ----------
        model_dir : Path
            Model bundle directory supplied by ``validate_model``.
        config : DictConfig
            Runtime model configuration supplied by ``validate_model``.
        """

        tokenizer_calls.append((model_dir, config))

    monkeypatch.setattr(utils_module, "validate_tokenizer", validate_tokenizer)
    monkeypatch.setattr(
        utils_module,
        "get_engine",
        lambda _: pytest.fail("Engine loaded before artifact preflight completed."),
    )

    with pytest.raises(ASRInitializationError, match=decoder_filename):
        validate_model(tmp_path, model_config)

    assert tokenizer_calls == [(tmp_path, model_config)]


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_model_rejects_encoder_before_loading_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    model_config = make_model_config(architecture)
    encoder_filename, decoder_filename = TRANSDUCER_MODEL_FILES[architecture]
    (tmp_path / encoder_filename).touch()
    (tmp_path / decoder_filename).touch()
    encoder = make_encoder_engine(model_config)
    encoder.shapes["encoder_output_lengths"] = (3,)

    def load_engine(engine_path: Path) -> FakeEngine:
        """Return the malformed encoder and reject premature decoder loading.

        Parameters
        ----------
        engine_path : Path
            Engine path requested by ``validate_model``.

        Returns
        -------
        FakeEngine
            Malformed encoder metadata used to trigger validation failure.
        """

        if engine_path.name == decoder_filename:
            pytest.fail("Decoder loaded after encoder validation failed.")
        return encoder

    monkeypatch.setattr(
        utils_module, "validate_tokenizer", lambda _model_dir, _model_config: None
    )
    monkeypatch.setattr(utils_module, "get_engine", load_engine)

    with pytest.raises(
        ASRInitializationError,
        match="encoder tensor encoder_output_lengths shape",
    ):
        validate_model(tmp_path, model_config)


def test_validate_model_rejects_decoder_before_loading_context_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_zipformer_config()
    encoder_path = tmp_path / "zipformer.trt"
    decoder_path = tmp_path / "decoder.trt"
    encoder_path.touch()
    decoder_path.touch()
    decoder = make_decoder_engine(model_config)
    decoder.shapes["tokens_log_prob"] = (
        decoder.shapes["tokens_log_prob"][0],
        model_config.vocab_size + 1,
    )
    engines = {
        encoder_path: make_encoder_engine(model_config),
        decoder_path: decoder,
    }

    monkeypatch.setattr(
        utils_module,
        "validate_tokenizer",
        lambda _model_dir, _model_config: None,
    )
    monkeypatch.setattr(utils_module, "get_engine", engines.__getitem__)
    monkeypatch.setattr(
        utils_module,
        "validate_zipformer_context_lookup",
        lambda _model_dir, _model_config: pytest.fail(
            "Context cache loaded after decoder validation failed."
        ),
    )

    with pytest.raises(ASRInitializationError, match="tokens_log_prob shape"):
        validate_model(tmp_path, model_config)


@pytest.mark.parametrize("architecture", ("zipformer", "parakeet"))
def test_validate_model_reports_missing_encoder_before_loading_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    model_config = make_model_config(architecture)
    encoder_filename = TRANSDUCER_MODEL_FILES[architecture][0]
    tokenizer_calls: list[tuple[Path, DictConfig]] = []

    def validate_tokenizer(model_dir: Path, config: DictConfig) -> None:
        """Record tokenizer validation before missing-engine preflight.

        Parameters
        ----------
        model_dir : Path
            Model bundle directory supplied by ``validate_model``.
        config : DictConfig
            Runtime model configuration supplied by ``validate_model``.
        """

        tokenizer_calls.append((model_dir, config))

    monkeypatch.setattr(utils_module, "validate_tokenizer", validate_tokenizer)
    monkeypatch.setattr(
        utils_module,
        "get_engine",
        lambda _path: pytest.fail("An engine was loaded before artifact preflight."),
    )

    with pytest.raises(ASRInitializationError, match=encoder_filename):
        validate_model(tmp_path, model_config)

    assert tokenizer_calls == [(tmp_path, model_config)]


def test_validate_model_accepts_complete_zipformer_ctc_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = make_zipformer_config("ctc_greedy_search")
    (tmp_path / "zipformer.trt").touch()
    encoder = make_encoder_engine(model_config)
    tokenizer_calls: list[tuple[Path, DictConfig]] = []
    loaded_paths: list[Path] = []

    def validate_tokenizer(model_dir: Path, config: DictConfig) -> None:
        """Record tokenizer validation for a complete CTC bundle.

        Parameters
        ----------
        model_dir : Path
            Model bundle directory supplied by ``validate_model``.
        config : DictConfig
            Runtime model configuration supplied by ``validate_model``.
        """

        tokenizer_calls.append((model_dir, config))

    monkeypatch.setattr(
        utils_module,
        "validate_tokenizer",
        validate_tokenizer,
    )
    monkeypatch.setattr(
        utils_module,
        "validate_zipformer_context_lookup",
        lambda _model_dir, _model_config: pytest.fail(
            "A CTC bundle unexpectedly validated a transducer context lookup."
        ),
    )

    def load_engine(engine_path: Path) -> FakeEngine:
        """Record and return the sole CTC encoder engine.

        Parameters
        ----------
        engine_path : Path
            Encoder engine path requested by ``validate_model``.

        Returns
        -------
        FakeEngine
            Valid Zipformer CTC encoder metadata.
        """

        loaded_paths.append(engine_path)
        return encoder

    monkeypatch.setattr(utils_module, "get_engine", load_engine)

    validate_model(tmp_path, model_config)

    assert tokenizer_calls == [(tmp_path, model_config)]
    assert loaded_paths == [tmp_path / "zipformer.trt"]


@pytest.mark.parametrize(
    ("architecture", "batch_size"), (("zipformer", 3), ("parakeet", 5))
)
def test_validate_model_accepts_complete_transducer_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    batch_size: int,
) -> None:
    model_config = make_model_config(architecture)
    encoder_filename, decoder_filename = TRANSDUCER_MODEL_FILES[architecture]
    encoder_path = tmp_path / encoder_filename
    decoder_path = tmp_path / decoder_filename
    encoder_path.touch()
    decoder_path.touch()
    engines = {
        encoder_path: make_encoder_engine(model_config, batch_size=batch_size),
        decoder_path: make_decoder_engine(model_config, batch_size=batch_size),
    }
    loaded_paths: list[Path] = []
    context_calls: list[tuple[Path, DictConfig]] = []

    monkeypatch.setattr(
        utils_module, "validate_tokenizer", lambda _model_dir, _model_config: None
    )

    def load_engine(engine_path: Path) -> FakeEngine:
        """Record and return metadata for a requested transducer engine.

        Parameters
        ----------
        engine_path : Path
            Encoder or decoder engine path requested by ``validate_model``.

        Returns
        -------
        FakeEngine
            Matching engine metadata from the complete test bundle.
        """

        loaded_paths.append(engine_path)
        return engines[engine_path]

    monkeypatch.setattr(utils_module, "get_engine", load_engine)

    def validate_context(model_dir: Path, config: DictConfig) -> None:
        """Record Zipformer context-cache validation and reject Parakeet use.

        Parameters
        ----------
        model_dir : Path
            Model bundle directory supplied by ``validate_model``.
        config : DictConfig
            Runtime model configuration supplied by ``validate_model``.
        """

        if architecture != "zipformer":
            pytest.fail("Parakeet unexpectedly validated a context lookup.")
        context_calls.append((model_dir, config))

    monkeypatch.setattr(
        utils_module,
        "validate_zipformer_context_lookup",
        validate_context,
    )

    validate_model(tmp_path, model_config)

    assert loaded_paths == [encoder_path, decoder_path]
    expected_context_calls = (
        [(tmp_path, model_config)] if architecture == "zipformer" else []
    )
    assert context_calls == expected_context_calls
