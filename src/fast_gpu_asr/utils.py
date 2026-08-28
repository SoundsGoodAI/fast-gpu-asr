#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""TensorRT engine loading and ASR model-bundle validation."""

from math import prod
from pathlib import Path
from pickle import UnpicklingError

import sentencepiece as spm
import tensorrt as trt
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .constants import (
    DECODER_TYPES,
    INT32_MAX,
    MODEL_TYPE_PARAKEET,
    MODEL_TYPE_ZIPFORMER,
    PARAKEET_DECODER_TENSORRT_FILE,
    PARAKEET_TENSORRT_FILE,
    TOKENIZER_FILE,
    ZIPFORMER_DECODER_CONTEXTS_FILE,
    ZIPFORMER_DECODER_TENSORRT_FILE,
    ZIPFORMER_TENSORRT_FILE,
)
from .tensorrt_plugins import load_tensorrt_plugins


class ASRInitializationError(Exception):
    """Raised when an ASR model bundle cannot be initialized."""


class ASRInferenceError(Exception):
    """Raised when ASR inference input or execution is invalid."""


def get_engine(engine_path: Path) -> trt.ICudaEngine:
    """Load one serialized TensorRT engine.

    Parameters
    ----------
    engine_path : Path
        Path to the serialized TensorRT engine.

    Returns
    -------
    trt.ICudaEngine
        Deserialized TensorRT engine.

    Raises
    ------
    ASRInitializationError
        Raised when the engine is unreadable, required plugins cannot be loaded,
        or TensorRT cannot deserialize the engine.
    """

    if engine_path.name in (PARAKEET_TENSORRT_FILE, ZIPFORMER_TENSORRT_FILE):
        try:
            load_tensorrt_plugins()
        except (OSError, RuntimeError) as error:
            raise ASRInitializationError(
                f"Failed to load TensorRT plugins for {engine_path}: {error}"
            ) from error

    logger = trt.Logger(trt.Logger.ERROR)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise ASRInitializationError("Failed to initialize TensorRT plugins.")

    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    except (OSError, RuntimeError) as error:
        raise ASRInitializationError(
            f"Failed to deserialize TensorRT engine {engine_path}: {error}"
        ) from error

    if engine is None:
        raise ASRInitializationError(
            f"Failed to deserialize TensorRT engine {engine_path}."
        )

    return engine


def get_names(engine: trt.ICudaEngine) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return input and output tensor names in engine order.

    Parameters
    ----------
    engine : trt.ICudaEngine
        TensorRT engine to inspect.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Input tensor names and output tensor names, each preserving engine
        order.

    Raises
    ------
    ASRInitializationError
        Raised when TensorRT reports an I/O tensor with neither input nor output
        mode.
    """

    input_names: list[str] = []
    output_names: list[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
        else:
            raise ASRInitializationError(
                f"TensorRT tensor {name} has unsupported I/O mode {mode}."
            )

    return tuple(input_names), tuple(output_names)


def validate_tokenizer(model_dir: Path, model_config: DictConfig) -> None:
    """Validate the SentencePiece tokenizer bundled with one model.

    Zipformer checkpoints may omit a final unknown token from their network
    outputs, while Parakeet represents every SentencePiece token and appends
    its blank outside the tokenizer vocabulary.

    Parameters
    ----------
    model_dir : Path
        Directory containing the exported model bundle.
    model_config : DictConfig
        Validated runtime model configuration.

    Raises
    ------
    ASRInitializationError
        Raised when ``bpe.model`` is missing or invalid, its effective
        vocabulary differs from the model configuration, or a Zipformer
        tokenizer does not contain the configured in-vocabulary ``<blk>`` token.
    """

    tokenizer_path = model_dir / TOKENIZER_FILE
    if not tokenizer_path.is_file():
        raise ASRInitializationError(
            f"Missing SentencePiece tokenizer {tokenizer_path}."
        )

    try:
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    except (OSError, RuntimeError) as error:
        raise ASRInitializationError(
            f"Failed to load SentencePiece tokenizer {tokenizer_path}: {error}"
        ) from error

    tokenizer_vocab_size = tokenizer.vocab_size()
    if model_config.model_type == MODEL_TYPE_PARAKEET:
        expected_vocab_size = tokenizer_vocab_size
    else:
        trailing_unknown = tokenizer.unk_id() == tokenizer_vocab_size - 1
        expected_vocab_size = tokenizer_vocab_size - int(trailing_unknown)

    if model_config.vocab_size != expected_vocab_size:
        raise ASRInitializationError(
            f"Expected tokenizer vocabulary size {model_config.vocab_size}, got "
            f"{expected_vocab_size}."
        )

    if model_config.model_type == MODEL_TYPE_ZIPFORMER:
        tokenizer_blank_id = tokenizer.piece_to_id("<blk>")
        if (
            not 0 <= tokenizer_blank_id < model_config.vocab_size
            or tokenizer.id_to_piece(tokenizer_blank_id) != "<blk>"
        ):
            raise ASRInitializationError(
                "Expected the Zipformer tokenizer to contain an in-vocabulary <blk> "
                "token."
            )
        if model_config.blank_id != tokenizer_blank_id:
            raise ASRInitializationError(
                f"Expected Zipformer blank_id {tokenizer_blank_id}, got "
                f"{model_config.blank_id}."
            )


def validate_model_config(model_config: DictConfig) -> None:
    """Validate metadata consumed by the TensorRT inference pipeline.

    The validation covers fields shared by all bundles and then applies the
    model-specific contracts for Zipformer and Parakeet. It verifies decoder
    support, tensor dimensions, profile bounds, blank-token metadata, and TDT
    duration values before any engine or CUDA buffer is initialized.

    Parameters
    ----------
    model_config : DictConfig
        Runtime configuration loaded from an exported ``model_config.yaml``.

    Raises
    ------
    ASRInitializationError
        Raised when the model or decoder type is unsupported, a required
        numeric value is invalid, or related dimensions and decoder metadata
        are inconsistent.
    """

    if not isinstance(model_config, DictConfig):
        raise ASRInitializationError(
            "Expected model_config to be a DictConfig, got "
            f"{type(model_config).__name__}."
        )

    try:
        OmegaConf.to_container(model_config, resolve=True, throw_on_missing=False)
    except OmegaConfBaseException as error:
        raise ASRInitializationError(
            f"Failed to resolve model configuration: {error}"
        ) from error

    for required_field in ("model_type", "decoder_type"):
        if OmegaConf.select(model_config, required_field, default=None) is None:
            raise ASRInitializationError(
                f"Missing required model configuration field {required_field}."
            )

    model_type = model_config.model_type
    decoder_type = model_config.decoder_type
    if model_type not in (MODEL_TYPE_ZIPFORMER, MODEL_TYPE_PARAKEET):
        raise ASRInitializationError(f"Unsupported model_type: {model_type}.")
    if decoder_type not in DECODER_TYPES:
        raise ASRInitializationError(
            f"Expected decoder_type to be one of {DECODER_TYPES}, got {decoder_type}."
        )
    if model_type == MODEL_TYPE_PARAKEET and decoder_type == "ctc_greedy_search":
        raise ASRInitializationError("Parakeet TDT models do not contain a CTC head.")

    required_fields = (
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
    if model_type == MODEL_TYPE_PARAKEET:
        required_fields += (
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
    else:
        required_fields += (
            "audio_encoder_params.downsampling_factors",
            "audio_encoder_params.encoder_dims",
            "audio_encoder_params.feedforward_dims",
            "audio_encoder_params.num_encoder_layers",
            "audio_encoder_params.output_dim",
            "audio_encoder_params.right_padding_samples",
            "audio_encoder_params.use_ctc",
        )
        if decoder_type != "ctc_greedy_search":
            required_fields += (
                "decoder_params.context_size",
                "decoder_params.decoder_dim",
                "decoder_params.joiner_dim",
            )

    for required_field in required_fields:
        if OmegaConf.select(model_config, required_field, default=None) is None:
            raise ASRInitializationError(
                f"Missing required model configuration field {required_field}."
            )

    zipformer_encoder_dims: tuple[int, ...] | None = None
    positive_integer_values = (
        ("model_samplerate", model_config.model_samplerate),
        ("vocab_size", model_config.vocab_size),
        (
            "audio_encoder_params.frame_shift_ms",
            model_config.audio_encoder_params.frame_shift_ms,
        ),
        (
            "audio_encoder_params.subsampling_factor",
            model_config.audio_encoder_params.subsampling_factor,
        ),
        (
            "audio_encoder_params.pos_emb_max_len",
            model_config.audio_encoder_params.pos_emb_max_len,
        ),
        ("decoder_params.beam", model_config.decoder_params.beam),
    )
    if model_config.model_type == MODEL_TYPE_PARAKEET:
        positive_integer_values += (
            (
                "audio_encoder_params.feature_dim",
                model_config.audio_encoder_params.feature_dim,
            ),
            (
                "audio_encoder_params.model_dim",
                model_config.audio_encoder_params.model_dim,
            ),
            (
                "audio_encoder_params.n_layers",
                model_config.audio_encoder_params.n_layers,
            ),
            ("decoder_params.encoder_dim", model_config.decoder_params.encoder_dim),
            ("decoder_params.decoder_dim", model_config.decoder_params.decoder_dim),
            ("decoder_params.joiner_dim", model_config.decoder_params.joiner_dim),
            (
                "decoder_params.pred_rnn_layers",
                model_config.decoder_params.pred_rnn_layers,
            ),
            (
                "decoder_params.num_extra_outputs",
                model_config.decoder_params.num_extra_outputs,
            ),
            (
                "decoder_params.max_symbols_per_timestep",
                model_config.decoder_params.max_symbols_per_timestep,
            ),
        )
    else:
        positive_integer_values += (
            (
                "audio_encoder_params.feature_dim",
                model_config.audio_encoder_params.feature_dim,
            ),
            (
                "audio_encoder_params.output_dim",
                model_config.audio_encoder_params.output_dim,
            ),
        )
        for name in (
            "encoder_dims",
            "num_encoder_layers",
            "downsampling_factors",
            "feedforward_dims",
        ):
            values = tuple(model_config.audio_encoder_params[name])
            if len(values) != 6:
                raise ASRInitializationError(
                    f"Expected audio_encoder_params.{name} to contain six positive "
                    f"integers, got {values}."
                )
            positive_integer_values += tuple(
                (f"audio_encoder_params.{name}[{index}]", value)
                for index, value in enumerate(values)
            )
            if name == "encoder_dims":
                zipformer_encoder_dims = values

        if model_config.decoder_type != "ctc_greedy_search":
            positive_integer_values += (
                (
                    "decoder_params.context_size",
                    model_config.decoder_params.context_size,
                ),
                (
                    "decoder_params.decoder_dim",
                    model_config.decoder_params.decoder_dim,
                ),
                (
                    "decoder_params.joiner_dim",
                    model_config.decoder_params.joiner_dim,
                ),
            )

    for name, value in positive_integer_values:
        if not isinstance(value, int) or not 1 <= value <= INT32_MAX:
            raise ASRInitializationError(
                f"Expected {name} to be a positive integer no greater than "
                f"{INT32_MAX}, got {value}."
            )

    if zipformer_encoder_dims is not None and not (
        zipformer_encoder_dims[0]
        <= zipformer_encoder_dims[1]
        <= zipformer_encoder_dims[2]
        <= zipformer_encoder_dims[3]
        >= zipformer_encoder_dims[4]
        >= zipformer_encoder_dims[5]
    ):
        raise ASRInitializationError(
            "audio_encoder_params.encoder_dims must be nondecreasing through the "
            "fourth stack and nonincreasing afterward, but got "
            f"{zipformer_encoder_dims}."
        )

    beam = model_config.decoder_params.beam
    decoder_type = model_config.decoder_type
    if decoder_type in ("ctc_greedy_search", "transducer_greedy_search") and beam != 1:
        raise ASRInitializationError(f"Expected beam=1 for {decoder_type}, got {beam}.")
    if beam > model_config.vocab_size:
        raise ASRInitializationError(
            f"Expected decoder_params.beam <= vocab_size, got beam={beam} and "
            f"vocab_size={model_config.vocab_size}."
        )

    audio_seconds = (
        model_config.audio_encoder_params.min_audio_seconds,
        model_config.audio_encoder_params.opt_audio_seconds,
        model_config.audio_encoder_params.max_audio_seconds,
    )
    if any(not isinstance(value, float) for value in audio_seconds):
        raise ASRInitializationError(
            "Expected min_audio_seconds, opt_audio_seconds, and max_audio_seconds "
            f"to be finite floats, got {audio_seconds}."
        )
    if not 0.0 < audio_seconds[0] <= audio_seconds[1] <= audio_seconds[2]:
        raise ASRInitializationError(
            "Expected 0 < min_audio_seconds <= opt_audio_seconds <= "
            f"max_audio_seconds, got {audio_seconds}."
        )

    right_padding_samples = 0
    if model_config.model_type == MODEL_TYPE_ZIPFORMER:
        right_padding_samples = model_config.audio_encoder_params.right_padding_samples
        if (
            not isinstance(right_padding_samples, int)
            or not 0 <= right_padding_samples <= INT32_MAX
        ):
            raise ASRInitializationError(
                "Expected right_padding_samples to be a non-negative signed-32-bit "
                f"integer, got {right_padding_samples}."
            )

        use_ctc = model_config.audio_encoder_params.use_ctc
        expected_use_ctc = model_config.decoder_type == "ctc_greedy_search"
        if not isinstance(use_ctc, bool) or use_ctc != expected_use_ctc:
            raise ASRInitializationError(
                f"Expected audio_encoder_params.use_ctc={expected_use_ctc} for "
                f"decoder_type={model_config.decoder_type}, got {use_ctc}."
            )

    max_audio_seconds = (INT32_MAX - right_padding_samples) / (
        model_config.model_samplerate
    )
    if audio_seconds[-1] > max_audio_seconds:
        raise ASRInitializationError(
            "The maximum audio profile exceeds signed 32-bit sample indexing: "
            f"max_audio_seconds={audio_seconds[-1]}, limit={max_audio_seconds}."
        )
    audio_sample_profile = tuple(
        round(seconds * model_config.model_samplerate) + right_padding_samples
        for seconds in audio_seconds
    )
    if not 1 <= audio_sample_profile[0] <= audio_sample_profile[-1] <= INT32_MAX:
        raise ASRInitializationError(
            "Expected the audio profile to contain between 1 and "
            f"{INT32_MAX} samples, got {audio_sample_profile}."
        )
    if (
        model_config.model_type == MODEL_TYPE_ZIPFORMER
        and right_padding_samples >= audio_sample_profile[0]
    ):
        raise ASRInitializationError(
            "Expected right_padding_samples to fit inside the minimum "
            f"audio profile, got {right_padding_samples}."
        )

    blank_penalty = model_config.decoder_params.blank_penalty
    max_float32 = torch.finfo(torch.float32).max
    if (
        not isinstance(blank_penalty, float)
        or not -max_float32 <= blank_penalty <= max_float32
    ):
        raise ASRInitializationError(
            "Expected decoder_params.blank_penalty to be a finite float32 value, "
            f"got {blank_penalty}."
        )

    blank_id = model_config.blank_id
    if model_config.model_type == MODEL_TYPE_PARAKEET:
        if not isinstance(blank_id, int) or blank_id != model_config.vocab_size:
            raise ASRInitializationError(
                f"Expected blank_id={model_config.vocab_size}, got {blank_id}."
            )
        if (
            model_config.audio_encoder_params.model_dim
            != model_config.decoder_params.encoder_dim
        ):
            raise ASRInitializationError(
                "audio_encoder_params.model_dim and decoder_params.encoder_dim "
                "must match."
            )

        configured_durations = model_config.decoder_params.tdt_durations
        durations = tuple(configured_durations)
        if not durations or any(
            not isinstance(duration, int) or not 0 <= duration <= INT32_MAX
            for duration in durations
        ):
            raise ASRInitializationError(
                "decoder_params.tdt_durations must contain non-negative signed "
                "32-bit integers."
            )
        if len(durations) != len(set(durations)):
            raise ASRInitializationError(
                "decoder_params.tdt_durations must contain unique values."
            )
        if len(durations) != model_config.decoder_params.num_extra_outputs:
            raise ASRInitializationError(
                "The number of decoder_params.tdt_durations must match "
                "decoder_params.num_extra_outputs."
            )
        if 0 not in durations or all(duration <= 0 for duration in durations):
            raise ASRInitializationError(
                "decoder_params.tdt_durations must contain zero and at least one "
                "positive duration."
            )

        positive_duration_count = sum(duration > 0 for duration in durations)
        candidate_count = beam * (len(durations) * beam + positive_duration_count)
        if candidate_count > INT32_MAX:
            raise ASRInitializationError(
                "The Parakeet per-utterance search table exceeds signed 32-bit "
                f"indexing: {candidate_count} candidates, limit={INT32_MAX}."
            )
    else:
        if not isinstance(blank_id, int) or not 0 <= blank_id < model_config.vocab_size:
            raise ASRInitializationError(
                "Expected Zipformer blank_id to be an integer in "
                f"[0, {model_config.vocab_size}), got {blank_id}."
            )

        expected_output_dim = (
            model_config.vocab_size
            if model_config.decoder_type == "ctc_greedy_search"
            else model_config.decoder_params.joiner_dim
        )
        if model_config.audio_encoder_params.output_dim != expected_output_dim:
            raise ASRInitializationError(
                f"Expected audio_encoder_params.output_dim={expected_output_dim}, "
                f"got {model_config.audio_encoder_params.output_dim}."
            )

        if model_config.decoder_type != "ctc_greedy_search":
            context_size = model_config.decoder_params.context_size
            if context_size > 2:
                raise ASRInitializationError(
                    "Expected decoder_params.context_size at most 2, got "
                    f"{context_size}."
                )

            candidate_count = beam * model_config.vocab_size
            if candidate_count > INT32_MAX:
                raise ASRInitializationError(
                    "The Zipformer per-utterance search table exceeds signed "
                    f"32-bit indexing: {candidate_count} candidates, limit={INT32_MAX}."
                )

            context_lookup_elements = (
                model_config.decoder_params.joiner_dim
                * (model_config.vocab_size + 1) ** context_size
            )
            if context_lookup_elements > INT32_MAX:
                raise ASRInitializationError(
                    "The Zipformer predictor context cache exceeds signed 32-bit "
                    f"kernel indexing: {context_lookup_elements} elements, "
                    f"limit={INT32_MAX}."
                )


def validate_encoder_engine(engine: trt.ICudaEngine, model_config: DictConfig) -> int:
    """Validate an encoder engine and return its fixed batch size.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized Zipformer or Parakeet encoder engine.
    model_config : DictConfig
        Validated runtime model configuration.

    Returns
    -------
    int
        Fixed batch size encoded in the TensorRT optimization profile.

    Raises
    ------
    ASRInitializationError
        Raised when tensor names, shapes, dtypes, locations, storage formats,
        or the audio optimization profile differ from the runtime contract.
    """

    input_names, output_names = get_names(engine)
    expected_input_names = ("audio", "audio_lengths")
    expected_output_names = ("encoder_output", "encoder_output_lengths")
    if sorted(input_names) != sorted(expected_input_names):
        raise ASRInitializationError(
            f"Expected encoder inputs {sorted(expected_input_names)}, "
            f"got {sorted(input_names)}."
        )
    if sorted(output_names) != sorted(expected_output_names):
        raise ASRInitializationError(
            f"Expected encoder outputs {sorted(expected_output_names)}, "
            f"got {sorted(output_names)}."
        )
    for name in input_names + output_names:
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ASRInitializationError(
                f"Expected encoder tensor {name} to reside on the GPU."
            )
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ASRInitializationError(
                f"Expected encoder tensor {name} to use linear storage."
            )

    audio_profile = tuple(
        tuple(shape) for shape in engine.get_tensor_profile_shape("audio", 0)
    )
    if len(audio_profile) != 3 or any(len(shape) != 2 for shape in audio_profile):
        raise ASRInitializationError(
            f"Expected three rank-2 encoder audio profile shapes, got {audio_profile}."
        )

    batch_size = audio_profile[0][0]
    if not 1 <= batch_size <= INT32_MAX or any(
        shape[0] != batch_size for shape in audio_profile
    ):
        raise ASRInitializationError(
            "Expected a fixed positive signed-32-bit encoder batch size, "
            f"got {audio_profile}."
        )

    right_padding_samples = (
        model_config.audio_encoder_params.right_padding_samples
        if model_config.model_type == MODEL_TYPE_ZIPFORMER
        else 0
    )
    expected_audio_profile = tuple(
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
    if audio_profile != expected_audio_profile:
        raise ASRInitializationError(
            f"Expected encoder audio profile {expected_audio_profile}, got "
            f"{audio_profile}."
        )

    lengths_profile = tuple(
        tuple(shape) for shape in engine.get_tensor_profile_shape("audio_lengths", 0)
    )
    expected_lengths_profile = ((batch_size,),) * 3
    if lengths_profile != expected_lengths_profile:
        raise ASRInitializationError(
            f"Expected encoder audio_lengths profile {expected_lengths_profile}, "
            f"got {lengths_profile}."
        )

    encoder_dim = (
        model_config.audio_encoder_params.output_dim
        if model_config.model_type == MODEL_TYPE_ZIPFORMER
        else model_config.audio_encoder_params.model_dim
    )
    expected_shapes = {
        "audio": (batch_size, -1),
        "audio_lengths": (batch_size,),
        "encoder_output": (batch_size, -1, encoder_dim),
        "encoder_output_lengths": (batch_size,),
    }
    for name, expected_shape in expected_shapes.items():
        shape = tuple(engine.get_tensor_shape(name))
        if shape != expected_shape:
            raise ASRInitializationError(
                f"Expected encoder tensor {name} shape {expected_shape}, got {shape}."
            )

    if engine.get_tensor_dtype("audio") != trt.float32:
        raise ASRInitializationError(
            f"Expected encoder audio dtype {trt.float32}, "
            f"got {engine.get_tensor_dtype('audio')}."
        )
    if engine.get_tensor_dtype("audio_lengths") != trt.int64:
        raise ASRInitializationError(
            f"Expected encoder audio_lengths dtype {trt.int64}, "
            f"got {engine.get_tensor_dtype('audio_lengths')}."
        )
    if engine.get_tensor_dtype("encoder_output") not in (
        trt.float32,
        trt.float16,
        trt.bfloat16,
    ):
        raise ASRInitializationError(
            "Expected encoder_output dtype to be FP32, FP16, or BF16, got "
            f"{engine.get_tensor_dtype('encoder_output')}."
        )
    if engine.get_tensor_dtype("encoder_output_lengths") != trt.int32:
        raise ASRInitializationError(
            f"Expected encoder_output_lengths dtype {trt.int32}, "
            f"got {engine.get_tensor_dtype('encoder_output_lengths')}."
        )

    return batch_size


def validate_decoder_engine(
    engine: trt.ICudaEngine, model_config: DictConfig, batch_size: int
) -> None:
    """Validate a transducer decoder engine against bundle metadata.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized Zipformer or Parakeet decoder engine.
    model_config : DictConfig
        Validated runtime model configuration.
    batch_size : int
        Fixed batch size exposed by the paired encoder engine.

    Raises
    ------
    ASRInitializationError
        Raised when decoder capacity, tensor names, shapes, or dtypes differ
        from the configured Zipformer or Parakeet runtime contract.
    """

    if not isinstance(batch_size, int) or not 1 <= batch_size <= INT32_MAX:
        raise ASRInitializationError(
            f"Expected a positive signed-32-bit decoder batch size, got {batch_size}."
        )

    decoder_capacity = batch_size * model_config.decoder_params.beam
    if decoder_capacity > INT32_MAX:
        raise ASRInitializationError(
            "The decoder capacity exceeds signed 32-bit kernel indexing: "
            f"{decoder_capacity} hypotheses, limit={INT32_MAX}."
        )

    input_names, output_names = get_names(engine)
    if model_config.model_type == MODEL_TYPE_ZIPFORMER:
        expected_input_names = ("decoder_input", "encoder_output")
        expected_output_names = ("tokens_log_prob",)
        joiner_dim = model_config.decoder_params.joiner_dim
        expected_shapes = {
            "decoder_input": (decoder_capacity, joiner_dim),
            "encoder_output": (decoder_capacity, joiner_dim),
            "tokens_log_prob": (decoder_capacity, model_config.vocab_size),
        }
        decoder_precision_tensor_names = ("decoder_input", "encoder_output")
        float32_tensor_names = ("tokens_log_prob",)
        int32_tensor_names = ()
    else:
        expected_input_names = (
            "encoder_output",
            "targets",
            "input_states_1",
            "input_states_2",
        )
        expected_output_names = (
            "token_log_probs",
            "duration_log_probs",
            "output_states_1",
            "output_states_2",
        )
        state_shape = (
            model_config.decoder_params.pred_rnn_layers,
            decoder_capacity,
            model_config.decoder_params.decoder_dim,
        )
        encoder_dim = model_config.decoder_params.encoder_dim
        expected_shapes = {
            "encoder_output": (decoder_capacity, encoder_dim),
            "targets": (decoder_capacity, 1),
            "input_states_1": state_shape,
            "input_states_2": state_shape,
            "token_log_probs": (decoder_capacity, model_config.vocab_size + 1),
            "duration_log_probs": (
                decoder_capacity,
                model_config.decoder_params.num_extra_outputs,
            ),
            "output_states_1": state_shape,
            "output_states_2": state_shape,
        }
        decoder_precision_tensor_names = (
            "encoder_output",
            "input_states_1",
            "input_states_2",
            "output_states_1",
            "output_states_2",
        )
        float32_tensor_names = ("token_log_probs", "duration_log_probs")
        int32_tensor_names = ("targets",)

    if sorted(input_names) != sorted(expected_input_names):
        raise ASRInitializationError(
            f"Expected decoder inputs {sorted(expected_input_names)}, "
            f"got {sorted(input_names)}."
        )
    if sorted(output_names) != sorted(expected_output_names):
        raise ASRInitializationError(
            f"Expected decoder outputs {sorted(expected_output_names)}, "
            f"got {sorted(output_names)}."
        )

    for name, expected_shape in expected_shapes.items():
        elements = prod(expected_shape)
        if elements > INT32_MAX:
            raise ASRInitializationError(
                f"Decoder tensor {name} exceeds signed 32-bit kernel indexing: "
                f"{elements} elements, limit={INT32_MAX}."
            )
        shape = tuple(engine.get_tensor_shape(name))
        if shape != expected_shape:
            raise ASRInitializationError(
                f"Expected decoder tensor {name} shape {expected_shape}, got {shape}."
            )

    decoder_precision_dtypes = {
        engine.get_tensor_dtype(name) for name in decoder_precision_tensor_names
    }
    if len(decoder_precision_dtypes) != 1 or not decoder_precision_dtypes.issubset(
        {trt.float32, trt.float16, trt.bfloat16}
    ):
        raise ASRInitializationError(
            "Expected decoder floating-point tensors to share an FP32, FP16, "
            f"or BF16 dtype, got {decoder_precision_dtypes}."
        )
    for name in float32_tensor_names:
        if engine.get_tensor_dtype(name) != trt.float32:
            raise ASRInitializationError(
                f"Expected decoder tensor {name} dtype {trt.float32}, "
                f"got {engine.get_tensor_dtype(name)}."
            )
    for name in int32_tensor_names:
        if engine.get_tensor_dtype(name) != trt.int32:
            raise ASRInitializationError(
                f"Expected decoder tensor {name} dtype {trt.int32}, "
                f"got {engine.get_tensor_dtype(name)}."
            )


def validate_zipformer_context_lookup(
    model_dir: Path, model_config: DictConfig
) -> None:
    """Validate the precomputed Zipformer predictor context cache.

    Parameters
    ----------
    model_dir : Path
        Directory containing the exported model bundle.
    model_config : DictConfig
        Validated Zipformer runtime model configuration.

    Raises
    ------
    ASRInitializationError
        Raised when the predictor cache is missing, is not a contiguous dense
        CPU tensor, has an unsupported dtype or nonfinite values, or its shape
        differs from the configured vocabulary, context size, or joiner dimension.
    """

    context_lookup_path = model_dir / ZIPFORMER_DECODER_CONTEXTS_FILE
    if not context_lookup_path.is_file():
        raise ASRInitializationError(
            f"Missing predictor context cache {context_lookup_path}."
        )

    try:
        context_lookup = torch.load(
            context_lookup_path, map_location="cpu", weights_only=True
        )
    except (EOFError, OSError, RuntimeError, UnpicklingError) as error:
        raise ASRInitializationError(
            f"Failed to load predictor context cache {context_lookup_path}: {error}"
        ) from error

    if not isinstance(context_lookup, torch.Tensor):
        raise ASRInitializationError(
            "Expected the predictor context cache to contain one tensor, "
            f"got {type(context_lookup).__name__}."
        )

    if context_lookup.layout != torch.strided or not context_lookup.is_contiguous():
        raise ASRInitializationError(
            "Expected the predictor context cache to be a contiguous dense CPU tensor, "
            f"got layout={context_lookup.layout}, device={context_lookup.device}, and "
            f"contiguous={context_lookup.is_contiguous()}."
        )

    expected_shape = (
        (model_config.vocab_size + 1) ** model_config.decoder_params.context_size,
        model_config.decoder_params.joiner_dim,
    )
    if context_lookup.shape != expected_shape:
        raise ASRInitializationError(
            f"Expected context lookup shape {expected_shape}, got "
            f"{context_lookup.shape}."
        )

    if context_lookup.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ASRInitializationError(
            "Expected FP16, FP32, or BF16 predictor context lookup, "
            f"got {context_lookup.dtype}."
        )

    if not torch.isfinite(context_lookup).all():
        raise ASRInitializationError(
            "Expected every predictor context lookup value to be finite."
        )


def validate_model(model_dir: Path, model_config: DictConfig) -> None:
    """Validate runtime metadata and required artifacts in a model bundle.

    Parameters
    ----------
    model_dir : Path
        Directory containing the exported model bundle.
    model_config : DictConfig
        Runtime configuration loaded from ``model_config.yaml``.

    Raises
    ------
    ASRInitializationError
        Raised when runtime metadata is invalid; the tokenizer, an engine, or
        the predictor cache is missing or incompatible; or an engine cannot be
        deserialized.
    """

    validate_model_config(model_config)
    validate_tokenizer(model_dir, model_config)
    encoder_filename = (
        ZIPFORMER_TENSORRT_FILE
        if model_config.model_type == MODEL_TYPE_ZIPFORMER
        else PARAKEET_TENSORRT_FILE
    )
    encoder_path = model_dir / encoder_filename
    if not encoder_path.is_file():
        raise ASRInitializationError(f"Missing TensorRT engine {encoder_path}.")

    decoder_path: Path | None = None
    if model_config.decoder_type != "ctc_greedy_search":
        decoder_filename = (
            ZIPFORMER_DECODER_TENSORRT_FILE
            if model_config.model_type == MODEL_TYPE_ZIPFORMER
            else PARAKEET_DECODER_TENSORRT_FILE
        )
        decoder_path = model_dir / decoder_filename
        if not decoder_path.is_file():
            raise ASRInitializationError(f"Missing TensorRT engine {decoder_path}.")

    encoder = get_engine(encoder_path)
    batch_size = validate_encoder_engine(encoder, model_config)

    if decoder_path is None:
        return

    decoder = get_engine(decoder_path)
    validate_decoder_engine(decoder, model_config, batch_size)
    if model_config.model_type == MODEL_TYPE_ZIPFORMER:
        validate_zipformer_context_lookup(model_dir, model_config)
