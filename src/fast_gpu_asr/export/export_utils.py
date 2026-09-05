#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Validate models, prepare ONNX graphs, and build TensorRT engines.

This module centralizes model validation, graph cleanup, and TensorRT builder
configuration shared by the Parakeet and Zipformer exporters.
"""

import argparse
import logging
import math
from collections import OrderedDict
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from omegaconf import DictConfig

from ..constants import (
    DECODER_TYPES,
    INT32_MAX,
    PARAKEET_MAX_ENCODER_FRAMES,
    PRECISION_DTYPES,
    TRANSDUCER_DECODER_TYPES,
)
from ..tensorrt_plugins import load_tensorrt_plugins

logger = logging.getLogger(__name__)


def validate_parakeet(model_config: DictConfig, args: argparse.Namespace) -> None:
    """Validate a source Parakeet configuration and TensorRT export profile.

    This validator accepts the offline Parakeet TDT architecture reconstructed
    by the exporter. It checks the feature extractor, subsampling, attention,
    convolution, prediction network, joiner, and fixed-batch audio profile.

    Parameters
    ----------
    model_config : DictConfig
        Configuration extracted from the source Parakeet ``.nemo`` archive.
    args : argparse.Namespace
        Parsed export arguments containing batch, beam, duration-profile, and
        TensorRT build settings.

    Raises
    ------
    ValueError
        Raised when the source architecture is unsupported or an export
        argument cannot produce a valid fixed-batch TensorRT profile.
    TypeError
        Raised when the configured TDT durations are not iterable.
    """

    expected_values = (
        ("sample_rate", model_config.sample_rate, 16000),
        ("preprocessor.sample_rate", model_config.preprocessor.sample_rate, 16000),
        ("preprocessor.normalize", model_config.preprocessor.normalize, "per_feature"),
        ("preprocessor.window", model_config.preprocessor.window, "hann"),
        ("preprocessor.frame_splicing", model_config.preprocessor.frame_splicing, 1),
        ("preprocessor.n_fft", model_config.preprocessor.get("n_fft", 512), 512),
        ("preprocessor.log", model_config.preprocessor.get("log", True), True),
        (
            "preprocessor.mag_power",
            model_config.preprocessor.get("mag_power", 2.0),
            2.0,
        ),
        (
            "preprocessor.mel_norm",
            model_config.preprocessor.get("mel_norm", "slaney"),
            "slaney",
        ),
        ("preprocessor.pad_to", model_config.preprocessor.get("pad_to", 0), 0),
        (
            "preprocessor.pad_value",
            model_config.preprocessor.get("pad_value", 0.0),
            0.0,
        ),
        ("encoder.subsampling", model_config.encoder.subsampling, "dw_striding"),
        ("encoder.subsampling_factor", model_config.encoder.subsampling_factor, 8),
        (
            "encoder.self_attention_model",
            model_config.encoder.self_attention_model,
            "rel_pos",
        ),
        (
            "encoder.att_context_style",
            model_config.encoder.att_context_style,
            "regular",
        ),
        ("encoder.xscaling", model_config.encoder.xscaling, False),
        ("encoder.untie_biases", model_config.encoder.untie_biases, True),
        ("encoder.use_bias", model_config.encoder.use_bias, False),
        ("encoder.conv_norm_type", model_config.encoder.conv_norm_type, "batch_norm"),
        ("decoder.blank_as_pad", model_config.decoder.blank_as_pad, True),
        ("joint.jointnet.activation", model_config.joint.jointnet.activation, "relu"),
    )
    for name, actual, expected in expected_values:
        if actual != expected:
            raise ValueError(f"Expected {name}={expected}, got {actual}.")

    if args.decoder_type not in TRANSDUCER_DECODER_TYPES:
        raise ValueError(
            f"Parakeet export supports only {TRANSDUCER_DECODER_TYPES}, "
            f"got {args.decoder_type}."
        )

    for name in ("encoder_precision", "decoder_precision"):
        value = getattr(args, name)
        if not isinstance(value, str) or value not in PRECISION_DTYPES:
            raise ValueError(
                f"{name} must be one of {tuple(PRECISION_DTYPES)}, got {value}."
            )

    configured_attention_context = model_config.encoder.att_context_size
    try:
        attention_context = list(configured_attention_context)
    except TypeError as error:
        raise ValueError(
            "Only full-context offline attention is supported; expected "
            f"encoder.att_context_size=[-1, -1], got {configured_attention_context}."
        ) from error
    if attention_context != [-1, -1]:
        raise ValueError(
            "Only full-context offline attention is supported; expected "
            f"encoder.att_context_size=[-1, -1], got {attention_context}."
        )

    positive_float_values = (
        ("preprocessor.window_stride", model_config.preprocessor.window_stride),
        ("preprocessor.window_size", model_config.preprocessor.window_size),
    )
    for name, value in positive_float_values:
        if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Expected {name} to be a positive finite float, got {value}."
            )

    preemph = model_config.preprocessor.get("preemph", 0.97)
    if not isinstance(preemph, float) or not 0.0 <= preemph < 1.0:
        raise ValueError(
            f"Expected preprocessor.preemph to be a float in [0.0, 1.0), got {preemph}."
        )

    low_freq = model_config.preprocessor.get("lowfreq", 0)
    high_freq = model_config.preprocessor.get(
        "highfreq", model_config.preprocessor.sample_rate // 2
    )
    if (
        not isinstance(low_freq, int)
        or not isinstance(high_freq, int)
        or not 0 <= low_freq < high_freq <= model_config.preprocessor.sample_rate // 2
    ):
        raise ValueError(
            "Expected integer Parakeet mel bounds satisfying "
            "0 <= lowfreq < highfreq <= Nyquist, got "
            f"lowfreq={low_freq} and highfreq={high_freq}."
        )

    positive_integer_values = (
        ("preprocessor.features", model_config.preprocessor.features),
        ("encoder.n_layers", model_config.encoder.n_layers),
        ("encoder.d_model", model_config.encoder.d_model),
        (
            "encoder.subsampling_conv_channels",
            model_config.encoder.subsampling_conv_channels,
        ),
        ("encoder.ff_expansion_factor", model_config.encoder.ff_expansion_factor),
        ("encoder.n_heads", model_config.encoder.n_heads),
        ("encoder.pos_emb_max_len", model_config.encoder.pos_emb_max_len),
        ("encoder.conv_kernel_size", model_config.encoder.conv_kernel_size),
        ("decoder.vocab_size", model_config.decoder.vocab_size),
        ("decoder.prednet.pred_hidden", model_config.decoder.prednet.pred_hidden),
        (
            "decoder.prednet.pred_rnn_layers",
            model_config.decoder.prednet.pred_rnn_layers,
        ),
        (
            "joint.jointnet.encoder_hidden",
            model_config.joint.jointnet.encoder_hidden,
        ),
        ("joint.jointnet.joint_hidden", model_config.joint.jointnet.joint_hidden),
        ("joint.num_extra_outputs", model_config.joint.num_extra_outputs),
        ("decoding.greedy.max_symbols", model_config.decoding.greedy.max_symbols),
    )
    for name, value in positive_integer_values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Expected {name} to be a positive integer, got {value}.")

    if model_config.encoder.d_model % 2 != 0:
        raise ValueError(
            "encoder.d_model must be even for relative positional encoding, got "
            f"{model_config.encoder.d_model}."
        )
    if model_config.encoder.d_model != model_config.joint.jointnet.encoder_hidden:
        raise ValueError(
            "encoder.d_model and joint.jointnet.encoder_hidden must match, got "
            f"{model_config.encoder.d_model} and "
            f"{model_config.joint.jointnet.encoder_hidden}."
        )
    if model_config.encoder.d_model % model_config.encoder.n_heads != 0:
        raise ValueError(
            "encoder.d_model must be divisible by encoder.n_heads, got "
            f"d_model={model_config.encoder.d_model} and "
            f"n_heads={model_config.encoder.n_heads}."
        )

    convolution_alignment = 4 if args.encoder_precision == "fp32" else 2
    if model_config.encoder.d_model % convolution_alignment != 0:
        raise ValueError(
            "encoder.d_model must be divisible by "
            f"{convolution_alignment} for {args.encoder_precision} "
            "Parakeet convolution."
        )
    if model_config.encoder.conv_kernel_size % 2 == 0:
        raise ValueError(
            "encoder.conv_kernel_size must be odd, got "
            f"{model_config.encoder.conv_kernel_size}."
        )

    durations = list(model_config.model_defaults.tdt_durations)
    if not durations or any(
        not isinstance(duration, int) or not 0 <= duration <= INT32_MAX
        for duration in durations
    ):
        raise ValueError(
            "model_defaults.tdt_durations must contain non-negative signed "
            "32-bit integers."
        )
    if len(durations) != len(set(durations)):
        raise ValueError("model_defaults.tdt_durations must contain unique values.")
    if len(durations) != model_config.joint.num_extra_outputs:
        raise ValueError(
            "The number of model_defaults.tdt_durations must match "
            "joint.num_extra_outputs."
        )
    if 0 not in durations or all(duration == 0 for duration in durations):
        raise ValueError(
            "model_defaults.tdt_durations must contain zero and at least one "
            "positive duration."
        )

    if not isinstance(args.batch_size, int) or args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")

    audio_seconds = (
        args.min_audio_seconds,
        args.opt_audio_seconds,
        args.max_audio_seconds,
    )
    if (
        not all(isinstance(value, (int, float)) for value in audio_seconds)
        or not 0.0 < audio_seconds[0] <= audio_seconds[1] <= audio_seconds[2]
    ):
        raise ValueError(
            "Expected 0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds, "
            f"got {audio_seconds}."
        )

    if audio_seconds[2] > INT32_MAX / model_config.sample_rate:
        raise ValueError(
            "max_audio_seconds exceeds the signed 32-bit TensorRT audio-sample "
            f"dimension at {model_config.sample_rate} Hz."
        )

    if not isinstance(args.beam, int) or args.beam < 1:
        raise ValueError(f"beam must be positive, got {args.beam}.")
    if args.beam > model_config.decoder.vocab_size:
        raise ValueError(
            f"beam must not exceed decoder.vocab_size, got beam={args.beam} and "
            f"decoder.vocab_size={model_config.decoder.vocab_size}."
        )

    decoder_capacity = args.batch_size * args.beam
    if decoder_capacity > INT32_MAX:
        raise ValueError(
            "batch_size * beam must fit in a signed 32-bit decoder dimension, "
            f"got {decoder_capacity}."
        )

    decoder_dim = model_config.decoder.prednet.pred_hidden
    pred_rnn_layers = model_config.decoder.prednet.pred_rnn_layers
    decoder_tensor_elements = {
        "encoder_output": decoder_capacity * model_config.joint.jointnet.encoder_hidden,
        "targets": decoder_capacity,
        "input_states_1": pred_rnn_layers * decoder_capacity * decoder_dim,
        "input_states_2": pred_rnn_layers * decoder_capacity * decoder_dim,
        "token_log_probs": decoder_capacity * (model_config.decoder.vocab_size + 1),
        "duration_log_probs": decoder_capacity * model_config.joint.num_extra_outputs,
        "output_states_1": pred_rnn_layers * decoder_capacity * decoder_dim,
        "output_states_2": pred_rnn_layers * decoder_capacity * decoder_dim,
    }
    for name, elements in decoder_tensor_elements.items():
        if elements > INT32_MAX:
            raise ValueError(
                f"Parakeet decoder tensor {name} exceeds signed 32-bit indexing: "
                f"{elements} elements, limit={INT32_MAX}."
            )

    model_dim = model_config.encoder.d_model
    feed_forward_dim = model_dim * model_config.encoder.ff_expansion_factor
    joiner_dim = model_config.joint.jointnet.joint_hidden
    vocab_size = model_config.decoder.vocab_size
    num_extra_outputs = model_config.joint.num_extra_outputs
    parameter_tensor_elements = {
        "encoder feed-forward weight": model_dim * feed_forward_dim,
        "decoder embedding weight": (vocab_size + 1) * decoder_dim,
        "decoder recurrent weight": 4 * decoder_dim * decoder_dim,
        "decoder encoder projection weight": model_dim * joiner_dim,
        "decoder output projection weight": (vocab_size + 1 + num_extra_outputs)
        * joiner_dim,
    }
    for name, elements in parameter_tensor_elements.items():
        if elements > INT32_MAX:
            raise ValueError(
                f"The Parakeet {name} exceeds signed 32-bit indexing: "
                f"{elements} elements, limit={INT32_MAX}."
            )

    sample_rate = model_config.sample_rate
    frame_shift_ms = round(model_config.preprocessor.window_stride * 1000)
    frame_length_ms = round(model_config.preprocessor.window_size * 1000)
    win_length = frame_length_ms * sample_rate // 1000
    hop_length = frame_shift_ms * sample_rate // 1000
    if win_length < 2:
        raise ValueError(
            f"preprocessor.window_size produces a {win_length}-sample window; "
            "expected at least 2."
        )
    if hop_length < 1:
        raise ValueError(
            "preprocessor.window_stride must produce at least one sample per frame."
        )
    if frame_shift_ms > frame_length_ms:
        raise ValueError(
            "preprocessor.window_stride must not exceed preprocessor.window_size."
        )

    n_fft = model_config.preprocessor.get("n_fft", 512)
    constructed_n_fft = 2 ** (win_length - 1).bit_length()
    if constructed_n_fft != n_fft:
        raise ValueError(
            f"preprocessor.window_size requires n_fft={constructed_n_fft}, "
            f"but the model config specifies n_fft={n_fft}."
        )

    min_samples = round(args.min_audio_seconds * sample_rate)
    if min_samples < 2 * hop_length:
        raise ValueError(
            f"min_audio_seconds must represent at least {2 * hop_length} samples "
            "for per-feature variance normalization."
        )

    max_samples = round(args.max_audio_seconds * sample_rate)
    feature_frames = max_samples // hop_length + 1
    # Match the feature plugin's in-place R2C buffer and aligned cuBLAS scratch.
    transform_bytes = args.batch_size * feature_frames * (n_fft + 2) * 4
    cublas_offset = (transform_bytes + 255) // 256 * 256
    if cublas_offset + (16 << 20) > INT32_MAX:
        raise ValueError(
            "The maximum Parakeet profile exceeds the feature plugin's signed "
            "32-bit TensorRT workspace limit."
        )

    encoder_frames = (((feature_frames + 1) // 2 + 1) // 2 + 1) // 2
    if encoder_frames > PARAKEET_MAX_ENCODER_FRAMES:
        raise ValueError(
            f"The maximum profile produces {encoder_frames} encoder frames, but "
            "the Parakeet FlashAttention plugin supports at most "
            f"{PARAKEET_MAX_ENCODER_FRAMES}."
        )
    if encoder_frames > model_config.encoder.pos_emb_max_len:
        raise ValueError(
            f"The maximum profile produces {encoder_frames} encoder frames, but "
            "encoder.pos_emb_max_len is only "
            f"{model_config.encoder.pos_emb_max_len}."
        )


def validate_zipformer(
    model_config: DictConfig,
    state_dict: OrderedDict[str, torch.Tensor],
    vocab_size: int,
    args: argparse.Namespace,
) -> None:
    """Validate a Zipformer checkpoint, configuration, and export profile.

    The validator restricts export to the supported offline Icefall Zipformer
    architecture. It checks feature extraction, encoder stack descriptions,
    decoder-head availability, vocabulary dimensions, and the fixed-batch
    TensorRT audio profile.

    Parameters
    ----------
    model_config : DictConfig
        Source configuration loaded from the Zipformer ``config.yaml``.
    state_dict : OrderedDict[str, torch.Tensor]
        Converted checkpoint state dictionary used to reconstruct the export
        modules.
    vocab_size : int
        Decoder vocabulary size represented by the SentencePiece tokenizer and
        model output heads.
    args : argparse.Namespace
        Parsed export arguments containing decoder, batch, beam, duration-profile,
        and TensorRT build settings.

    Raises
    ------
    ValueError
        Raised when the source configuration or an export argument is
        unsupported.
    RuntimeError
        Raised when the checkpoint lacks the selected decoder head or its
        weight and bias shapes disagree with the encoder or decoder dimensions.
    """

    decoder_type = args.decoder_type
    if decoder_type not in DECODER_TYPES:
        raise ValueError(
            f"Zipformer export supports only {DECODER_TYPES}, got {decoder_type}."
        )

    for name in ("encoder_precision", "decoder_precision"):
        value = getattr(args, name)
        if not isinstance(value, str) or value not in PRECISION_DTYPES:
            raise ValueError(
                f"{name} must be one of {tuple(PRECISION_DTYPES)}, got {value}."
            )

    expected_values = (
        (
            "model_params.subsampling_factor",
            model_config.model_params.subsampling_factor,
            4,
        ),
        ("model_params.causal", model_config.model_params.causal, False),
        (
            "model_params.use_attention_decoder",
            model_config.model_params.use_attention_decoder,
            False,
        ),
        (
            "feature_opts.frame_opts.samp_freq",
            model_config.feature_opts.frame_opts.samp_freq,
            16000,
        ),
        (
            "feature_opts.frame_opts.frame_shift_ms",
            model_config.feature_opts.frame_opts.frame_shift_ms,
            10,
        ),
        (
            "feature_opts.frame_opts.frame_length_ms",
            model_config.feature_opts.frame_opts.frame_length_ms,
            25,
        ),
        (
            "feature_opts.frame_opts.dither",
            model_config.feature_opts.frame_opts.dither,
            0.0,
        ),
        (
            "feature_opts.frame_opts.preemph_coeff",
            model_config.feature_opts.frame_opts.preemph_coeff,
            0.97,
        ),
        (
            "feature_opts.frame_opts.window_type",
            model_config.feature_opts.frame_opts.window_type,
            "povey",
        ),
        (
            "feature_opts.frame_opts.snip_edges",
            model_config.feature_opts.frame_opts.snip_edges,
            False,
        ),
    )
    for name, actual, expected in expected_values:
        if actual != expected:
            raise ValueError(f"Expected {name}={expected}, got {actual}.")

    for name in ("use_transducer", "use_ctc"):
        value = model_config.model_params[name]
        if not isinstance(value, bool):
            raise ValueError(
                f"Expected model_params.{name} to be a boolean, got {value}."
            )

    if decoder_type == "ctc_greedy_search" and not model_config.model_params.use_ctc:
        raise ValueError("The model configuration does not enable the CTC head.")
    if (
        decoder_type in ("transducer_greedy_search", "transducer_modified_beam_search")
        and not model_config.model_params.use_transducer
    ):
        raise ValueError("The model configuration does not enable the transducer head.")

    positive_integer_values = (
        ("vocab_size", vocab_size),
        ("min_encoder_input_frames", model_config.min_encoder_input_frames),
        ("feature_opts.mel_opts.num_bins", model_config.feature_opts.mel_opts.num_bins),
        ("model_params.feature_dim", model_config.model_params.feature_dim),
        ("model_params.pos_dim", model_config.model_params.pos_dim),
        ("model_params.decoder_dim", model_config.model_params.decoder_dim),
        ("model_params.joiner_dim", model_config.model_params.joiner_dim),
        ("model_params.context_size", model_config.model_params.context_size),
        ("decoding.beam_size", model_config.decoding.beam_size),
    )
    for name, value in positive_integer_values:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"Expected {name} to be a positive integer, got {value}.")

    if model_config.model_params.pos_dim % 2 != 0:
        raise ValueError(
            "Expected model_params.pos_dim to be even, got "
            f"{model_config.model_params.pos_dim}."
        )

    if decoder_type != "ctc_greedy_search":
        if model_config.model_params.context_size > 2:
            raise ValueError(
                "The predictor context cache supports context_size at most 2, got "
                f"{model_config.model_params.context_size}."
            )

        decoder_dim = model_config.model_params.decoder_dim
        convolution_groups = decoder_dim // 4
        if model_config.model_params.context_size > 1 and (
            convolution_groups < 1 or decoder_dim % convolution_groups != 0
        ):
            raise ValueError(
                "model_params.decoder_dim must produce a positive "
                "grouped-convolution count that divides it exactly, got "
                f"{decoder_dim}."
            )

    if model_config.feature_opts.mel_opts.num_bins < 7:
        raise ValueError(
            "Zipformer convolutional subsampling requires at least seven mel "
            f"bins, got {model_config.feature_opts.mel_opts.num_bins}."
        )

    if model_config.min_encoder_input_frames < 9:
        raise ValueError(
            "Zipformer convolutional subsampling requires at least nine input "
            f"frames, got {model_config.min_encoder_input_frames}."
        )

    low_freq = model_config.feature_opts.mel_opts.low_freq
    high_freq = model_config.feature_opts.mel_opts.high_freq
    if (
        not isinstance(low_freq, int)
        or not isinstance(high_freq, int)
        or not (
            0
            <= low_freq
            < high_freq
            <= model_config.feature_opts.frame_opts.samp_freq // 2
        )
    ):
        raise ValueError(
            "Expected integer Zipformer mel bounds satisfying "
            "0 <= low_freq < high_freq <= Nyquist, got "
            f"low_freq={low_freq} and high_freq={high_freq}."
        )

    if (
        model_config.feature_opts.mel_opts.num_bins
        != model_config.model_params.feature_dim
    ):
        raise ValueError(
            "feature_opts.mel_opts.num_bins and model_params.feature_dim must match."
        )

    sequence_names = (
        "num_encoder_layers",
        "downsampling_factor",
        "feedforward_dim",
        "num_heads",
        "encoder_dim",
        "cnn_module_kernel",
    )
    parsed_sequences: dict[str, list[int]] = {}
    for name in sequence_names:
        value = model_config.model_params[name]
        try:
            values = [int(item) for item in value.split(",")]
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Expected model_params.{name} to contain only integers, got {value}."
            ) from error

        if len(values) != 6 or any(item < 1 for item in values):
            raise ValueError(
                f"Expected model_params.{name} to contain six positive integers, "
                f"got {values}."
            )
        parsed_sequences[name] = values

    encoder_dims = parsed_sequences["encoder_dim"]
    cnn_module_kernels = parsed_sequences["cnn_module_kernel"]

    convolution_alignment = 4 if args.encoder_precision == "fp32" else 2
    if any(dimension % convolution_alignment != 0 for dimension in encoder_dims):
        raise ValueError(
            "Every model_params.encoder_dim value must be divisible by "
            f"{convolution_alignment} for {args.encoder_precision} Zipformer "
            "convolution."
        )

    output_assembly_alignment = 4 if args.encoder_precision == "fp32" else 8
    if any(
        dimension % output_assembly_alignment != 0 for dimension in encoder_dims[3:]
    ):
        raise ValueError(
            "The final three model_params.encoder_dim values must be divisible by "
            f"{output_assembly_alignment} for {args.encoder_precision} Zipformer "
            "output assembly."
        )

    if not (
        encoder_dims[0]
        <= encoder_dims[1]
        <= encoder_dims[2]
        <= encoder_dims[3]
        >= encoder_dims[4]
        >= encoder_dims[5]
    ):
        raise ValueError(
            "model_params.encoder_dim must be nondecreasing through the fourth "
            f"stack and nonincreasing afterward, but got {encoder_dims}."
        )

    if any(kernel_size % 2 == 0 for kernel_size in cnn_module_kernels):
        raise ValueError(
            "Every model_params.cnn_module_kernel value must be odd, got "
            f"{cnn_module_kernels}."
        )

    head_dims: dict[str, int] = {}
    for name in ("query_head_dim", "value_head_dim", "pos_head_dim"):
        value = model_config.model_params[name]
        if not isinstance(value, (int, str)):
            raise ValueError(
                f"Expected model_params.{name} to contain one integer, got {value}."
            )
        try:
            parsed_value = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Expected model_params.{name} to contain one integer, got {value}."
            ) from error

        if parsed_value < 1:
            raise ValueError(
                f"Expected model_params.{name} to contain one positive integer, got "
                f"{parsed_value}."
            )

        head_dims[name] = parsed_value

    if head_dims["pos_head_dim"] != 4:
        raise ValueError(
            "The Zipformer relative-attention TensorRT plugin requires "
            f"model_params.pos_head_dim=4, got {head_dims['pos_head_dim']}."
        )

    use_ctc = decoder_type == "ctc_greedy_search"
    projection_prefix = "ctc_output.1" if use_ctc else "projection_output"
    projection_weight = f"{projection_prefix}.weight"
    projection_bias = f"{projection_prefix}.bias"
    if projection_weight not in state_dict or projection_bias not in state_dict:
        raise RuntimeError(
            f"The checkpoint does not contain the {projection_prefix} head "
            f"required by {decoder_type}."
        )

    weight = state_dict[projection_weight]
    if weight.ndim != 2:
        raise RuntimeError(
            f"The {projection_prefix} weight must have rank 2, got "
            f"shape {tuple(weight.shape)}."
        )

    bias = state_dict[projection_bias]
    output_dim = weight.size(0)
    if bias.ndim != 1 or bias.size(0) != output_dim:
        raise RuntimeError(
            f"The {projection_prefix} bias must have shape ({output_dim},), got "
            f"{tuple(bias.shape)}."
        )

    expected_input_dim, input_dim = max(encoder_dims), weight.size(1)
    if input_dim != expected_input_dim:
        raise RuntimeError(
            f"The {projection_prefix} head accepts {input_dim} input features, but "
            f"model_params.encoder_dim requires {expected_input_dim}."
        )

    if use_ctc and output_dim != vocab_size:
        raise RuntimeError(
            f"The CTC head contains {output_dim} outputs, "
            f"but the decoder vocabulary contains {vocab_size} tokens."
        )

    if not use_ctc and output_dim != model_config.model_params.joiner_dim:
        raise RuntimeError(
            f"The transducer projection contains {output_dim} outputs, but "
            f"model_params.joiner_dim is {model_config.model_params.joiner_dim}."
        )

    if not isinstance(args.batch_size, int) or args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")

    audio_seconds = (
        args.min_audio_seconds,
        args.opt_audio_seconds,
        args.max_audio_seconds,
    )
    if (
        not all(isinstance(value, (int, float)) for value in audio_seconds)
        or not 0.0 < audio_seconds[0] <= audio_seconds[1] <= audio_seconds[2]
    ):
        raise ValueError(
            "Expected 0 < min_audio_seconds <= opt_audio_seconds <= "
            f"max_audio_seconds, got {audio_seconds}."
        )

    if audio_seconds[2] > INT32_MAX / model_config.feature_opts.frame_opts.samp_freq:
        raise ValueError(
            "max_audio_seconds exceeds the signed 32-bit TensorRT audio-sample "
            "dimension at "
            f"{model_config.feature_opts.frame_opts.samp_freq} Hz."
        )

    if not isinstance(args.beam, int) or not 1 <= args.beam <= vocab_size:
        raise ValueError(
            f"beam must be a positive integer between 1 and {vocab_size}, got "
            f"{args.beam}."
        )

    if decoder_type != "ctc_greedy_search":
        decoder_capacity = args.batch_size
        if decoder_type == "transducer_modified_beam_search":
            decoder_capacity *= args.beam
        if decoder_capacity > INT32_MAX:
            raise ValueError(
                "The Zipformer decoder capacity exceeds signed 32-bit indexing: "
                f"{decoder_capacity} hypotheses, limit={INT32_MAX}."
            )

        decoder_tensor_elements = {
            "decoder_input": decoder_capacity * model_config.model_params.joiner_dim,
            "encoder_output": decoder_capacity * model_config.model_params.joiner_dim,
            "tokens_log_prob": decoder_capacity * vocab_size,
        }
        for name, elements in decoder_tensor_elements.items():
            if elements > INT32_MAX:
                raise ValueError(
                    f"Zipformer decoder tensor {name} exceeds signed 32-bit "
                    f"indexing: {elements} elements, limit={INT32_MAX}."
                )

    if args.batch_size > (1 << 16) - 1:
        raise ValueError(
            "batch_size exceeds the Zipformer resampling plugin's CUDA grid.z "
            f"limit of 65535, got {args.batch_size}."
        )

    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    frame_shift_ms = model_config.feature_opts.frame_opts.frame_shift_ms
    frame_length_ms = model_config.feature_opts.frame_opts.frame_length_ms
    frame_length = frame_length_ms * sample_rate // 1000
    frame_shift = frame_shift_ms * sample_rate // 1000

    min_audio_samples = round(args.min_audio_seconds * sample_rate)
    min_feature_frames = (min_audio_samples + frame_shift // 2) // frame_shift
    if min_feature_frames < model_config.min_encoder_input_frames:
        raise ValueError(
            f"min_audio_seconds produces {min_feature_frames} feature frames, but "
            f"the model requires at least {model_config.min_encoder_input_frames}."
        )

    max_audio_samples = round(args.max_audio_seconds * sample_rate)
    max_feature_frames = (max_audio_samples + frame_shift // 2) // frame_shift
    max_encoder_frames = (max_feature_frames - 7) // 2
    if max_encoder_frames > (1 << 16) - 1:
        raise ValueError(
            "max_audio_seconds produces "
            f"{max_encoder_frames} encoder frames, exceeding the Zipformer "
            "resampling plugin's CUDA grid.y limit of 65535."
        )

    fft_length = 2 ** (frame_length - 1).bit_length()
    # Match the feature plugin's in-place R2C buffer and aligned cuBLAS scratch.
    transform_bytes = args.batch_size * max_feature_frames * (fft_length + 2) * 4
    cublas_offset = (transform_bytes + 255) // 256 * 256
    if cublas_offset + (16 << 20) > INT32_MAX:
        raise ValueError(
            "The maximum Zipformer profile exceeds the feature plugin's signed "
            "32-bit TensorRT workspace limit."
        )


def remove_onnx_artifacts(onnx_path: Path) -> None:
    """Remove an ONNX graph and its local external-data artifacts.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX graph. External-data locations referenced by the graph
        are removed from its directory or subdirectories. Missing external-data
        files are ignored; a missing graph raises ``FileNotFoundError``.

    Raises
    ------
    ValueError
        Raised when an external-data location is empty, absolute, or resolves
        outside the ONNX model directory.

    Notes
    -----
    The graph is removed even when parsing or external-data cleanup fails.
    Parsing and filesystem errors propagate to the caller; cleanup is not
    transactional.
    """

    model_dir = onnx_path.parent.resolve()
    try:
        model = onnx.load(onnx_path, load_external_data=False)
        external_data_locations = {
            entry.value
            for tensor in onnx.external_data_helper._get_all_tensors(model)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        for location in external_data_locations:
            external_path = onnx_path.parent / location
            resolved_path = external_path.resolve()
            if (
                not location
                or Path(location).is_absolute()
                or not resolved_path.is_relative_to(model_dir)
            ):
                raise ValueError(
                    "Unsafe ONNX external-data location outside the model "
                    f"directory: {location}."
                )
            external_path.unlink(missing_ok=True)
    finally:
        onnx_path.unlink(missing_ok=True)


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    optimization_level: int,
) -> None:
    """Build and serialize a TensorRT engine from an ONNX graph.

    Every dynamic network input must have exactly one entry in ``profiles``;
    static inputs must not have profile entries. The builder enables TF32,
    sparse-weight optimizations, all tactic sources exposed by the installed
    TensorRT release, and the requested optimization level. The workspace pool
    retains TensorRT's default limit of total device memory; it is not a budget
    based on currently free memory. FP16 and BF16 builder flags are enabled when
    available. Native custom plugins referenced by the ONNX graph are loaded
    before parsing.

    Parameters
    ----------
    onnx_path : Path
        Path to the TensorRT-compatible ONNX graph.
    engine_path : Path
        Destination path for the serialized TensorRT engine.
    profiles : dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
        Dynamic input names mapped to their minimum, optimum, and maximum
        shapes. Pass an empty dictionary for a fully static graph.
    optimization_level : int
        TensorRT builder optimization level from 0 through 5.

    Raises
    ------
    ValueError
        Raised when the optimization level is unsupported, profile names do
        not exactly match the dynamic network inputs, or TensorRT rejects a
        profile or profile shape.
    RuntimeError
        Raised when TensorRT plugins cannot be initialized, the ONNX graph
        cannot be parsed, tactic sources are rejected, or the serialized engine
        cannot be built.
    """

    if not isinstance(optimization_level, int) or not 0 <= optimization_level <= 5:
        raise ValueError(
            "optimization_level must be an integer from 0 through 5, got "
            f"{optimization_level}."
        )

    trt_logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(trt_logger, ""):
        raise RuntimeError("Failed to initialize TensorRT plugins.")

    load_tensorrt_plugins()

    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(
        trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS
    )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    builder_config = builder.create_builder_config()
    builder_config.engine_capability = trt.EngineCapability.STANDARD
    tactic_sources = 0
    for tactic_source in trt.TacticSource.__members__.values():
        tactic_sources |= 1 << int(tactic_source)
    if not builder_config.set_tactic_sources(tactic_sources):
        raise RuntimeError(f"TensorRT rejected tactic source mask {tactic_sources}.")
    builder_config.set_flag(trt.BuilderFlag.TF32)
    builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if hasattr(trt.BuilderFlag, "FP16"):
        builder_config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "BF16"):
        builder_config.set_flag(trt.BuilderFlag.BF16)
    builder_config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)
    builder_config.builder_optimization_level = optimization_level

    dynamic_input_names = {
        network.get_input(index).name
        for index in range(network.num_inputs)
        if any(dimension == -1 for dimension in network.get_input(index).shape)
    }
    if dynamic_input_names != set(profiles):
        raise ValueError(
            f"Expected TensorRT profiles for {sorted(dynamic_input_names)}, got "
            f"{sorted(profiles)}."
        )

    if profiles:
        optimization_profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in profiles.items():
            try:
                optimization_profile.set_shape(
                    input_name, min_shape, opt_shape, max_shape
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid TensorRT profile for {input_name}: "
                    f"{min_shape}, {opt_shape}, {max_shape}."
                ) from error

        if builder_config.add_optimization_profile(optimization_profile) < 0:
            raise ValueError("TensorRT rejected optimization profile.")

    logger.info("Building %s with profiles=%s.", engine_path, profiles)
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine {engine_path}.")

    with open(engine_path, "wb") as engine_file:
        engine_file.write(serialized_engine)
