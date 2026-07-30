#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Validate models, prepare ONNX graphs, and build TensorRT engines.

This module centralizes model validation, graph cleanup, and TensorRT builder
configuration shared by the Parakeet and Zipformer exporters.
"""

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from omegaconf import DictConfig

from ..constants import TRANSDUCER_DECODER_TYPES

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
        TensorRT workspace settings.

    Raises
    ------
    ValueError
        Raised when the source architecture is unsupported or an export
        argument cannot produce a valid fixed-batch TensorRT profile.
    """

    expected_values = (
        ("sample_rate", model_config.sample_rate, 16000),
        ("preprocessor.sample_rate", model_config.preprocessor.sample_rate, 16000),
        ("preprocessor.normalize", model_config.preprocessor.normalize, "per_feature"),
        ("preprocessor.window", model_config.preprocessor.window, "hann"),
        ("preprocessor.frame_splicing", model_config.preprocessor.frame_splicing, 1),
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

    if list(model_config.encoder.att_context_size) != [-1, -1]:
        raise ValueError(
            "Only full-context offline attention is supported; expected "
            f"encoder.att_context_size=[-1, -1], got "
            f"{list(model_config.encoder.att_context_size)}."
        )

    positive_float_values = (
        ("preprocessor.window_stride", model_config.preprocessor.window_stride),
        ("preprocessor.window_size", model_config.preprocessor.window_size),
    )
    for name, value in positive_float_values:
        if not isinstance(value, float) or value <= 0.0:
            raise ValueError(f"Expected {name} to be a positive float, got {value}.")

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
    )
    for name, value in positive_integer_values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Expected {name} to be a positive integer, got {value}.")

    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")

    audio_seconds = (
        args.min_audio_seconds,
        args.opt_audio_seconds,
        args.max_audio_seconds,
    )
    if not 0.0 < audio_seconds[0] <= audio_seconds[1] <= audio_seconds[2]:
        raise ValueError(
            "Expected 0 < min_audio_seconds <= opt_audio_seconds <= "
            f"max_audio_seconds, got {audio_seconds}."
        )

    if args.beam < 1:
        raise ValueError(f"beam must be positive, got {args.beam}.")
    if args.workspace_gib <= 0.0:
        raise ValueError(f"workspace_gib must be positive, got {args.workspace_gib}.")

    sample_rate = model_config.sample_rate
    hop_length = round(model_config.preprocessor.window_stride * sample_rate)
    frame_shift_ms = round(model_config.preprocessor.window_stride * 1000)
    frame_length_ms = round(model_config.preprocessor.window_size * 1000)
    win_length = frame_length_ms * sample_rate // 1000
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

    min_samples = round(args.min_audio_seconds * sample_rate)
    if min_samples < 2 * hop_length:
        raise ValueError(
            f"min_audio_seconds must represent at least {2 * hop_length} samples "
            "for per-feature variance normalization."
        )

    max_samples = round(args.max_audio_seconds * sample_rate)
    feature_frames = max_samples // hop_length + 1
    encoder_frames = (((feature_frames + 1) // 2 + 1) // 2 + 1) // 2
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
        Parsed export arguments containing decoder, batch, beam,
        duration-profile, and TensorRT workspace settings.

    Raises
    ------
    ValueError
        Raised when the source configuration or an export argument is
        unsupported.
    RuntimeError
        Raised when the checkpoint lacks the selected decoder head or its CTC
        output dimension does not match the tokenizer vocabulary.
    """

    decoder_type = args.decoder_type
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
    encoder_dims: list[int] = []
    for name in sequence_names:
        value = model_config.model_params[name]
        try:
            values = [int(item) for item in value.split(",")]
        except ValueError as error:
            raise ValueError(
                f"Expected model_params.{name} to contain only integers, got {value}."
            ) from error
        if len(values) != 6 or any(item < 1 for item in values):
            raise ValueError(
                f"Expected model_params.{name} to contain six positive integers, "
                f"got {values}."
            )
        if name == "encoder_dim":
            encoder_dims = values

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

    for name in ("query_head_dim", "value_head_dim", "pos_head_dim"):
        value = model_config.model_params[name]
        try:
            parsed_value = int(value)
        except ValueError as error:
            raise ValueError(
                f"Expected model_params.{name} to contain one integer, got {value}."
            ) from error
        if parsed_value < 1:
            raise ValueError(
                f"Expected model_params.{name} to contain one positive integer, "
                f"got {parsed_value}."
            )

    use_ctc = decoder_type == "ctc_greedy_search"
    projection_prefix = "ctc_output.1" if use_ctc else "projection_output"
    projection_weight = f"{projection_prefix}.weight"
    if projection_weight not in state_dict:
        raise RuntimeError(
            f"The checkpoint does not contain the {projection_prefix} head "
            f"required by {decoder_type}."
        )

    output_dim = state_dict[projection_weight].size(0)
    if use_ctc and output_dim != vocab_size:
        raise RuntimeError(
            f"The CTC head contains {output_dim} outputs, "
            f"but the decoder vocabulary contains {vocab_size} tokens."
        )

    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")

    audio_seconds = (
        args.min_audio_seconds,
        args.opt_audio_seconds,
        args.max_audio_seconds,
    )
    if not 0.0 < audio_seconds[0] <= audio_seconds[1] <= audio_seconds[2]:
        raise ValueError(
            "Expected 0 < min_audio_seconds <= opt_audio_seconds <= "
            f"max_audio_seconds, got {audio_seconds}."
        )

    if args.beam < 1:
        raise ValueError(f"beam must be positive, got {args.beam}.")
    if args.workspace_gib <= 0.0:
        raise ValueError(f"workspace_gib must be positive, got {args.workspace_gib}.")

    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    frame_shift_ms = model_config.feature_opts.frame_opts.frame_shift_ms
    frame_length_ms = model_config.feature_opts.frame_opts.frame_length_ms
    frame_length = frame_length_ms * sample_rate // 1000
    frame_shift = frame_shift_ms * sample_rate // 1000
    if frame_length < 2:
        raise ValueError(
            f"feature_opts.frame_opts.frame_length_ms produces a "
            f"{frame_length}-sample window; expected at least 2."
        )
    if frame_shift < 1:
        raise ValueError(
            "feature_opts.frame_opts.frame_shift_ms must produce at least "
            "one sample per frame."
        )
    if frame_shift_ms > frame_length_ms:
        raise ValueError(
            "feature_opts.frame_opts.frame_shift_ms must not exceed "
            "feature_opts.frame_opts.frame_length_ms."
        )

    min_audio_samples = round(args.min_audio_seconds * sample_rate)
    min_feature_frames = (min_audio_samples + frame_shift // 2) // frame_shift
    if min_feature_frames < model_config.min_encoder_input_frames:
        raise ValueError(
            f"min_audio_seconds produces {min_feature_frames} feature frames, but "
            f"the model requires at least {model_config.min_encoder_input_frames}."
        )


def remove_onnx_artifacts(onnx_path: Path) -> None:
    """Remove an ONNX graph and its adjacent external-data artifacts.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX graph. External-data locations referenced by the graph
        are removed from the same directory.
    """

    if onnx_path.is_file():
        model = onnx.load(onnx_path, load_external_data=False)
        external_data_locations = {
            entry.value
            for tensor in onnx.external_data_helper._get_all_tensors(model)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        for location in external_data_locations:
            (onnx_path.parent / location).unlink(missing_ok=True)
    onnx_path.unlink(missing_ok=True)


def remove_scatternd_attributes(graph: "onnx.GraphProto") -> None:
    """Remove unsupported no-op ``ScatterND`` reduction attributes in place.

    PyTorch ONNX export can attach ``reduction="none"`` to ``ScatterND`` nodes.
    TensorRT does not accept that optional attribute, although omitting it has
    the same semantics. Nested control-flow graphs are processed recursively.

    Parameters
    ----------
    graph : onnx.GraphProto
        ONNX graph to modify.

    Raises
    ------
    ValueError
        Raised when a ``ScatterND`` node uses a reduction mode other than
        ``"none"``, which cannot be removed without changing graph behavior.
    """

    for node in graph.node:
        if node.op_type == "ScatterND":
            for index in reversed(range(len(node.attribute))):
                attribute = node.attribute[index]
                if attribute.name != "reduction":
                    continue
                if attribute.s != b"none":
                    raise ValueError(
                        "Only ScatterND reduction='none' can be removed safely."
                    )
                node.attribute.pop(index)

        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                remove_scatternd_attributes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    remove_scatternd_attributes(subgraph)


def prepare_onnx_for_tensorrt(input_path: Path, output_path: Path) -> None:
    """Create a TensorRT-compatible copy of an ONNX graph.

    The graph is loaded without materializing external tensor data, then its
    unsupported no-op ``ScatterND`` attributes are removed recursively. External
    tensor references therefore remain external.

    Parameters
    ----------
    input_path : Path
        Path to the source ONNX graph.
    output_path : Path
        Path at which to save the rewritten graph.

    Raises
    ------
    ValueError
        Raised when a ``ScatterND`` node has a reduction mode that cannot be
        removed safely.
    """

    model = onnx.load(input_path, load_external_data=False)
    remove_scatternd_attributes(model.graph)
    onnx.save(model, output_path)


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    workspace_bytes: int,
) -> None:
    """Build and serialize a TensorRT engine from an ONNX graph.

    Every dynamic network input must have exactly one entry in ``profiles``;
    static inputs must not have profile entries. The builder enables TF32,
    sparse-weight optimizations, and optimization level 5. FP16 is enabled
    when the installed TensorRT release exposes its weak-typing builder flag.

    Parameters
    ----------
    onnx_path : Path
        Path to the TensorRT-compatible ONNX graph.
    engine_path : Path
        Destination path for the serialized TensorRT engine.
    profiles : dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
        Dynamic input names mapped to their minimum, optimum, and maximum
        shapes. Pass an empty dictionary for a fully static graph.
    workspace_bytes : int
        Maximum TensorRT workspace size in bytes.

    Raises
    ------
    ValueError
        Raised when profile names do not exactly match the dynamic network
        inputs or when TensorRT rejects a profile shape.
    RuntimeError
        Raised when TensorRT plugins cannot be initialized, the ONNX graph
        cannot be parsed, or the serialized engine cannot be built.
    """

    trt_logger = trt.Logger(trt.Logger.INFO)
    if not trt.init_libnvinfer_plugins(trt_logger, ""):
        raise RuntimeError("Failed to initialize TensorRT plugins.")

    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    builder_config = builder.create_builder_config()
    builder_config.engine_capability = trt.EngineCapability.STANDARD
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    builder_config.set_flag(trt.BuilderFlag.TF32)
    builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if hasattr(trt.BuilderFlag, "FP16"):
        builder_config.set_flag(trt.BuilderFlag.FP16)
    builder_config.builder_optimization_level = 5

    dynamic_input_names = {
        network.get_input(index).name
        for index in range(network.num_inputs)
        if any(dimension == -1 for dimension in network.get_input(index).shape)
    }
    if dynamic_input_names != set(profiles):
        raise ValueError(
            f"Expected TensorRT profiles for {sorted(dynamic_input_names)}, got "
            f"{sorted(profiles)}.",
        )

    if profiles:
        optimization_profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in profiles.items():
            profile_result = optimization_profile.set_shape(
                input_name, min_shape, opt_shape, max_shape
            )
            if profile_result is False:
                raise ValueError(
                    f"Invalid TensorRT profile for {input_name}: "
                    f"{min_shape}, {opt_shape}, {max_shape}."
                )
        builder_config.add_optimization_profile(optimization_profile)

    logger.info("Building %s with profiles=%s.", engine_path, profiles)
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine {engine_path}.")

    with open(engine_path, "wb") as engine_file:
        engine_file.write(serialized_engine)
