#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Export SoundsGoodAI Zipformer checkpoints for batched TensorRT inference.

The exporter reconstructs the condensed offline encoder and transducer decoder
from ``model.pt``, exports fixed-batch ONNX graphs, and builds TensorRT engines.
Only the encoder audio-time dimension is dynamic; batch size and decoder
hypothesis capacity are fixed when the engines are built.
"""

import argparse
import logging
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import DictConfig, OmegaConf

from ..constants import (
    DECODER_TYPES,
    MODEL_CONFIG_FILE,
    MODEL_TYPE_ZIPFORMER,
    ONNX_OPSET_VERSION,
    PRECISION_DTYPES,
    TOKENIZER_FILE,
    ZIPFORMER_DECODER_CONTEXTS_FILE,
    ZIPFORMER_DECODER_ONNX_FILE,
    ZIPFORMER_DECODER_TENSORRT_FILE,
    ZIPFORMER_ONNX_FILE,
    ZIPFORMER_TENSORRT_FILE,
)
from ..utils import validate_model, validate_model_config
from .export_utils import (
    build_tensorrt_engine,
    remove_onnx_artifacts,
    validate_zipformer,
)
from .model.zipformer.decoder import Decoder, Joiner
from .model.zipformer.zipformer import Zipformer2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse Zipformer TensorRT export arguments.

    Returns
    -------
    argparse.Namespace
        Command-line values consumed by :func:`export_zipformer`.
    """

    parser = argparse.ArgumentParser(
        description="Export a published SoundsGoodAI Zipformer model to TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model.pt beside config.yaml and bpe.model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where TensorRT engines and model metadata are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Fixed number of utterances in each inference batch.",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        required=True,
        choices=DECODER_TYPES,
        help="Decoder included in the exported Zipformer bundle.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        required=True,
        help=(
            "Beam width used by modified beam search; greedy-search decoders "
            "force beam 1."
        ),
    )
    parser.add_argument(
        "--encoder-precision",
        type=str,
        choices=tuple(PRECISION_DTYPES),
        default="fp32",
        help=(
            "Floating-point precision used by Zipformer subsampling and encoder "
            "stacks; feature extraction remains fp32."
        ),
    )
    parser.add_argument(
        "--decoder-precision",
        type=str,
        choices=tuple(PRECISION_DTYPES),
        default="fp32",
        help=(
            "Floating-point precision used by the predictor context cache and "
            "transducer joiner."
        ),
    )
    parser.add_argument(
        "--min-audio-seconds",
        type=float,
        required=True,
        help="Minimum audio duration supported by the TensorRT profile.",
    )
    parser.add_argument(
        "--opt-audio-seconds",
        type=float,
        required=True,
        help="Typical audio duration TensorRT should optimize for.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        required=True,
        help="Maximum audio duration supported by the TensorRT profile.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=5,
        help="TensorRT builder optimization level.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep intermediate ONNX models.",
    )
    return parser.parse_args()


def adjust_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert an Icefall Zipformer checkpoint to the condensed model layout.

    Log-domain scale parameters are materialized, learned downsampling logits
    are converted to normalized weights, and module prefixes are renamed.
    Training-only per-layer bypass scales are dropped; the two inference-only
    first-stack tensors are then synthesized.

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.Tensor]
        Icefall model state dictionary.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Converted state dictionary compatible with :class:`Zipformer2` and
        :class:`Decoder`.

    Raises
    ------
    KeyError
        Raised when the checkpoint lacks the first encoder attention projection
        required to infer its channel dimension.
    ValueError
        Raised when a ConvNeXt pointwise convolution does not have a singleton
        two-dimensional kernel or when two source keys map to the same inference
        key.
    """

    adjusted_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    bypass_scale_pattern = re.compile(r"encoder_\d+\.layers\.\d+\.bypass_scale")

    for original_key, original_value in state_dict.items():
        key = original_key
        value = original_value
        if "log_scale" in key:
            key = key.replace("log_scale", "scale")
            value = torch.exp(value)
        if "downsample" in key:
            value = torch.softmax(value, dim=0).unsqueeze(1)

        key = (
            key.replace("encoder_embed", "subsampling")
            .replace("subsampling.conv.0.weight", "subsampling.conv1.weight")
            .replace("subsampling.conv.0.bias", "subsampling.conv1.bias")
            .replace("subsampling.conv.4.weight", "subsampling.conv2.weight")
            .replace("subsampling.conv.4.bias", "subsampling.conv2.bias")
            .replace("subsampling.conv.7.weight", "subsampling.conv3.weight")
            .replace("subsampling.conv.7.bias", "subsampling.conv3.bias")
            .replace("subsampling.convnext.", "subsampling.")
            .replace("downsample.bias", "downsample.weights")
            .replace("downsample_output.bias", "downsample_output.weights")
            .replace("encoder.layers", "layers")
            .replace("encoders.0", "encoder_1")
            .replace("encoders.1", "encoder_2")
            .replace("encoders.2", "encoder_3")
            .replace("encoders.3", "encoder_4")
            .replace("encoders.4", "encoder_5")
            .replace("encoders.5", "encoder_6")
            .replace("out_combiner.bypass_scale", "bypass_scale")
            .replace("joiner.encoder_proj", "projection_output")
        )
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
        if bypass_scale_pattern.fullmatch(key):
            continue
        if key in (
            "subsampling.pointwise_conv1.weight",
            "subsampling.pointwise_conv2.weight",
        ):
            if value.ndim != 4 or value.shape[2:] != (1, 1):
                raise ValueError(
                    f"Expected pointwise Conv2d weight {key} to have shape "
                    f"(out_channels, in_channels, 1, 1), got {tuple(value.shape)}."
                )
            value = value[:, :, 0, 0]
        if key in adjusted_state_dict:
            raise ValueError(
                f"Checkpoint tensor {original_key} and an earlier tensor both map "
                f"to {key}."
            )
        adjusted_state_dict[key] = value

    encoder_1_dim = adjusted_state_dict[
        "encoder_1.layers.0.self_attn_weights.in_proj.weight"
    ].size(1)
    adjusted_state_dict["encoder_1.bypass_scale"] = torch.ones(
        encoder_1_dim, dtype=torch.float32
    )
    adjusted_state_dict["encoder_1.downsample.weights"] = torch.zeros(
        1, 1, dtype=torch.float32
    )
    return adjusted_state_dict


def make_model(
    model_config: DictConfig,
    state_dict: OrderedDict[str, torch.Tensor],
    vocab_size: int,
    pos_emb_max_len: int,
    decoder_type: str,
    subsampling_batch_partitions: int,
    encoder_dtype: torch.dtype,
    decoder_dtype: torch.dtype,
) -> tuple[Zipformer2, Decoder | None, Joiner | None]:
    """Construct condensed Zipformer modules and load checkpoint weights.

    Parameters
    ----------
    model_config : DictConfig
        Validated published model configuration.
    state_dict : OrderedDict[str, torch.Tensor]
        Converted checkpoint state dictionary from :func:`adjust_state_dict`.
    vocab_size : int
        Number of output tokens represented by the selected checkpoint head.
    pos_emb_max_len : int
        Maximum positional encoding length compiled into the encoder.
    decoder_type : str
        Decoder whose output head is included in the exported bundle.
    subsampling_batch_partitions : int
        Number of batch partitions used by the convolutional subsampling
        frontend.
    encoder_dtype : torch.dtype
        Floating-point dtype used by the encoder.
    decoder_dtype : torch.dtype
        Floating-point dtype used by the predictor context cache and joiner.

    Returns
    -------
    tuple[Zipformer2, Decoder | None, Joiner | None]
        Evaluation-mode waveform encoder and, for transducer modes, separate
        predictor and joiner modules.

    Raises
    ------
    RuntimeError
        Raised when checkpoint keys or tensor shapes do not exactly match the
        reconstructed architecture.
    """

    model_params = model_config.model_params
    encoder_dims = [int(value) for value in model_params.encoder_dim.split(",")]
    num_encoder_layers = [
        int(value) for value in model_params.num_encoder_layers.split(",")
    ]
    downsampling_factors = [
        int(value) for value in model_params.downsampling_factor.split(",")
    ]
    num_heads = [int(value) for value in model_params.num_heads.split(",")]
    feedforward_dims = [int(value) for value in model_params.feedforward_dim.split(",")]
    cnn_module_kernels = [
        int(value) for value in model_params.cnn_module_kernel.split(",")
    ]
    bypass_scales = [
        state_dict[f"encoder_{index}.bypass_scale"] for index in range(1, 7)
    ]

    use_ctc = decoder_type == "ctc_greedy_search"
    projection_prefix = "ctc_output.1" if use_ctc else "projection_output"
    output_dim = state_dict[f"{projection_prefix}.weight"].size(0)

    frame_opts = model_config.feature_opts.frame_opts
    mel_opts = model_config.feature_opts.mel_opts
    encoder = Zipformer2(
        samp_freq=frame_opts.samp_freq,
        frame_shift_ms=frame_opts.frame_shift_ms,
        frame_length_ms=frame_opts.frame_length_ms,
        feature_dim=model_params.feature_dim,
        preemph=frame_opts.preemph_coeff,
        low_freq=mel_opts.low_freq,
        high_freq=mel_opts.high_freq,
        min_frames=model_config.min_encoder_input_frames,
        subsample_output_dim=state_dict["subsampling.out.weight"].size(0),
        subsample_layer1_channels=state_dict["subsampling.conv1.weight"].size(0),
        subsample_layer2_channels=state_dict["subsampling.conv2.weight"].size(0),
        subsample_layer3_channels=state_dict["subsampling.conv3.weight"].size(0),
        subsampling_batch_partitions=subsampling_batch_partitions,
        encoder_dims=encoder_dims,
        num_encoder_layers=num_encoder_layers,
        downsampling_factors=downsampling_factors,
        bypass_scales=bypass_scales,
        num_heads=num_heads,
        feedforward_dims=feedforward_dims,
        cnn_module_kernels=cnn_module_kernels,
        query_head_dim=int(model_params.query_head_dim),
        pos_head_dim=int(model_params.pos_head_dim),
        value_head_dim=int(model_params.value_head_dim),
        pos_dim=model_params.pos_dim,
        pos_max_len=pos_emb_max_len,
        output_dim=output_dim,
        use_ctc=use_ctc,
        dtype=encoder_dtype,
    )
    encoder_state_dict = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key.startswith(("encoder_", "subsampling.", "downsample_output."))
    )
    encoder_state_dict["projection_output.weight"] = state_dict[
        f"{projection_prefix}.weight"
    ]
    encoder_state_dict["projection_output.bias"] = state_dict[
        f"{projection_prefix}.bias"
    ]
    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder.eval()

    if use_ctc:
        return encoder, None, None

    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_dim=model_params.decoder_dim,
        joiner_dim=model_params.joiner_dim,
        context_size=model_params.context_size,
        dtype=decoder_dtype,
    )
    decoder_state_dict = OrderedDict(
        (
            key.replace("decoder.embedding", "embedding")
            .replace("decoder.conv", "conv")
            .replace("joiner.decoder_proj", "decoder_proj"),
            value,
        )
        for key, value in state_dict.items()
        if key.startswith(
            ("decoder.embedding.", "decoder.conv.", "joiner.decoder_proj.")
        )
    )
    decoder.load_state_dict(decoder_state_dict, strict=True)
    decoder.eval()

    joiner = Joiner(model_params.joiner_dim, vocab_size, dtype=decoder_dtype)
    joiner_state_dict = OrderedDict(
        (key.replace("joiner.output_linear", "output_proj"), value)
        for key, value in state_dict.items()
        if key.startswith("joiner.output_linear.")
    )
    joiner.load_state_dict(joiner_state_dict, strict=True)
    joiner.eval()

    return encoder, decoder, joiner


def get_subsampling_batch_partitions(
    model_config: DictConfig, layer3_channels: int, args: argparse.Namespace
) -> int:
    """Determine a CASK-safe batch partition count for subsampling.

    TensorRT CASK convolution tactics use signed 32-bit element offsets. This
    helper derives the third-convolution output shape from the maximum audio
    profile and partitions the batch when that tensor would contain more than
    ``2**31`` elements. The third-convolution output is also the depthwise
    convolution input and output, so the resulting partition count protects
    both convolutions.

    Parameters
    ----------
    model_config : DictConfig
        Validated source Zipformer configuration containing feature extraction
        and model dimensions.
    layer3_channels : int
        Number of output channels shared by the third subsampling convolution
        output and the following depthwise convolution.
    args : argparse.Namespace
        Validated export arguments containing the batch size and maximum audio
        duration.

    Returns
    -------
    int
        Smallest partition count for which every subsampling batch partition
        stays within the CASK element limit.
    """

    if not isinstance(layer3_channels, int) or layer3_channels < 1:
        raise ValueError(
            f"layer3_channels must be a positive integer, got {layer3_channels}."
        )

    frame_opts = model_config.feature_opts.frame_opts
    frame_length = frame_opts.frame_length_ms * frame_opts.samp_freq // 1000
    frame_shift = frame_opts.frame_shift_ms * frame_opts.samp_freq // 1000
    conv_features = model_config.model_params.feature_dim

    max_samples = round(args.max_audio_seconds * frame_opts.samp_freq)
    reflected_samples = max_samples + frame_length - frame_shift // 2
    feature_frames = (reflected_samples - frame_length) // frame_shift + 1
    conv_frames = (feature_frames - 5) // 2 - 1
    conv_features = ((conv_features - 3) // 2 - 2) // 2 + 1
    elements_per_item = layer3_channels * conv_frames * conv_features

    cask_element_limit = 1 << 31
    if elements_per_item > cask_element_limit:
        raise ValueError(
            "One Zipformer subsampling item exceeds TensorRT's CASK element "
            f"limit: {elements_per_item} elements, limit={cask_element_limit}."
        )

    partitions = (
        args.batch_size * elements_per_item + cask_element_limit - 1
    ) // cask_element_limit
    while (
        args.batch_size + partitions - 1
    ) // partitions * elements_per_item > cask_element_limit:
        partitions += 1

    return partitions


def make_runtime_config(
    model_config: DictConfig,
    vocab_size: int,
    blank_id: int,
    pos_emb_max_len: int,
    args: argparse.Namespace,
) -> DictConfig:
    """Build the compact runtime configuration stored in a Zipformer bundle.

    Parameters
    ----------
    model_config : DictConfig
        Validated source Zipformer configuration.
    vocab_size : int
        Number of output tokens represented by the selected checkpoint head.
    blank_id : int
        Tokenizer ID of the Zipformer blank token.
    pos_emb_max_len : int
        Maximum positional encoding length compiled into the encoder.
    args : argparse.Namespace
        Validated export arguments.

    Returns
    -------
    DictConfig
        Runtime configuration consumed by :class:`fast_gpu_asr.ASR`.
    """

    model_params = model_config.model_params
    frame_opts = model_config.feature_opts.frame_opts
    right_padding_samples = frame_opts.frame_length_ms * frame_opts.samp_freq // 2000
    decoder_params = {"beam": args.beam, "blank_penalty": 0.0}
    if args.decoder_type != "ctc_greedy_search":
        decoder_params.update(
            {
                "context_size": model_params.context_size,
                "decoder_dim": model_params.decoder_dim,
                "joiner_dim": model_params.joiner_dim,
            },
        )

    runtime_config = OmegaConf.create(
        {
            "model_type": MODEL_TYPE_ZIPFORMER,
            "decoder_type": args.decoder_type,
            "model_samplerate": frame_opts.samp_freq,
            "vocab_size": vocab_size,
            "blank_id": blank_id,
            "audio_encoder_params": {
                "feature_dim": model_params.feature_dim,
                "encoder_dims": [
                    int(value) for value in model_params.encoder_dim.split(",")
                ],
                "num_encoder_layers": [
                    int(value) for value in model_params.num_encoder_layers.split(",")
                ],
                "downsampling_factors": [
                    int(value) for value in model_params.downsampling_factor.split(",")
                ],
                "feedforward_dims": [
                    int(value) for value in model_params.feedforward_dim.split(",")
                ],
                "pos_emb_max_len": pos_emb_max_len,
                "output_dim": (
                    vocab_size
                    if args.decoder_type == "ctc_greedy_search"
                    else model_params.joiner_dim
                ),
                "frame_shift_ms": frame_opts.frame_shift_ms,
                "right_padding_samples": right_padding_samples,
                "subsampling_factor": model_params.subsampling_factor,
                "use_ctc": args.decoder_type == "ctc_greedy_search",
                "min_audio_seconds": args.min_audio_seconds,
                "opt_audio_seconds": args.opt_audio_seconds,
                "max_audio_seconds": args.max_audio_seconds,
            },
            "decoder_params": decoder_params,
        },
    )
    validate_model_config(runtime_config)
    return runtime_config


def export_model_to_onnx(
    encoder: Zipformer2,
    decoder: Decoder | None,
    joiner: Joiner | None,
    model_config: DictConfig,
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    """Export fixed-batch Zipformer encoder and decoder ONNX graphs.

    The encoder accepts a fixed number of padded waveforms while allowing their
    shared sample capacity to vary. Its input includes reflected right context,
    while ``audio_lengths`` retains the unpadded waveform lengths. For
    transducer modes, the predictor is precomputed into a context table and the
    exported ONNX decoder contains the fully static joiner with
    ``batch_size * beam`` hypothesis slots.

    Parameters
    ----------
    encoder : Zipformer2
        Loaded waveform-to-encoder model.
    decoder : Decoder | None
        Loaded stateless transducer predictor, or ``None`` for CTC.
    joiner : Joiner | None
        Loaded transducer joiner, or ``None`` for CTC.
    model_config : DictConfig
        Validated published model configuration.
    args : argparse.Namespace
        Export settings, including output directory, fixed batch size, beam,
        and optimal profile duration.

    Returns
    -------
    tuple[Path, Path | None]
        Paths to the encoder and optional transducer decoder ONNX graphs.

    Raises
    ------
    RuntimeError
        Raised when a transducer decoder is provided without its joiner.
    """

    encoder_path = args.output_dir / ZIPFORMER_ONNX_FILE
    decoder_path = args.output_dir / ZIPFORMER_DECODER_ONNX_FILE

    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    # Use the profile optimum as the Dynamo example. At the minimum duration,
    # subsampling produces singleton dimensions that ONNX shape propagation may
    # specialize even though waveform time is dynamic.
    audio_samples = round(args.opt_audio_seconds * sample_rate)
    right_padding_samples = (
        model_config.feature_opts.frame_opts.frame_length_ms * sample_rate // 2000
    )
    audio = torch.zeros(
        args.batch_size, audio_samples + right_padding_samples, dtype=torch.float32
    )
    audio_lengths = torch.full((args.batch_size,), audio_samples, dtype=torch.int64)

    logger.info("Exporting the batched Zipformer encoder to %s.", encoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            encoder,
            (audio, audio_lengths),
            encoder_path,
            dynamic_shapes={
                "audio": {1: torch.export.Dim.DYNAMIC},
                "audio_lengths": {},
            },
            input_names=("audio", "audio_lengths"),
            output_names=("encoder_output", "encoder_output_lengths"),
            opset_version=ONNX_OPSET_VERSION,
        )

    if decoder is None:
        return encoder_path, None
    if joiner is None:
        raise RuntimeError("The Zipformer transducer joiner was not initialized.")

    context_lookup = decoder.make_context_lookup(chunk_size=8192)
    context_lookup_path = args.output_dir / ZIPFORMER_DECODER_CONTEXTS_FILE
    logger.info(
        "Writing %s %s predictor contexts to %s.",
        context_lookup.size(0),
        context_lookup.dtype,
        context_lookup_path,
    )
    torch.save(context_lookup, context_lookup_path)
    decoder_batch = args.batch_size
    if args.decoder_type == "transducer_modified_beam_search":
        decoder_batch *= args.beam
    logger.info("Exporting the batched Zipformer decoder to %s.", decoder_path)
    decoder_dtype = joiner.output_proj.weight.dtype
    with torch.inference_mode():
        torch.onnx.export(
            joiner,
            (
                torch.zeros(
                    decoder_batch,
                    model_config.model_params.joiner_dim,
                    dtype=decoder_dtype,
                ),
                torch.zeros(
                    decoder_batch,
                    model_config.model_params.joiner_dim,
                    dtype=decoder_dtype,
                ),
            ),
            decoder_path,
            input_names=("decoder_input", "encoder_output"),
            output_names=("tokens_log_prob",),
            opset_version=ONNX_OPSET_VERSION,
        )

    return encoder_path, decoder_path


def export_zipformer(args: argparse.Namespace) -> None:
    """Run the complete offline Zipformer TensorRT export.

    The source checkpoint, configuration, and tokenizer are read from one model
    directory. The tokenizer and a compact runtime configuration are written
    to the destination, checkpoint weights are loaded into condensed inference
    modules, and ONNX graphs are exported before TensorRT engines are built.
    Transducer bundles also contain a precomputed predictor context table.
    Intermediate ONNX artifacts are retained only in debug mode.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed export arguments from :func:`parse_args`.

    Raises
    ------
    FileNotFoundError
        Raised when ``model.pt``, ``config.yaml``, or ``bpe.model`` is missing.
    ValueError
        Raised when the model configuration or TensorRT profile is unsupported.
    RuntimeError
        Raised when checkpoint loading, ONNX conversion, or TensorRT engine
        construction fails.
    KeyError
        Raised when the checkpoint lacks its model state or a required tensor.
    """

    if args.decoder_type in ("transducer_greedy_search", "ctc_greedy_search"):
        if args.beam != 1:
            logger.warning(
                "Overriding beam=%s with beam=1 because %s requires beam 1.",
                args.beam,
                args.decoder_type,
            )
        args.beam = 1

    model_dir = args.model_path.parent
    config_path = model_dir / "config.yaml"
    tokenizer_path = model_dir / TOKENIZER_FILE
    for required_path in (config_path, args.model_path, tokenizer_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing required model file: {required_path}.")

    output_dir = args.output_dir.resolve()
    for required_path in (config_path, args.model_path, tokenizer_path):
        if required_path.resolve().is_relative_to(output_dir):
            raise ValueError(
                f"Output directory {args.output_dir} contains required source file "
                f"{required_path}."
            )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    model_config = OmegaConf.load(config_path)

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = tokenizer.vocab_size() - 1
    if tokenizer.unk_id() != vocab_size:
        logger.warning(
            "Expected the SentencePiece unknown token to be the final vocabulary "
            "item, but got unk_id=%s and vocab_size=%s; retaining all entries.",
            tokenizer.unk_id(),
            vocab_size + 1,
        )
        vocab_size += 1

    blank_id = tokenizer.piece_to_id("<blk>")
    if not 0 <= blank_id < vocab_size or tokenizer.id_to_piece(blank_id) != "<blk>":
        raise ValueError(
            "The Zipformer tokenizer must contain an exact <blk> piece inside "
            f"the decoder vocabulary, got ID {blank_id}."
        )

    logger.info("Loading Zipformer checkpoint %s.", args.model_path)
    checkpoint = torch.load(
        args.model_path, map_location=torch.device("cpu"), weights_only=True
    )["model"]
    state_dict = adjust_state_dict(OrderedDict(checkpoint.items()))
    del checkpoint
    validate_zipformer(model_config, state_dict, vocab_size, args)

    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    frame_shift = (
        model_config.feature_opts.frame_opts.frame_shift_ms * sample_rate // 1000
    )
    max_audio_samples = round(args.max_audio_seconds * sample_rate)
    pos_emb_max_len = ((max_audio_samples + frame_shift // 2) // frame_shift + 1) // 2
    subsampling_batch_partitions = get_subsampling_batch_partitions(
        model_config, state_dict["subsampling.conv3.weight"].size(0), args
    )
    if subsampling_batch_partitions > 1:
        logger.info(
            "Splitting Zipformer convolutional subsampling into %s batch partitions "
            "to stay within TensorRT's CASK element limit.",
            subsampling_batch_partitions,
        )
    encoder, decoder, joiner = make_model(
        model_config,
        state_dict,
        vocab_size,
        pos_emb_max_len,
        args.decoder_type,
        subsampling_batch_partitions,
        PRECISION_DTYPES[args.encoder_precision],
        PRECISION_DTYPES[args.decoder_precision],
    )
    del state_dict

    shutil.copyfile(tokenizer_path, args.output_dir / TOKENIZER_FILE)
    runtime_config = make_runtime_config(
        model_config, vocab_size, blank_id, pos_emb_max_len, args
    )
    logger.info("Zipformer runtime config:\n%s", OmegaConf.to_yaml(runtime_config))
    OmegaConf.save(runtime_config, args.output_dir / MODEL_CONFIG_FILE)

    encoder_onnx_path, decoder_onnx_path = export_model_to_onnx(
        encoder, decoder, joiner, model_config, args
    )
    del encoder, decoder, joiner

    audio_samples = (
        round(args.min_audio_seconds * sample_rate),
        round(args.opt_audio_seconds * sample_rate),
        round(args.max_audio_seconds * sample_rate),
    )
    right_padding_samples = (
        model_config.feature_opts.frame_opts.frame_length_ms * sample_rate // 2000
    )
    encoder_profiles = {
        "audio": tuple(
            (args.batch_size, samples + right_padding_samples)
            for samples in audio_samples
        ),
    }
    build_tensorrt_engine(
        encoder_onnx_path,
        args.output_dir / ZIPFORMER_TENSORRT_FILE,
        encoder_profiles,
        args.optimization_level,
    )
    if decoder_onnx_path is not None:
        build_tensorrt_engine(
            decoder_onnx_path,
            args.output_dir / ZIPFORMER_DECODER_TENSORRT_FILE,
            {},
            args.optimization_level,
        )

    if not args.debug:
        remove_onnx_artifacts(encoder_onnx_path)
        if decoder_onnx_path is not None:
            remove_onnx_artifacts(decoder_onnx_path)

    published_config = OmegaConf.load(args.output_dir / MODEL_CONFIG_FILE)
    validate_model(args.output_dir, published_config)

    logger.info("Zipformer TensorRT export completed in %s.", args.output_dir)


def main() -> None:
    """Configure logging and run the command-line Zipformer exporter."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    export_zipformer(parse_args())


if __name__ == "__main__":
    main()
