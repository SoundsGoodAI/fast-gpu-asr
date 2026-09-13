#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Export NVIDIA Parakeet TDT and CTC checkpoints for batched TensorRT inference.

The exporter reads a model archive, reconstructs the condensed Parakeet encoder
and the selected TDT decoder or CTC head, exports ONNX graphs, and builds engines.
Only audio duration is dynamic; batch size and decoder hypothesis capacity
are fixed when the engines are built.
"""

import argparse
import logging
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from ..constants import (
    DECODER_TYPES,
    MODEL_CONFIG_FILE,
    MODEL_TYPE_PARAKEET,
    ONNX_OPSET_VERSION,
    PARAKEET_DECODER_ONNX_FILE,
    PARAKEET_DECODER_TENSORRT_FILE,
    PARAKEET_ONNX_FILE,
    PARAKEET_TENSORRT_FILE,
    PRECISION_DTYPES,
    TOKENIZER_FILE,
)
from ..utils import validate_model, validate_model_config
from .export_utils import (
    build_tensorrt_engine,
    remove_onnx_artifacts,
    validate_parakeet,
)
from .model.parakeet.decoder import Decoder
from .model.parakeet.parakeet import ParakeetEncoder

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse Parakeet TensorRT export arguments.

    Returns
    -------
    argparse.Namespace
        Command-line values consumed by :func:`export_parakeet`.
    """

    parser = argparse.ArgumentParser(
        description="Export an NVIDIA Parakeet TDT or CTC .nemo model to TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the source Parakeet TDT or CTC .nemo archive.",
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
        help="Decoder included in the exported Parakeet bundle.",
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
        "--blank-penalty",
        type=float,
        default=0.0,
        help=(
            "Value subtracted from blank-token log probabilities during CTC or "
            "TDT decoding. Positive values discourage blanks; zero leaves "
            "scores unchanged. Must be a finite float32 value."
        ),
    )
    parser.add_argument(
        "--encoder-precision",
        type=str,
        choices=tuple(PRECISION_DTYPES),
        default="fp32",
        help=(
            "Floating-point precision used by Parakeet subsampling and Conformer "
            "layers; features and CTC logits remain fp32. With input scaling, "
            "fp16 rescales the first block's weights and normalization epsilon "
            "to keep its residual in range."
        ),
    )
    parser.add_argument(
        "--decoder-precision",
        type=str,
        choices=tuple(PRECISION_DTYPES),
        default="fp32",
        help=(
            "Floating-point precision used internally by the Parakeet prediction "
            "network and joiner; unused for CTC."
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
        "--debug", action="store_true", help="Keep intermediate ONNX models."
    )
    return parser.parse_args()


def extract_member(archive: tarfile.TarFile, filename: str, output_dir: Path) -> Path:
    """Extract one required file from a Parakeet archive.

    Parameters
    ----------
    archive : tarfile.TarFile
        Open Parakeet archive.
    filename : str
        Required member basename.
    output_dir : Path
        Directory where the member is extracted using its basename.

    Returns
    -------
    Path
        Path to the extracted file.

    Raises
    ------
    FileNotFoundError
        Raised when no matching member exists or its contents cannot be read.
    ValueError
        Raised when multiple regular archive members have the requested
        basename.
    """

    basename = Path(filename).name
    members = [
        candidate
        for candidate in archive.getmembers()
        if candidate.isfile() and Path(candidate.name).name == basename
    ]
    if not members:
        raise FileNotFoundError(f"Missing {filename} in Parakeet archive.")
    if len(members) > 1:
        member_names = sorted(member.name for member in members)
        raise ValueError(
            f"Expected one {filename} in Parakeet archive, got {member_names}."
        )

    member = members[0]

    source = archive.extractfile(member)
    if source is None:
        raise FileNotFoundError(f"Unable to read {member.name} from Parakeet archive.")

    output_path = output_dir / basename
    with source, open(output_path, "wb") as output_file:
        shutil.copyfileobj(source, output_file)

    return output_path


def extract_parakeet_archive(
    model_path: Path, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Extract files required to reconstruct a Parakeet model.

    The tokenizer filename is read from the extracted Parakeet configuration so
    archives using a ``nemo:`` tokenizer URI are handled correctly.

    Parameters
    ----------
    model_path : Path
        Path to the source ``.nemo`` tar archive.
    output_dir : Path
        Temporary directory where required members are extracted.

    Returns
    -------
    tuple[Path, Path, Path]
        Paths to ``model_config.yaml``, ``model_weights.ckpt``, and the
        SentencePiece tokenizer, in that order.

    Raises
    ------
    FileNotFoundError
        Raised when the archive lacks a required model file.
    tarfile.TarError
        Raised when the source is not a readable tar archive.
    """

    with tarfile.open(model_path) as archive:
        config_path = extract_member(archive, "model_config.yaml", output_dir)
        checkpoint_path = extract_member(archive, "model_weights.ckpt", output_dir)
        model_config = OmegaConf.load(config_path)
        tokenizer_path = str(model_config.tokenizer.model_path).removeprefix("nemo:")
        tokenizer_path = extract_member(archive, Path(tokenizer_path).name, output_dir)

    return config_path, checkpoint_path, tokenizer_path


def adjust_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert Parakeet checkpoint keys to the condensed model layout.

    Prediction-network and joiner prefixes are renamed to match :class:`Decoder`.
    Pointwise Conv1d weights are squeezed to the matrix shape expected by the
    equivalent linear layers in the condensed encoder. Attention query, key,
    and value weights and biases are concatenated for one fused projection.
    The Macaron FFN output factor is folded into each second projection to
    remove runtime scaling.

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.Tensor]
        State dictionary extracted from ``model_weights.ckpt``. FFN output
        weights and biases are scaled in place; convert each loaded checkpoint
        only once.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Converted state dictionary compatible with the local export modules.

    Raises
    ------
    ValueError
        Raised when renamed checkpoint keys collide, attention projections are
        incomplete or incompatible, or a pointwise convolution cannot be
        represented by a linear layer.
    """

    adjusted_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    adjusted_sources: dict[str, str] = {}
    for source_key, value in state_dict.items():
        key = (
            source_key.replace("decoder.prediction.embed", "embedding")
            .replace("decoder.prediction.dec_rnn.lstm", "lstm")
            .replace("joint.pred", "decoder_proj")
            .replace("joint.enc", "encoder_proj")
            .replace("joint.joint_net.2", "output_proj")
            .replace("decoder.decoder_layers.0", "projection_output")
            .replace("encoder.pre_encode.conv.0", "encoder.pre_encode.conv1")
            .replace("encoder.pre_encode.conv.2", "encoder.pre_encode.conv2")
            .replace("encoder.pre_encode.conv.3", "encoder.pre_encode.pointwise_conv1")
            .replace("encoder.pre_encode.conv.5", "encoder.pre_encode.conv3")
            .replace("encoder.pre_encode.conv.6", "encoder.pre_encode.pointwise_conv2")
        )
        if key == "projection_output.weight" or key.endswith(
            ("conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight")
        ):
            if value.ndim != 3 or value.size(2) != 1:
                raise ValueError(
                    f"Expected pointwise Conv1d weight {key} to have shape "
                    f"(out_channels, in_channels, 1), got {tuple(value.shape)}."
                )
            value = value.squeeze(2)
        if key.endswith(
            (
                "feed_forward1.linear2.weight",
                "feed_forward2.linear2.weight",
                "feed_forward1.linear2.bias",
                "feed_forward2.linear2.bias",
            )
        ):
            value *= 0.5

        if key in adjusted_state_dict:
            raise ValueError(
                f"Checkpoint keys {adjusted_sources[key]} and {source_key} both "
                f"map to {key}."
            )

        adjusted_state_dict[key] = value
        adjusted_sources[key] = source_key

    for parameter in ("weight", "bias"):
        projection_suffixes = {
            "query": f".self_attn.linear_q.{parameter}",
            "key": f".self_attn.linear_k.{parameter}",
            "value": f".self_attn.linear_v.{parameter}",
        }
        projection_prefixes = {
            key.removesuffix(suffix)
            for key in adjusted_state_dict
            for suffix in projection_suffixes.values()
            if key.endswith(suffix)
        }
        for prefix in sorted(projection_prefixes):
            query_key = f"{prefix}{projection_suffixes['query']}"
            key_key = f"{prefix}{projection_suffixes['key']}"
            value_key = f"{prefix}{projection_suffixes['value']}"
            qkv_key = f"{prefix}.self_attn.linear_qkv.{parameter}"

            missing_keys = tuple(
                key
                for key in (query_key, key_key, value_key)
                if key not in adjusted_state_dict
            )
            if missing_keys:
                raise ValueError(
                    f"Missing attention projection companions {missing_keys} for "
                    f"{prefix}."
                )
            if qkv_key in adjusted_state_dict:
                raise ValueError(
                    "Checkpoint contains both split attention projections and "
                    f"{qkv_key}."
                )

            projections = (
                adjusted_state_dict[query_key],
                adjusted_state_dict[key_key],
                adjusted_state_dict[value_key],
            )
            rank = 2 if parameter == "weight" else 1
            if any(projection.ndim != rank for projection in projections) or any(
                projection.shape != projections[0].shape
                or projection.dtype != projections[0].dtype
                or projection.device != projections[0].device
                for projection in projections[1:]
            ):
                projection_metadata = tuple(
                    (tuple(projection.shape), projection.dtype, projection.device)
                    for projection in projections
                )
                raise ValueError(
                    f"Expected matching rank-{rank} query, key, and value projection "
                    f"{parameter}s for {prefix}, got {projection_metadata}."
                )

            for projection_key in (query_key, key_key, value_key):
                del adjusted_state_dict[projection_key]

            adjusted_state_dict[qkv_key] = torch.cat(projections, dim=0)

    return adjusted_state_dict


def make_model(
    model_config: DictConfig,
    state_dict: OrderedDict[str, torch.Tensor],
    subsampling_batch_partitions: int,
    encoder_dtype: torch.dtype,
    decoder_dtype: torch.dtype,
    decoder_type: str,
) -> tuple[ParakeetEncoder, Decoder | None]:
    """Construct condensed Parakeet modules and load checkpoint weights.

    For FP32/BF16, checkpoint input scaling is folded into the subsampling
    projection's weights and bias. FP16 instead divides the first block's branch
    output weights/biases by the input scale and its LayerNorm epsilons by the
    squared scale. This keeps the residual unscaled and avoids FP16 overflow
    without an FP32 first block. Weight scaling happens in place before dtype
    conversion; construct each model from a freshly loaded checkpoint.

    Parameters
    ----------
    model_config : DictConfig
        Validated Parakeet model configuration.
    state_dict : OrderedDict[str, torch.Tensor]
        Converted checkpoint state dictionary from :func:`adjust_state_dict`.
        Input-scaling weights and biases are modified in place when enabled.
    subsampling_batch_partitions : int
        Number of batch partitions used across the three-convolution
        subsampling frontend to avoid TensorRT's CASK ``int32`` element limit.
    encoder_dtype : torch.dtype
        Floating-point dtype used internally by the encoder.
    decoder_dtype : torch.dtype
        Floating-point dtype used internally by the prediction network and joiner.
    decoder_type : str
        Export CTC log probabilities or TDT encoder embeddings and decoder.

    Returns
    -------
    tuple[ParakeetEncoder, Decoder | None]
        Evaluation-mode waveform encoder and optional TDT decoder.

    Raises
    ------
    RuntimeError
        Raised when checkpoint keys or tensor shapes do not exactly match the
        reconstructed architecture.
    """

    encoder = ParakeetEncoder(
        samp_freq=model_config.preprocessor.sample_rate,
        frame_shift_ms=round(model_config.preprocessor.window_stride * 1000),
        frame_length_ms=round(model_config.preprocessor.window_size * 1000),
        feature_dim=model_config.preprocessor.features,
        preemph=model_config.preprocessor.get("preemph", 0.97),
        low_freq=model_config.preprocessor.get("lowfreq", 0),
        high_freq=model_config.preprocessor.get(
            "highfreq", model_config.preprocessor.sample_rate // 2
        ),
        n_layers=model_config.encoder.n_layers,
        model_dim=model_config.encoder.d_model,
        subsampling_conv_channels=model_config.encoder.subsampling_conv_channels,
        feed_forward_expansion_factor=model_config.encoder.ff_expansion_factor,
        n_heads=model_config.encoder.n_heads,
        pos_emb_max_len=model_config.encoder.pos_emb_max_len,
        conv_kernel_size=model_config.encoder.conv_kernel_size,
        subsampling_batch_partitions=subsampling_batch_partitions,
        dtype=encoder_dtype,
        use_bias=model_config.encoder.get("use_bias", True),
        vocab_size=(
            model_config.decoder.num_classes
            if decoder_type == "ctc_greedy_search"
            else None
        ),
    )
    encoder_prefixes = ("encoder.",)
    if decoder_type == "ctc_greedy_search":
        encoder_prefixes += ("projection_output.",)

    encoder_state_dict = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key.startswith(encoder_prefixes)
    )
    if model_config.encoder.get("xscaling", True):
        xscale = model_config.encoder.d_model**0.5
        if encoder_dtype == torch.float16:
            # LN(s*x, eps) = LN(x, eps/s**2). Scale each residual branch by 1/s;
            # the final norm_out restores the original block output scale.
            for projection in (
                "feed_forward1.linear2",
                "self_attn.linear_out",
                "conv.pointwise_conv2",
                "feed_forward2.linear2",
            ):
                for parameter in ("weight", "bias"):
                    key = f"encoder.layers.0.{projection}.{parameter}"
                    if key in encoder_state_dict:
                        encoder_state_dict[key] /= xscale

            for module in encoder.encoder.layers[0].modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.eps /= xscale**2
        else:
            for key in ("encoder.pre_encode.out.weight", "encoder.pre_encode.out.bias"):
                encoder_state_dict[key] *= xscale
    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder.eval()

    if decoder_type == "ctc_greedy_search":
        return encoder, None

    decoder = Decoder(
        vocab_size=model_config.decoder.vocab_size,
        encoder_dim=model_config.joint.jointnet.encoder_hidden,
        decoder_dim=model_config.decoder.prednet.pred_hidden,
        joiner_dim=model_config.joint.jointnet.joint_hidden,
        pred_rnn_layers=model_config.decoder.prednet.pred_rnn_layers,
        num_extra_outputs=model_config.joint.num_extra_outputs,
        dtype=decoder_dtype,
    )
    decoder_state_dict = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key.startswith(
            ("embedding.", "lstm.", "decoder_proj.", "encoder_proj.", "output_proj.")
        )
    )
    decoder.load_state_dict(decoder_state_dict, strict=True)
    decoder.eval()

    return encoder, decoder


def get_subsampling_batch_partitions(
    model_config: DictConfig, args: argparse.Namespace
) -> int:
    """Determine a CASK-safe batch partition count for subsampling.

    TensorRT CASK convolution tactics use signed 32-bit element offsets. The
    first-convolution output is the largest intermediate tensor in the
    three-convolution frontend. This helper derives its shape from the maximum
    audio profile and partitions the entire frontend when that tensor would
    contain more than ``2**31`` elements.

    Parameters
    ----------
    model_config : DictConfig
        Validated source Parakeet configuration containing feature extraction
        and encoder dimensions.
    args : argparse.Namespace
        Validated export arguments containing the batch size and maximum audio
        duration.

    Returns
    -------
    int
        Smallest partition count for which every subsampling batch partition
        stays within the CASK element limit.
    """

    sample_rate = model_config.preprocessor.sample_rate
    hop_length = round(model_config.preprocessor.window_stride * sample_rate)
    max_samples = round(args.max_audio_seconds * sample_rate)
    feature_frames = max_samples // hop_length + 1
    conv1_frames = (feature_frames + 1) // 2
    conv1_features = (model_config.preprocessor.features + 1) // 2
    elements_per_item = (
        model_config.encoder.subsampling_conv_channels * conv1_frames * conv1_features
    )

    cask_element_limit = 1 << 31
    if elements_per_item > cask_element_limit:
        raise ValueError(
            "One Parakeet subsampling item exceeds TensorRT CASK's signed "
            f"32-bit element-offset limit: {elements_per_item} elements."
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
    model_config: DictConfig, args: argparse.Namespace
) -> DictConfig:
    """Build the compact runtime configuration stored in a Parakeet bundle.

    Beam width and blank penalty are saved for both CTC and TDT decoding.
    The penalty is applied by the runtime, not baked into the TensorRT engines.

    Parameters
    ----------
    model_config : DictConfig
        Validated source Parakeet configuration.
    args : argparse.Namespace
        Validated export arguments.

    Returns
    -------
    DictConfig
        Runtime configuration consumed by :class:`fast_gpu_asr.ASR`.
    """

    use_ctc = args.decoder_type == "ctc_greedy_search"
    vocab_size = (
        model_config.decoder.num_classes if use_ctc else model_config.decoder.vocab_size
    )
    decoder_params = {"beam": args.beam, "blank_penalty": args.blank_penalty}
    if not use_ctc:
        decoder_params.update(
            {
                "encoder_dim": model_config.joint.jointnet.encoder_hidden,
                "decoder_dim": model_config.decoder.prednet.pred_hidden,
                "joiner_dim": model_config.joint.jointnet.joint_hidden,
                "pred_rnn_layers": model_config.decoder.prednet.pred_rnn_layers,
                "num_extra_outputs": model_config.joint.num_extra_outputs,
                "max_symbols_per_timestep": model_config.decoding.greedy.max_symbols,
                "tdt_durations": list(model_config.model_defaults.tdt_durations),
            }
        )
    runtime_config = OmegaConf.create(
        {
            "model_type": MODEL_TYPE_PARAKEET,
            "decoder_type": args.decoder_type,
            "model_samplerate": model_config.sample_rate,
            "vocab_size": vocab_size,
            "blank_id": vocab_size,
            "audio_encoder_params": {
                "feature_dim": model_config.preprocessor.features,
                "frame_shift_ms": round(model_config.preprocessor.window_stride * 1000),
                "n_layers": model_config.encoder.n_layers,
                "model_dim": model_config.encoder.d_model,
                "pos_emb_max_len": model_config.encoder.pos_emb_max_len,
                "subsampling_factor": model_config.encoder.subsampling_factor,
                "min_audio_seconds": args.min_audio_seconds,
                "opt_audio_seconds": args.opt_audio_seconds,
                "max_audio_seconds": args.max_audio_seconds,
            },
            "decoder_params": decoder_params,
        }
    )
    validate_model_config(runtime_config)
    return runtime_config


def export_model_to_onnx(
    encoder: ParakeetEncoder,
    decoder: Decoder | None,
    model_config: DictConfig,
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    """Export fixed-batch Parakeet encoder and decoder ONNX graphs.

    The encoder accepts a fixed number of padded waveforms and their valid sample
    counts while allowing waveform duration to vary. The decoder is fully static
    and reserves ``batch_size * beam`` hypothesis slots.

    Parameters
    ----------
    encoder : ParakeetEncoder
        Loaded waveform-to-encoder model.
    decoder : Decoder | None
        Loaded TDT prediction-network and joiner model, or None for CTC.
    model_config : DictConfig
        Validated Parakeet model configuration.
    args : argparse.Namespace
        Export settings, including output directory, batch size, beam, and
        optimal profile duration.

    Returns
    -------
    tuple[Path, Path | None]
        Encoder ONNX path and optional TDT decoder ONNX path. CTC exports only
        the encoder with its projection and log-softmax head.
    """

    encoder_path = args.output_dir / PARAKEET_ONNX_FILE
    decoder_path = args.output_dir / PARAKEET_DECODER_ONNX_FILE

    # Use the profile optimum as the Dynamo example. At the minimum duration,
    # subsampling produces singleton dimensions that ONNX shape propagation may
    # specialize even though waveform time is dynamic.
    audio_samples = round(args.opt_audio_seconds * model_config.sample_rate)
    audio = torch.zeros(args.batch_size, audio_samples, dtype=torch.float32)
    audio_lengths = torch.full((args.batch_size,), audio_samples, dtype=torch.int64)

    logger.info("Exporting the batched Parakeet encoder to %s.", encoder_path)
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

    if args.decoder_type == "ctc_greedy_search":
        return encoder_path, None
    if decoder is None:
        raise RuntimeError("The Parakeet TDT decoder was not initialized.")

    decoder_batch = args.batch_size * args.beam
    pred_rnn_layers = model_config.decoder.prednet.pred_rnn_layers
    decoder_dim = model_config.decoder.prednet.pred_hidden
    encoder_dim = model_config.joint.jointnet.encoder_hidden
    decoder_dtype = decoder.output_proj.weight.dtype

    logger.info("Exporting the batched TDT decoder to %s.", decoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            decoder,
            (
                torch.zeros(decoder_batch, encoder_dim, dtype=decoder_dtype),
                torch.zeros(decoder_batch, 1, dtype=torch.int32),
                torch.zeros(
                    pred_rnn_layers, decoder_batch, decoder_dim, dtype=decoder_dtype
                ),
                torch.zeros(
                    pred_rnn_layers, decoder_batch, decoder_dim, dtype=decoder_dtype
                ),
            ),
            decoder_path,
            input_names=(
                "encoder_output",
                "targets",
                "input_states_1",
                "input_states_2",
            ),
            output_names=(
                "token_log_probs",
                "duration_log_probs",
                "output_states_1",
                "output_states_2",
            ),
            opset_version=ONNX_OPSET_VERSION,
        )

    return encoder_path, decoder_path


def export_parakeet(args: argparse.Namespace) -> None:
    """Run the complete offline Parakeet TensorRT export.

    The output directory is deleted and recreated before the source archive is
    extracted into a temporary directory. Its tokenizer and a compact runtime
    configuration are written to the output directory, while the checkpoint is
    converted into an encoder ONNX graph and, for TDT, a separate decoder graph.
    TensorRT engines are then built using the requested fixed batch and
    duration profile. Intermediate ONNX artifacts are retained only in debug
    mode.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed export arguments from :func:`parse_args`.

    Raises
    ------
    FileNotFoundError
        Raised when the Parakeet archive or a required archive member is missing.
    ValueError
        Raised when the model configuration, decoding settings, or TensorRT
        profile is unsupported, or the output directory contains the archive.
    RuntimeError
        Raised when checkpoint loading, ONNX conversion, or TensorRT engine
        construction fails.
    """

    if args.decoder_type in ("transducer_greedy_search", "ctc_greedy_search"):
        if args.beam != 1:
            logger.warning(
                "Overriding beam=%s with beam=1 because %s requires beam 1.",
                args.beam,
                args.decoder_type,
            )
        args.beam = 1

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Parakeet model archive does not exist: {args.model_path}."
        )
    if args.model_path.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError(
            f"Output directory {args.output_dir} contains required source file "
            f"{args.model_path}."
        )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="fast_gpu_asr_parakeet_") as tmp_dir:
        config_path, checkpoint_path, tokenizer_path = extract_parakeet_archive(
            args.model_path, Path(tmp_dir)
        )
        model_config = OmegaConf.load(config_path)
        validate_parakeet(model_config, args)

        logger.info("Loading Parakeet checkpoint %s.", checkpoint_path)
        state_dict: OrderedDict[str, torch.Tensor] = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        state_dict = adjust_state_dict(state_dict)
        subsampling_batch_partitions = get_subsampling_batch_partitions(
            model_config, args
        )
        if subsampling_batch_partitions > 1:
            logger.info(
                "Splitting Parakeet convolutional subsampling into %s batch "
                "partitions to stay within TensorRT's CASK element limit.",
                subsampling_batch_partitions,
            )
        encoder, decoder = make_model(
            model_config,
            state_dict,
            subsampling_batch_partitions,
            PRECISION_DTYPES[args.encoder_precision],
            PRECISION_DTYPES[args.decoder_precision],
            args.decoder_type,
        )
        del state_dict

        shutil.copyfile(tokenizer_path, args.output_dir / TOKENIZER_FILE)
        runtime_config = make_runtime_config(model_config, args)
        logger.info("Parakeet runtime config:\n%s", OmegaConf.to_yaml(runtime_config))
        OmegaConf.save(runtime_config, args.output_dir / MODEL_CONFIG_FILE)

        encoder_onnx_path, decoder_onnx_path = export_model_to_onnx(
            encoder, decoder, model_config, args
        )
        del encoder, decoder

    audio_samples = (
        round(args.min_audio_seconds * model_config.sample_rate),
        round(args.opt_audio_seconds * model_config.sample_rate),
        round(args.max_audio_seconds * model_config.sample_rate),
    )
    encoder_profiles = {
        "audio": tuple((args.batch_size, samples) for samples in audio_samples)
    }
    build_tensorrt_engine(
        encoder_onnx_path,
        args.output_dir / PARAKEET_TENSORRT_FILE,
        encoder_profiles,
        args.optimization_level,
    )
    if decoder_onnx_path is not None:
        build_tensorrt_engine(
            decoder_onnx_path,
            args.output_dir / PARAKEET_DECODER_TENSORRT_FILE,
            {},
            args.optimization_level,
        )

    if not args.debug:
        remove_onnx_artifacts(encoder_onnx_path)
        if decoder_onnx_path is not None:
            remove_onnx_artifacts(decoder_onnx_path)

    validate_model(args.output_dir, OmegaConf.load(args.output_dir / MODEL_CONFIG_FILE))

    logger.info("Parakeet TensorRT export completed in %s.", args.output_dir)


def main() -> None:
    """Configure logging and run the command-line Parakeet exporter."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    export_parakeet(parse_args())


if __name__ == "__main__":
    main()
