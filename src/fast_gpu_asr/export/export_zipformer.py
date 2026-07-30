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
    ZIPFORMER_DECODER_ONNX_FILE,
    ZIPFORMER_DECODER_TENSORRT_FILE,
    ZIPFORMER_ONNX_FILE,
    ZIPFORMER_TENSORRT_FILE,
)
from ..utils import validate_runtime_config
from .export_utils import (
    build_tensorrt_engine,
    prepare_onnx_for_tensorrt,
    remove_onnx_artifacts,
    validate_zipformer,
)
from .model.zipformer.decoder import Decoder
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
        help="Modified-beam hypotheses per utterance; greedy search forces beam 1.",
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
        "--workspace-gib",
        type=float,
        default=8.0,
        help="TensorRT workspace limit in GiB.",
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
            .replace("subsampling.conv.4.weight", "subsampling.conv.2.weight")
            .replace("subsampling.conv.4.bias", "subsampling.conv.2.bias")
            .replace("subsampling.conv.7.weight", "subsampling.conv.4.weight")
            .replace("subsampling.conv.7.bias", "subsampling.conv.4.bias")
            .replace("downsample.bias", "downsample.weights")
            .replace("downsample_output.bias", "downsample_output.weights")
            .replace("encoder.layers", "layers")
            .replace("encoders.0", "encoder_1")
            .replace("encoders.1", "encoder_2")
            .replace("encoders.2", "encoder_3")
            .replace("encoders.3", "encoder_4")
            .replace("encoders.4", "encoder_5")
            .replace("encoders.5", "encoder_6")
            .replace("joiner.encoder_proj", "projection_output")
        )
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
        if bypass_scale_pattern.fullmatch(key):
            continue
        adjusted_state_dict[key] = value

    encoder_1_dim = adjusted_state_dict[
        "encoder_1.layers.0.self_attn_weights.in_proj.weight"
    ].size(1)
    adjusted_state_dict["encoder_1.out_combiner.bypass_scale"] = torch.ones(
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
) -> tuple[Zipformer2, Decoder | None]:
    """Construct condensed Zipformer modules and load checkpoint weights.

    Parameters
    ----------
    model_config : DictConfig
        Validated published model configuration.
    state_dict : OrderedDict[str, torch.Tensor]
        Converted checkpoint state dictionary from :func:`adjust_state_dict`.
    vocab_size : int
        Transducer output vocabulary size, excluding the unknown token.
    pos_emb_max_len : int
        Maximum positional encoding length compiled into the encoder.
    decoder_type : str
        Decoder whose output head is included in the exported bundle.

    Returns
    -------
    tuple[Zipformer2, Decoder | None]
        Evaluation-mode waveform encoder and, for transducer modes, the merged
        decoder and joiner.

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
        subsample_layer1_channels=state_dict["subsampling.conv.0.weight"].size(0),
        subsample_layer2_channels=state_dict["subsampling.conv.2.weight"].size(0),
        subsample_layer3_channels=state_dict["subsampling.conv.4.weight"].size(0),
        encoder_dims=encoder_dims,
        num_encoder_layers=num_encoder_layers,
        downsampling_factors=downsampling_factors,
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
        return encoder, None

    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_dim=model_params.decoder_dim,
        joiner_dim=model_params.joiner_dim,
        context_size=model_params.context_size,
    )
    decoder_state_dict = OrderedDict(
        (
            key.replace("decoder.embedding", "embedding")
            .replace("decoder.conv", "conv")
            .replace("joiner.decoder_proj", "decoder_proj")
            .replace("joiner.output_linear", "output_proj"),
            value,
        )
        for key, value in state_dict.items()
        if key.startswith(
            (
                "decoder.embedding.",
                "decoder.conv.",
                "joiner.decoder_proj.",
                "joiner.output_linear.",
            )
        )
    )
    decoder.load_state_dict(decoder_state_dict, strict=True)
    decoder.eval()

    return encoder, decoder


def make_runtime_config(
    model_config: DictConfig,
    vocab_size: int,
    pos_emb_max_len: int,
    args: argparse.Namespace,
) -> DictConfig:
    """Build the compact runtime configuration stored in a Zipformer bundle.

    Parameters
    ----------
    model_config : DictConfig
        Validated source Zipformer configuration.
    vocab_size : int
        Decoder vocabulary size excluding the SentencePiece unknown token.
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
            "model_samplerate": model_config.feature_opts.frame_opts.samp_freq,
            "vocab_size": vocab_size,
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
                "subsampling_factor": model_params.subsampling_factor,
                "use_ctc": args.decoder_type == "ctc_greedy_search",
                "min_audio_seconds": args.min_audio_seconds,
                "opt_audio_seconds": args.opt_audio_seconds,
                "max_audio_seconds": args.max_audio_seconds,
            },
            "decoder_params": decoder_params,
        },
    )
    validate_runtime_config(runtime_config)
    return runtime_config


def export_model_to_onnx(
    encoder: Zipformer2,
    decoder: Decoder | None,
    model_config: DictConfig,
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    """Export fixed-batch Zipformer encoder and decoder ONNX graphs.

    The encoder accepts a fixed number of padded waveforms while allowing their
    shared sample capacity to vary. The decoder is fully static and reserves
    ``batch_size * beam`` hypothesis slots.

    Parameters
    ----------
    encoder : Zipformer2
        Loaded waveform-to-encoder model.
    decoder : Decoder | None
        Loaded stateless transducer decoder and joiner, or ``None`` for CTC.
    model_config : DictConfig
        Validated published model configuration.
    args : argparse.Namespace
        Export settings, including output directory, fixed batch size, beam,
        and optimal profile duration.

    Returns
    -------
    tuple[Path, Path | None]
        Paths to the encoder and optional transducer decoder ONNX graphs.
    """

    encoder_path = args.output_dir / ZIPFORMER_ONNX_FILE
    decoder_path = args.output_dir / ZIPFORMER_DECODER_ONNX_FILE

    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    audio_samples = round(args.opt_audio_seconds * sample_rate)
    audio = torch.zeros(args.batch_size, audio_samples, dtype=torch.float32)
    audio_lengths = torch.full((args.batch_size,), audio_samples, dtype=torch.int32)

    logger.info("Exporting the batched Zipformer encoder to %s.", encoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            torch.jit.script(encoder),
            (audio, audio_lengths),
            encoder_path,
            dynamo=False,
            dynamic_axes={
                "audio": {1: "num_samples"},
                "encoder_output": {1: "num_encoder_frames"},
            },
            input_names=("audio", "audio_lengths"),
            output_names=("encoder_output", "encoder_output_lengths"),
            opset_version=ONNX_OPSET_VERSION,
        )

    if decoder is None:
        return encoder_path, None

    decoder_batch = args.batch_size
    if args.decoder_type == "transducer_modified_beam_search":
        decoder_batch *= args.beam
    logger.info("Exporting the batched Zipformer decoder to %s.", decoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            torch.jit.script(decoder),
            (
                torch.zeros(
                    decoder_batch,
                    model_config.model_params.context_size,
                    dtype=torch.int32,
                ),
                torch.zeros(
                    decoder_batch,
                    model_config.model_params.joiner_dim,
                    dtype=torch.float32,
                ),
            ),
            decoder_path,
            dynamo=False,
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
    """

    if args.decoder_type in ("transducer_greedy_search", "ctc_greedy_search"):
        if args.beam != 1:
            logger.warning(
                "Overriding beam=%s with beam=1 for %s.", args.beam, args.decoder_type
            )
        args.beam = 1

    model_dir = args.model_path.parent
    config_path = model_dir / "config.yaml"
    tokenizer_path = model_dir / "bpe.model"
    for required_path in (config_path, args.model_path, tokenizer_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing required model file: {required_path}.")

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
    encoder, decoder = make_model(
        model_config,
        state_dict,
        vocab_size,
        pos_emb_max_len,
        args.decoder_type,
    )
    del state_dict

    shutil.copyfile(tokenizer_path, args.output_dir / "bpe.model")
    runtime_config = make_runtime_config(
        model_config, vocab_size, pos_emb_max_len, args
    )
    logger.info("Zipformer runtime config:\n%s", OmegaConf.to_yaml(runtime_config))
    OmegaConf.save(runtime_config, args.output_dir / MODEL_CONFIG_FILE)

    encoder_onnx_path, decoder_onnx_path = export_model_to_onnx(
        encoder, decoder, model_config, args
    )
    del encoder, decoder

    encoder_tensorrt_onnx_path = args.output_dir / "zipformer-tensorrt.onnx"
    decoder_tensorrt_onnx_path = args.output_dir / "decoder-tensorrt.onnx"
    prepare_onnx_for_tensorrt(encoder_onnx_path, encoder_tensorrt_onnx_path)
    if decoder_onnx_path is not None:
        prepare_onnx_for_tensorrt(decoder_onnx_path, decoder_tensorrt_onnx_path)

    audio_samples = (
        round(args.min_audio_seconds * sample_rate),
        round(args.opt_audio_seconds * sample_rate),
        round(args.max_audio_seconds * sample_rate),
    )
    encoder_profiles = {
        "audio": tuple((args.batch_size, samples) for samples in audio_samples),
    }
    workspace_bytes = round(args.workspace_gib * 2**30)
    build_tensorrt_engine(
        encoder_tensorrt_onnx_path,
        args.output_dir / ZIPFORMER_TENSORRT_FILE,
        encoder_profiles,
        workspace_bytes,
    )
    if decoder_onnx_path is not None:
        build_tensorrt_engine(
            decoder_tensorrt_onnx_path,
            args.output_dir / ZIPFORMER_DECODER_TENSORRT_FILE,
            {},
            workspace_bytes,
        )

    if not args.debug:
        remove_onnx_artifacts(encoder_onnx_path)
        remove_onnx_artifacts(encoder_tensorrt_onnx_path)
        if decoder_onnx_path is not None:
            remove_onnx_artifacts(decoder_onnx_path)
            remove_onnx_artifacts(decoder_tensorrt_onnx_path)

    logger.info("Zipformer TensorRT export completed in %s.", args.output_dir)


def main() -> None:
    """Configure logging and run the command-line Zipformer exporter."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    export_zipformer(parse_args())


if __name__ == "__main__":
    main()
