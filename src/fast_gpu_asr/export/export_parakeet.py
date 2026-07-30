#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Export NVIDIA Parakeet TDT checkpoints for batched TensorRT inference.

The exporter reads a model archive, reconstructs the condensed Parakeet encoder
and TDT decoder, exports fixed-batch ONNX graphs, and builds TensorRT engines.
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
    MODEL_CONFIG_FILE,
    MODEL_TYPE_PARAKEET,
    ONNX_OPSET_VERSION,
    PARAKEET_ONNX_FILE,
    PARAKEET_TENSORRT_FILE,
    TDT_DECODER_ONNX_FILE,
    TDT_DECODER_TENSORRT_FILE,
    TRANSDUCER_DECODER_TYPES,
)
from ..utils import validate_runtime_config
from .export_utils import (
    build_tensorrt_engine,
    remove_onnx_artifacts,
    validate_parakeet,
)
from .model.parakeet.decoder import Decoder
from .model.parakeet.parakeet import ParakeetTDTEncoder

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse Parakeet TensorRT export arguments.

    Returns
    -------
    argparse.Namespace
        Command-line values consumed by :func:`export_parakeet`.
    """

    parser = argparse.ArgumentParser(
        description="Export an NVIDIA Parakeet TDT .nemo model to TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the source Parakeet TDT .nemo archive.",
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
        choices=TRANSDUCER_DECODER_TYPES,
        help="Decoder included in the exported Parakeet bundle.",
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


def extract_member(archive: tarfile.TarFile, filename: str, output_dir: Path) -> Path:
    """Extract one required file from a Parakeet archive.

    Parameters
    ----------
    archive : tarfile.TarFile
        Open Parakeet archive.
    filename : str
        Required member name or path suffix.
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
    """

    member = next(
        (
            candidate
            for candidate in archive.getmembers()
            if candidate.name.endswith(filename)
        ),
        None,
    )
    if member is None:
        raise FileNotFoundError(f"Missing {filename} in Parakeet archive.")

    source = archive.extractfile(member)
    if source is None:
        raise FileNotFoundError(f"Unable to read {member.name} from Parakeet archive.")

    output_path = output_dir / Path(filename).name
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
        tokenizer_path = str(model_config.tokenizer.model_path)
        if tokenizer_path.startswith("nemo:"):
            tokenizer_path = tokenizer_path.removeprefix("nemo:")
        tokenizer_path = extract_member(archive, Path(tokenizer_path).name, output_dir)

    return config_path, checkpoint_path, tokenizer_path


def adjust_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert Parakeet checkpoint keys to the condensed model layout.

    Prediction-network and joiner prefixes are renamed to match :class:`Decoder`.
    Pointwise Conv1d weights are squeezed to the matrix shape expected by the
    equivalent linear layers in the condensed encoder.

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.Tensor]
        State dictionary extracted from ``model_weights.ckpt``.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Converted state dictionary compatible with the local export modules.

    Raises
    ------
    ValueError
        Raised when a pointwise convolution does not have a singleton kernel
        dimension and therefore cannot be represented by a linear layer.
    """

    adjusted_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        key = (
            key.replace("decoder.prediction.embed", "embedding")
            .replace("decoder.prediction.dec_rnn.lstm", "lstm")
            .replace("joint.pred", "decoder_proj")
            .replace("joint.enc", "encoder_proj")
            .replace("joint.joint_net.2", "output_proj")
            .replace("encoder.pre_encode.conv.0", "encoder.pre_encode.conv1")
            .replace("encoder.pre_encode.conv.2", "encoder.pre_encode.conv2")
            .replace("encoder.pre_encode.conv.3", "encoder.pre_encode.pointwise_conv1")
            .replace("encoder.pre_encode.conv.5", "encoder.pre_encode.conv3")
            .replace("encoder.pre_encode.conv.6", "encoder.pre_encode.pointwise_conv2")
        )
        if key.endswith(
            ("conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight"),
        ):
            if value.ndim != 3 or value.size(2) != 1:
                raise ValueError(
                    f"Expected pointwise Conv1d weight {key} to have shape "
                    f"(out_channels, in_channels, 1), got {tuple(value.shape)}.",
                )
            value = value.squeeze(2)
        adjusted_state_dict[key] = value

    return adjusted_state_dict


def make_model(
    model_config: DictConfig, state_dict: OrderedDict[str, torch.Tensor]
) -> tuple[ParakeetTDTEncoder, Decoder]:
    """Construct condensed Parakeet modules and load checkpoint weights.

    Parameters
    ----------
    model_config : DictConfig
        Validated Parakeet model configuration.
    state_dict : OrderedDict[str, torch.Tensor]
        Converted checkpoint state dictionary from :func:`adjust_state_dict`.

    Returns
    -------
    tuple[ParakeetTDTEncoder, Decoder]
        Evaluation-mode waveform encoder and TDT decoder.

    Raises
    ------
    RuntimeError
        Raised when checkpoint keys or tensor shapes do not exactly match the
        reconstructed architecture.
    """

    encoder = ParakeetTDTEncoder(
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
    )
    encoder_state_dict = OrderedDict(
        (key, value) for key, value in state_dict.items() if key.startswith("encoder.")
    )
    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder.eval()

    decoder = Decoder(
        vocab_size=model_config.decoder.vocab_size,
        encoder_dim=model_config.joint.jointnet.encoder_hidden,
        decoder_dim=model_config.decoder.prednet.pred_hidden,
        joiner_dim=model_config.joint.jointnet.joint_hidden,
        pred_rnn_layers=model_config.decoder.prednet.pred_rnn_layers,
        num_extra_outputs=model_config.joint.num_extra_outputs,
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


def make_runtime_config(
    model_config: DictConfig, args: argparse.Namespace
) -> DictConfig:
    """Build the compact runtime configuration stored in a Parakeet bundle.

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

    runtime_config = OmegaConf.create(
        {
            "model_type": MODEL_TYPE_PARAKEET,
            "decoder_type": args.decoder_type,
            "model_samplerate": model_config.sample_rate,
            "vocab_size": model_config.decoder.vocab_size,
            "blank_id": model_config.decoder.vocab_size,
            "audio_encoder_params": {
                "feature_dim": model_config.preprocessor.features,
                "n_layers": model_config.encoder.n_layers,
                "model_dim": model_config.encoder.d_model,
                "pos_emb_max_len": model_config.encoder.pos_emb_max_len,
                "subsampling_factor": model_config.encoder.subsampling_factor,
                "min_audio_seconds": args.min_audio_seconds,
                "opt_audio_seconds": args.opt_audio_seconds,
                "max_audio_seconds": args.max_audio_seconds,
            },
            "decoder_params": {
                "encoder_dim": model_config.joint.jointnet.encoder_hidden,
                "decoder_dim": model_config.decoder.prednet.pred_hidden,
                "joiner_dim": model_config.joint.jointnet.joint_hidden,
                "pred_rnn_layers": model_config.decoder.prednet.pred_rnn_layers,
                "num_extra_outputs": model_config.joint.num_extra_outputs,
                "beam": args.beam,
                "blank_penalty": 0.0,
                "max_symbols_per_timestep": model_config.decoding.greedy.max_symbols,
                "tdt_durations": list(model_config.model_defaults.tdt_durations),
            },
        },
    )
    validate_runtime_config(runtime_config)
    return runtime_config


def export_model_to_onnx(
    encoder: ParakeetTDTEncoder,
    decoder: Decoder,
    model_config: DictConfig,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Export fixed-batch Parakeet encoder and decoder ONNX graphs.

    The encoder accepts a fixed number of padded waveforms while allowing the
    waveform duration to vary. The decoder is fully static and reserves
    ``batch_size * beam`` hypothesis slots.

    Parameters
    ----------
    encoder : ParakeetTDTEncoder
        Loaded waveform-to-encoder model.
    decoder : Decoder
        Loaded TDT prediction-network and joiner model.
    model_config : DictConfig
        Validated Parakeet model configuration.
    args : argparse.Namespace
        Export settings, including output directory, batch size, beam, and
        optimal profile duration.

    Returns
    -------
    tuple[Path, Path]
        Paths to the encoder and decoder ONNX graphs.
    """

    encoder_path = args.output_dir / PARAKEET_ONNX_FILE
    decoder_path = args.output_dir / TDT_DECODER_ONNX_FILE

    dummy_samples = round(args.opt_audio_seconds * model_config.sample_rate)
    audio = torch.zeros(args.batch_size, dummy_samples, dtype=torch.float32)
    audio_lengths = torch.full((args.batch_size,), dummy_samples, dtype=torch.int32)

    logger.info("Exporting the batched Parakeet encoder to %s.", encoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            torch.jit.script(encoder),
            (audio, audio_lengths),
            str(encoder_path),
            dynamo=False,
            external_data=True,
            dynamic_axes={
                "audio": {1: "num_samples"},
                "encoder_output": {1: "num_encoder_frames"},
            },
            input_names=("audio", "audio_lengths"),
            output_names=("encoder_output", "encoder_output_lengths"),
            opset_version=ONNX_OPSET_VERSION,
        )

    decoder_batch = args.batch_size * args.beam
    pred_rnn_layers = model_config.decoder.prednet.pred_rnn_layers
    decoder_dim = model_config.decoder.prednet.pred_hidden
    encoder_dim = model_config.joint.jointnet.encoder_hidden

    logger.info("Exporting the batched TDT decoder to %s.", decoder_path)
    with torch.inference_mode():
        torch.onnx.export(
            torch.jit.script(decoder),
            (
                torch.zeros(decoder_batch, encoder_dim, dtype=torch.float32),
                torch.zeros(decoder_batch, 1, dtype=torch.int32),
                torch.zeros(
                    pred_rnn_layers, decoder_batch, decoder_dim, dtype=torch.float32
                ),
                torch.zeros(
                    pred_rnn_layers, decoder_batch, decoder_dim, dtype=torch.float32
                ),
            ),
            str(decoder_path),
            dynamo=False,
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

    The source archive is extracted into a temporary directory. Its tokenizer
    and a compact runtime configuration are written to the output directory,
    while the checkpoint is converted into encoder and decoder ONNX graphs.
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
        Raised when the model configuration or TensorRT profile is unsupported.
    RuntimeError
        Raised when checkpoint loading, ONNX conversion, or TensorRT engine
        construction fails.
    """

    if args.decoder_type == "transducer_greedy_search":
        if args.beam != 1:
            logger.warning(
                "Overriding beam=%s with beam=1 for %s.", args.beam, args.decoder_type
            )
        args.beam = 1

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Parakeet model archive does not exist: {args.model_path}.",
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
        encoder, decoder = make_model(model_config, state_dict)
        del state_dict

        shutil.copyfile(tokenizer_path, args.output_dir / "bpe.model")
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
        "audio": tuple((args.batch_size, samples) for samples in audio_samples),
    }
    workspace_bytes = round(args.workspace_gib * 2**30)
    build_tensorrt_engine(
        encoder_onnx_path,
        args.output_dir / PARAKEET_TENSORRT_FILE,
        encoder_profiles,
        workspace_bytes,
    )
    build_tensorrt_engine(
        decoder_onnx_path,
        args.output_dir / TDT_DECODER_TENSORRT_FILE,
        {},
        workspace_bytes,
    )

    if not args.debug:
        remove_onnx_artifacts(encoder_onnx_path)
        remove_onnx_artifacts(decoder_onnx_path)

    logger.info("Parakeet TensorRT export completed in %s.", args.output_dir)


def main() -> None:
    """Configure logging and run the command-line Parakeet exporter."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    export_parakeet(parse_args())


if __name__ == "__main__":
    main()
